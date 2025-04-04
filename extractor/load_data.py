import os
import json
import asyncio
import logging
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass, field

import yaml
from dotenv import load_dotenv
from Levenshtein import ratio # Используем ratio для нормализованного значения от 0 до 1
from langchain_openai import ChatOpenAI
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# --- Конфигурация логирования ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Загрузка конфигурации и переменных окружения ---
load_dotenv()

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Загружает конфигурацию из YAML файла."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logging.info(f"Конфигурация успешно загружена из {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f"Файл конфигурации {config_path} не найден.")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Ошибка парсинга YAML файла {config_path}: {e}")
        raise

CONFIG = load_config()

# --- Датаклассы для структурирования данных ---
@dataclass
class Entity:
    name: str
    descriptions: List[str] = field(default_factory=list)
    # Можно добавить другие атрибуты сущности, если нужно (например, type)

@dataclass
class Relationship:
    source: str
    target: str
    description: str
    # Можно добавить другие атрибуты связи (например, type)

# --- Инициализация LLM ---
def initialize_llm(model_key: str, config: Dict[str, Any]) -> ChatOpenAI:
    """Инициализирует модель ChatOpenAI на основе конфигурации."""
    try:
        llm = ChatOpenAI(
            model=os.getenv("RAG_LLM_MODEL") or config["llm"][model_key], # Приоритет у переменной окружения
            base_url=os.getenv('RAG_BASE_URL'),
            temperature=config["llm"]["temperature"],
            top_p=config["llm"]["top_p"],
            max_tokens=config["llm"]["max_tokens"],
            timeout=config["llm"]["request_timeout"],
            openai_api_key=os.getenv('OPENAI_API_KEY') # Убедитесь, что ключ передается
        )
        logging.info(f"LLM '{config['llm'][model_key]}' инициализирована с URL: {os.getenv('RAG_BASE_URL')}")
        return llm
    except Exception as e:
        logging.error(f"Ошибка инициализации LLM: {e}")
        raise

# Основная LLM для извлечения сущностей/связей
llm_extractor = initialize_llm("chunk_processing_model", CONFIG) | SimpleJsonOutputParser()
# LLM для суммаризации (может быть та же модель)
llm_summarizer = initialize_llm("summarization_model", CONFIG) # Без JSON парсера, т.к. нужен просто текст

# --- Функции загрузки и обработки документов ---
def load_documents(files_dir: str) -> List[Document]:
    """Загружает документы из указанной директории."""
    loaded_docs = []
    logging.info(f"Загрузка документов из директории: {files_dir}")
    if not os.path.isdir(files_dir):
        logging.error(f"Директория не найдена: {files_dir}")
        return []
    try:
        for filename in os.listdir(files_dir):
            file_path = os.path.join(files_dir, filename)
            if filename.endswith(".txt"):
                loader = TextLoader(file_path, encoding='utf-8') # Указываем кодировку
            elif filename.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            elif filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            else:
                continue # Пропускаем неподдерживаемые файлы
            logging.debug(f"Загрузка файла: {filename}")
            loaded_docs.extend(loader.load())
        logging.info(f"Загружено {len(loaded_docs)} документов.")
    except Exception as e:
        logging.error(f"Ошибка при загрузке документов: {e}")
        # Можно решить, прерывать ли выполнение или продолжать с тем, что есть
    return loaded_docs

def split_documents(documents: List[Document], config: Dict[str, Any]) -> List[Document]:
    """Разбивает документы на чанки."""
    if not documents:
        logging.warning("Нет документов для разбиения на чанки.")
        return []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["processing"]["chunk_size"],
        chunk_overlap=config["processing"]["chunk_overlap"]
    )
    chunks = text_splitter.split_documents(documents)
    logging.info(f"Документы разбиты на {len(chunks)} чанков.")
    return chunks

# --- Асинхронная обработка чанков ---
async def process_chunk(chunk: Document, llm_chain: Any, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Асинхронно обрабатывает один чанк текста с помощью LLM."""
    system_prompt = config["prompts"]["extraction_system_prompt"]
    user_template = config["prompts"]["extraction_user_template"]
    prompt = user_template.format(system_message=system_prompt, prompt=chunk.page_content)
    try:
        # Используем ainvoke для асинхронного вызова
        response = await llm_chain.ainvoke(prompt)
        # Валидация: Проверяем, что ответ - это словарь
        if isinstance(response, dict):
             # Проверяем наличие ключей entities и relationships (даже если они пустые списки)
            if "entities" in response and "relationships" in response:
                logging.debug(f"Чанк обработан успешно. Найдено сущностей: {len(response.get('entities', []))}, связей: {len(response.get('relationships', []))}")
                return response
            else:
                logging.warning(f"Ответ LLM для чанка не содержит ключи 'entities' и 'relationships': {response}")
                return None # Возвращаем None если структура некорректна
        else:
             # Если вернулась строка (например, ошибка парсинга JSON внутри SimpleJsonOutputParser)
             logging.warning(f"Ответ LLM для чанка не является словарем (возможно, ошибка JSON): {response}")
             # Попытка исправить JSON, если он обрамлен ```json ... ```
             if isinstance(response, str) and response.strip().startswith("```json"):
                 clean_response = response.strip()[7:-3].strip() # Убираем обертку
                 try:
                     parsed_response = json.loads(clean_response)
                     if isinstance(parsed_response, dict) and "entities" in parsed_response and "relationships" in parsed_response:
                         logging.info("Успешно исправлен и распарсен JSON из строки.")
                         return parsed_response
                     else:
                          logging.warning(f"Исправленный JSON не имеет нужной структуры: {parsed_response}")
                          return None
                 except json.JSONDecodeError as json_err:
                     logging.error(f"Ошибка декодирования исправленного JSON: {json_err}. Исходная строка: {clean_response}")
                     return None
             return None
    except Exception as e:
        logging.error(f"Ошибка при обработке чанка LLM: {e}", exc_info=True) # Добавляем traceback
        return None

# --- Слияние дубликатов сущностей ---
def merge_entities(
    all_entities: List[Dict[str, Any]], threshold: float
) -> Tuple[Dict[str, Entity], Dict[str, str]]:
    """
    Объединяет дублирующиеся сущности на основе схожести имен (Левенштейн).

    Args:
        all_entities: Список всех извлеченных сущностей (словари).
        threshold: Порог схожести (0.0 до 1.0).

    Returns:
        Кортеж:
            - Словарь уникальных сущностей {canonical_name: Entity}.
            - Словарь для переименования {old_name: canonical_name}.
    """
    unique_entities: Dict[str, Entity] = {}
    name_mapping: Dict[str, str] = {} # Карта для обновления связей {old_name: new_name}

    for entity_data in all_entities:
        if not isinstance(entity_data, dict):
            logging.warning(f"Пропуск некорректных данных сущности: {entity_data}")
            continue
        name = entity_data.get("name")
        description = entity_data.get("description")

        if not name or not isinstance(name, str) or not name.strip():
            logging.warning(f"Пропуск сущности без имени или с некорректным именем: {entity_data}")
            continue
        name = name.strip() # Убираем лишние пробелы

        if not description or not isinstance(description, str) or not description.strip():
            # Можно пропустить или добавить с пустым описанием
            description = "" # Используем пустую строку, если описание отсутствует

        matched = False
        # Ищем похожую сущность среди уже добавленных уникальных
        for canonical_name in list(unique_entities.keys()): # list() для безопасной итерации при возможном изменении
             # Проверяем прямое совпадение или высокую схожесть
            similarity = ratio(name.lower(), canonical_name.lower()) # Сравниваем в нижнем регистре
            if similarity >= threshold:
                logging.debug(f"Обнаружено слияние: '{name}' -> '{canonical_name}' (Схожесть: {similarity:.2f})")
                # Добавляем описание к существующей сущности
                if description: # Добавляем только непустые описания
                    unique_entities[canonical_name].descriptions.append(description)
                # Обновляем карту имен, если текущее имя еще не было смаплено
                if name != canonical_name and name not in name_mapping:
                    name_mapping[name] = canonical_name
                matched = True
                break # Нашли совпадение, переходим к следующей сущности из all_entities

        # Если совпадений не найдено, добавляем как новую уникальную сущность
        if not matched:
            logging.debug(f"Добавлена новая уникальная сущность: '{name}'")
            unique_entities[name] = Entity(name=name, descriptions=[desc for desc in [description] if desc]) # Сохраняем описание в списке
            name_mapping[name] = name # Сама на себя для консистентности карты

    logging.info(f"Слияние сущностей завершено. Уникальных сущностей: {len(unique_entities)}. Обнаружено слияний: {len(name_mapping) - len(unique_entities)}.")
    return unique_entities, name_mapping

# --- Суммаризация описаний ---
async def summarize_description(
    entity_name: str,
    descriptions: List[str],
    llm_summarizer: Any, # Тип Any для LangChain объектов
    config: Dict[str, Any]
) -> str:
    """Асинхронно суммирует список описаний для одной сущности."""
    if not descriptions:
        return ""
    if len(descriptions) == 1:
        return descriptions[0] # Возвращаем единственное описание как есть

    text_to_summarize = "\n---\n".join(descriptions) # Объединяем описания разделителем
    system_prompt = config["prompts"]["summarization_system_prompt"].format(entity_name=entity_name)
    user_template = config["prompts"]["summarization_user_template"]
    prompt = user_template.format(system_message=system_prompt, text_to_summarize=text_to_summarize)

    try:
        # Используем ainvoke для асинхронного вызова
        summary_response = await llm_summarizer.ainvoke(prompt)

        # LangChain может вернуть AIMessage или строку, извлекаем контент
        if hasattr(summary_response, 'content'):
            summary = summary_response.content.strip()
        elif isinstance(summary_response, str):
            summary = summary_response.strip()
        else:
             logging.warning(f"Неожиданный тип ответа от LLM-суммаризатора для '{entity_name}': {type(summary_response)}")
             summary = ". ".join(descriptions) # Возвращаем объединенные описания в случае ошибки

        logging.debug(f"Описание для '{entity_name}' суммировано.")
        return summary
    except Exception as e:
        logging.error(f"Ошибка при суммировании описания для '{entity_name}': {e}", exc_info=True)
        # В случае ошибки возвращаем просто объединенные описания
        return ". ".join(descriptions)

# --- Основная функция ---
async def main():
    """Основной асинхронный процесс генерации графа."""
    data_path = os.getenv("DATA_PATH")
    if not data_path:
        logging.error("Переменная окружения DATA_PATH не установлена.")
        return

    # 1. Загрузка и разбиение документов
    documents = load_documents(data_path)
    chunks = split_documents(documents, CONFIG)
    if not chunks:
        logging.warning("Нет чанков для обработки.")
        return

    # 2. Асинхронная обработка чанков
    logging.info(f"Начало асинхронной обработки {len(chunks)} чанков...")
    tasks = [process_chunk(chunk, llm_extractor, CONFIG) for chunk in chunks]
    llm_results = await asyncio.gather(*tasks)
    logging.info("Обработка чанков завершена.")

    # Фильтруем None результаты (ошибки или некорректный формат)
    valid_results = [res for res in llm_results if res is not None]
    logging.info(f"Получено {len(valid_results)} валидных ответов от LLM.")

    # 3. Сбор всех сущностей и связей
    all_entities_raw: List[Dict[str, Any]] = []
    all_relationships_raw: List[Dict[str, Any]] = []
    for res in valid_results:
        # Дополнительная проверка на тип перед извлечением
        if isinstance(res.get("entities"), list):
             all_entities_raw.extend(ent for ent in res["entities"] if isinstance(ent, dict) and ent.get("name")) # Проверяем формат каждой сущности
        if isinstance(res.get("relationships"), list):
            all_relationships_raw.extend(rel for rel in res["relationships"] if isinstance(rel, dict) and rel.get("source") and rel.get("target")) # Проверяем формат каждой связи

    logging.info(f"Всего извлечено сущностей (до слияния): {len(all_entities_raw)}")
    logging.info(f"Всего извлечено связей (до обновления): {len(all_relationships_raw)}")


    # 4. Слияние дубликатов сущностей
    unique_entities_map, name_mapping = merge_entities(
        all_entities_raw, CONFIG["processing"]["levenshtein_threshold"]
    )

    # 5. Асинхронная Суммаризация описаний
    logging.info("Начало суммирования описаний для сущностей...")
    summarization_tasks = []
    for name, entity_obj in unique_entities_map.items():
        if len(entity_obj.descriptions) > 1:
            summarization_tasks.append(
                summarize_description(name, entity_obj.descriptions, llm_summarizer, CONFIG)
            )
        elif len(entity_obj.descriptions) == 1:
            # Если описание одно, просто "распаковываем" его из списка
             summarization_tasks.append(asyncio.sleep(0, result=entity_obj.descriptions[0])) # Возвращаем как результат "сна"
        else:
            # Если описаний нет
            summarization_tasks.append(asyncio.sleep(0, result="")) # Возвращаем пустую строку

    summarized_descriptions = await asyncio.gather(*summarization_tasks)
    logging.info("Суммирование описаний завершено.")

    # Обновляем сущности суммированными описаниями
    final_entities_list: List[Dict[str, Any]] = []
    entity_names = list(unique_entities_map.keys()) # Сохраняем порядок
    for i, name in enumerate(entity_names):
        final_entities_list.append({
            "name": name,
            "description": summarized_descriptions[i]
            # Добавьте сюда другие поля сущности, если они есть
        })

    # Проверка существования финальных имен сущностей для связей
    final_entity_names_set: Set[str] = {entity["name"] for entity in final_entities_list}


    # 6. Обновление и фильтрация связей
    final_relationships_list: List[Dict[str, Any]] = []
    seen_relationships: Set[Tuple[str, str, str]] = set() # Для удаления дубликатов связей

    logging.info("Обновление и фильтрация связей...")
    for rel_data in all_relationships_raw:
        original_source = rel_data.get("source")
        original_target = rel_data.get("target")
        description = rel_data.get("description", "") # Описание связи

        if not original_source or not original_target:
            logging.warning(f"Пропуск связи с отсутствующим source или target: {rel_data}")
            continue

        # Получаем канонические имена из карты слияния
        final_source = name_mapping.get(original_source.strip())
        final_target = name_mapping.get(original_target.strip())

        # Проверяем, существуют ли финальные source и target в списке уникальных сущностей
        if final_source and final_target and \
           final_source in final_entity_names_set and \
           final_target in final_entity_names_set:

            # Создаем кортеж для проверки уникальности связи
            rel_tuple = (final_source, final_target, description)

            if rel_tuple not in seen_relationships:
                final_relationships_list.append({
                    "source": final_source,
                    "target": final_target,
                    "description": description
                    # Добавьте сюда другие поля связи, если они есть
                })
                seen_relationships.add(rel_tuple)
            else:
                 logging.debug(f"Пропуск дублирующей связи: {rel_tuple}")

        else:
            logging.warning(f"Пропуск связи из-за отсутствия source ('{final_source}' из '{original_source}') или target ('{final_target}' из '{original_target}') в финальном списке сущностей или карте слияния. Исходная связь: {rel_data}")

    logging.info(f"Финальное количество сущностей: {len(final_entities_list)}")
    logging.info(f"Финальное количество связей: {len(final_relationships_list)}")


    # 7. Сохранение результатов в JSON (для отладки)
    final_result = {
        "entities": final_entities_list,
        "relationships": final_relationships_list
    }
    output_file = CONFIG["output"]["json_file"]
    try:
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(final_result, f, indent=4, ensure_ascii=False) # ensure_ascii=False для кириллицы
        logging.info(f"Результаты сохранены в файл: {output_file}")
    except IOError as e:
        logging.error(f"Не удалось сохранить результаты в файл {output_file}: {e}")

# --- Запуск основного цикла ---
if __name__ == "__main__":
    # Установка политики цикла событий для Windows, если необходимо
    # if os.name == 'nt':
    #      asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())