import os
import json
import asyncio
import logging
import re
from typing import List, Dict, Any, Tuple, Optional, Set, Union
from dataclasses import dataclass, field
from utils import remove_tokens, preprocess_text

import yaml
from dotenv import load_dotenv
from Levenshtein import ratio
from langchain_openai import ChatOpenAI
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document, AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.exceptions import OutputParserException

# --- Импортируем необходимые библиотеки для лемматизации---
import nltk  
from nltk.stem import WordNetLemmatizer  

# Инициализируем лемматизатор
__lemmatizer = WordNetLemmatizer() 

# Скачиваем необходимые ресурсы при первом использовани (для лемматизации на английском) 
nltk.download('punkt')
nltk.download('punkt_tab')  
nltk.download('wordnet')  

# --- Загрузка переменных окружения ---
load_dotenv()

# --- Датаклассы ---
@dataclass
class Entity:
    name: str
    type: str
    descriptions: List[str] = field(default_factory=list)

@dataclass
class Relationship:
    source: str
    target: str
    description: str


# --- Функции ---
def _strip_json_string(ai_message: Union[AIMessage, str]) -> str:
    """
    Удаляет обертку ```json ... ``` или просто ``` ... ``` из содержимого AI-сообщения и возвращает новое AI-сообщение.
    Также чистит от лишних пробелов и заменяет кавычки на стандартные.
    """
    stripped_content = ai_message.content.strip()  # Убираем пробелы в начале и конце
    
    stripped_content = stripped_content.replace('“', '"').replace('”', '"')
    stripped_content = stripped_content.replace('‘', "'").replace('’', "'")
    
    if stripped_content.startswith("```json"):
        stripped_content = stripped_content[7:].strip()  # Убираем обертку в начале

    if stripped_content.startswith("```"):
        stripped_content = stripped_content[3:].strip() # Убираем обертку в начале (Если первый случай не сработал)

    if stripped_content.endswith("```"):
        stripped_content = stripped_content[:-3].strip()  # Убираем обертку в конце

    # Проверяем, что содержимое не пустое
    ai_message.content = stripped_content if stripped_content else "{}"  # Если пустое, возвращаем пустой JSON
    
    # Возвращаем новое AI-сообщение с очищенным содержимым
    return ai_message 


def __lemmatize_phrase(phrase):  
    """
    Лемматизирует фразу, заменяя слова на их начальную форму (лемму).
    """
    words = nltk.word_tokenize(phrase)  # Токенизация фразы на отдельные слова  
    lemmatized_words = [__lemmatizer.lemmatize(word, pos='n') for word in words]  
    return ' '.join(lemmatized_words)  # Объединяем обратно в строку  

def _preprocess_text(text: str) -> str:
    """Преобразует текст, убирая лишние пробелы, точки и символы. Затем выполняет лемматизацию"""
    # замена '-' на пробел
    text = text.replace('-', ' ')
    # Убираем лишние пробелы
    text = re.sub(r'\s+', ' ', text).strip()
    # Убираем точки в конце текста
    if text.endswith('.'):
        text = text[:-1]
    # Убираем символы, которые не нужны
    text = re.sub(r'[^\w\s]', '', text)

    # Лемматизируем текст
    text = __lemmatize_phrase(text.lower())

    return text

def _get_similarity(name1: str, name2: str) -> float:
    """
    Возвращает коэффициент схожести между двумя строками по Левенштейну.
    """
    if not name1 or not name2:
        return 0.0
    
    # Приводим к нижнему регистру и убираем лишние пробелы
    # name1 = __lemmatize_phrase(name1.strip().lower())
    # name2 = __lemmatize_phrase(name2.strip().lower())

    return ratio(name1.lower(), name2.lower())




class GraphExtractor:
    """
    Класс для извлечения сущностей и связей из текстовых документов
    и построения графа знаний на основе конфигурации из YAML файла.
    """

    def __init__(self, config_path: str):
        """
        Инициализирует GraphExtractor.

        Args:
            config_path: Обязательный путь к YAML файлу конфигурации.

        Raises:
            FileNotFoundError: Если файл конфигурации не найден.
            ValueError: Если файл конфигурации некорректен или не содержит необходимых ключей.
        """
        if not config_path or not isinstance(config_path, str):
             raise ValueError("Необходимо указать действительный путь к файлу конфигурации (config_path).")

        self.config = self._load_config(config_path)
        self._initialize_llms()

    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        """Загружает конфигурацию из YAML файла."""
        logging.info(f"Загрузка конфигурации из файла: {config_path}")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            if not isinstance(config, dict):
                 raise ValueError(f"Файл конфигурации {config_path} не содержит валидный YAML словарь.")
            logging.info(f"Конфигурация успешно загружена.")
            # TODO: Добавить валидацию наличия необходимых ключей в config (например, llm, prompts и т.д.)
            return config
        except FileNotFoundError:
            logging.error(f"Файл конфигурации не найден: {config_path}")
            raise # Передаем исключение выше
        except yaml.YAMLError as e:
            logging.error(f"Ошибка парсинга YAML файла {config_path}: {e}")
            raise ValueError(f"Некорректный формат YAML в файле {config_path}") from e
        except Exception as e:
            logging.error(f"Неожиданная ошибка при загрузке config файла {config_path}: {e}", exc_info=True)
            raise ValueError(f"Ошибка при чтении файла конфигурации {config_path}") from e


    def _initialize_llms(self):
        """Инициализирует LLM модели на основе конфигурации."""
        try:
            # Базовая LLM для извлечения
            llm_base = ChatOpenAI(
                model=os.getenv("RAG_LLM_MODEL") or self.config["llm"]["chunk_processing_model"],
                base_url=os.getenv('RAG_BASE_URL'),
                temperature=self.config["llm"].get("temperature", 0.5), # Используем .get с дефолтом для необязательных параметров
                top_p=self.config["llm"].get("top_p", 0.9),
                max_tokens=self.config["llm"].get("max_tokens", 2048),
                timeout=self.config["llm"].get("request_timeout", 600),
                openai_api_key=os.getenv('OPENAI_API_KEY')
            )
            logging.info(f"LLM Extractor '{llm_base.model_name}' инициализирована с URL: {llm_base.openai_api_base}")

            # LLM для суммаризации
            self.llm_summarizer = ChatOpenAI(
                 model=os.getenv("RAG_LLM_MODEL_SUMMARIZER") or self.config["llm"]["summarization_model"],
                 base_url=os.getenv('RAG_BASE_URL'),
                 temperature=self.config["llm"].get("temperature", 0.5),
                 top_p=self.config["llm"].get("top_p", 0.9),
                 max_tokens=self.config["llm"].get("max_tokens", 2048),
                 timeout=self.config["llm"].get("request_timeout", 600),
                 openai_api_key=os.getenv('OPENAI_API_KEY')
            )
            logging.info(f"LLM Summarizer '{self.llm_summarizer.model_name}' инициализирована с URL: {self.llm_summarizer.openai_api_base}")

            # Создаем цепочку для экстрактора с очисткой и парсингом
            self.llm_extractor_chain = llm_base | RunnableLambda(_strip_json_string) | SimpleJsonOutputParser()

        except KeyError as e:
             logging.error(f"Отсутствует необходимый ключ в конфигурации LLM: {e}")
             raise ValueError(f"Ошибка конфигурации LLM: отсутствует ключ {e}") from e
        except Exception as e:
            logging.error(f"Ошибка инициализации LLM: {e}", exc_info=True)
            # Можно перевыбросить специфичное исключение для инициализации
            raise RuntimeError("Не удалось инициализировать LLM модели.") from e

    @staticmethod
    def save_to_json(save_json_path: Optional[str] = None, final_result: Optional[Dict[str, Any]] = None):
        """
        Сохраняет данные в JSON файл.
        """
        if save_json_path:
            try:
                # Создаем директорию, если она не существует
                output_dir = os.path.dirname(save_json_path)
                if output_dir: # Убедимся, что путь не просто имя файла
                     os.makedirs(output_dir, exist_ok=True)
                with open(save_json_path, "w", encoding='utf-8') as f:
                    json.dump(final_result, f, indent=4, ensure_ascii=False)
                logging.info(f"Результаты сохранены в файл: {save_json_path}")
            except IOError as e:
                logging.error(f"Не удалось сохранить результаты в файл {save_json_path}: {e}")
                return False
            except Exception as e:
                logging.error(f"Неожиданная ошибка при сохранении JSON в {save_json_path}: {e}", exc_info=True)
                return False
            return True
        return False # Если путь не указан, ничего не сохраняем

    @staticmethod
    def _load_documents(files_dir: str) -> List[Document]:
        """Загружает документы из указанной директории."""
        loaded_docs = []
        logging.info(f"Загрузка документов из директории: {files_dir}")
        if not os.path.isdir(files_dir):
            logging.error(f"Директория не найдена: {files_dir}")
            return []
        try:
            for filename in os.listdir(files_dir):
                file_path = os.path.join(files_dir, filename)
                loader = None
                if filename.endswith(".txt"):
                    loader = TextLoader(file_path, encoding='utf-8')
                elif filename.endswith(".docx"):
                    loader = Docx2txtLoader(file_path)
                elif filename.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)

                if loader:
                    try:
                        logging.debug(f"Загрузка файла: {filename}")
                        loaded_docs.extend(loader.load())
                    except Exception as load_err:
                         logging.error(f"Ошибка загрузки файла {filename}: {load_err}", exc_info=True)
            logging.info(f"Загружено {len(loaded_docs)} документов.")
        except Exception as e:
            logging.error(f"Ошибка при доступе к директории {files_dir}: {e}", exc_info=True)
        return loaded_docs


    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """Разбивает документы на чанки согласно конфигурации."""
        # (Код метода без изменений, использует self.config)
        if not documents:
            logging.warning("Нет документов для разбиения на чанки.")
            return []
        try:
             # Проверка наличия ключей в конфиге
             chunk_size = self.config["processing"]["chunk_size"]
             chunk_overlap = self.config["processing"]["chunk_overlap"]
        except KeyError as e:
             logging.error(f"Отсутствует ключ в конфигурации processing: {e}. Используются значения по умолчанию (1000, 200).")
             chunk_size = 1000
             chunk_overlap = 200
        except TypeError:
             logging.error("Раздел 'processing' в конфигурации имеет неверный формат. Используются значения по умолчанию (1000, 200).")
             chunk_size = 1000
             chunk_overlap = 200


        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)
        logging.info(f"Документы разбиты на {len(chunks)} чанков.")
        return chunks

    async def _process_chunk(self, chunk: Document) -> Optional[Dict[str, Any]]:
        """Асинхронно обрабатывает один чанк текста с помощью LLM extractor chain."""
        try:
            system_prompt = self.config["prompts"]["extraction_system_prompt"]
            user_template = self.config["prompts"]["extraction_user_template"]
        except KeyError as e:
             logging.error(f"Отсутствует ключ промпта в конфигурации: {e}")
             return None # Не можем продолжить без промпта
        except TypeError:
             logging.error("Раздел 'prompts' в конфигурации имеет неверный формат.")
             return None


        prompt = user_template.format(system_message=system_prompt, prompt=chunk.page_content)

        try:
            # Используем ainvoke для асинхронного вызова
            response = await self.llm_extractor_chain.ainvoke(prompt)
           
            # Проверяем наличие ключей entities и relationships (даже если они пустые списки)
            if "entities" in response and "relationships" in response:
                logging.debug(f"Чанк обработан успешно. Найдено сущностей: {len(response.get('entities', []))}, связей: {len(response.get('relationships', []))}")
                return response
            
            # Если ключи не найдены, логируем предупреждение
            logging.warning(f"Ответ LLM для чанка не содержит ключи 'entities' или 'relationships': {response}")            

            return None # Возвращаем None если структура некорректна

        except OutputParserException as json_parse_error:
            logging.error(f"Ошибка парсинга JSON ответа LLM: {json_parse_error}", exc_info=False)
            return None
        
        except Exception as e:
            logging.error(f"Ошибка при обработке чанка LLM: {e}", exc_info=True) # Добавляем traceback
            return None
            


    @staticmethod
    def _merge_entities(all_entities: List[Dict[str, Any]], threshold: float) -> Tuple[Dict[str, Entity], Dict[str, str]]:
        """
        Объединяет дублирующиеся сущности на основе схожести имен (Левенштейн).
        
        Returns:
            Кортеж:
                - Словарь уникальных сущностей {canonical_name: Entity}.
                - Словарь для переименования {old_name: canonical_name}.
        """
        unique_entities: Dict[str, Entity] = {}
        name_mapping: Dict[str, str] = {} # Карта для обновления связей {old_name: new_name}

        logging.info(f"Начало слияния {len(all_entities)} извлеченных сущностей...")
        for entity_data in all_entities:
            if not isinstance(entity_data, dict):
                logging.warning(f"Пропуск некорректных данных сущности: {entity_data}")
                continue
            name = entity_data.get("name")
            e_type = entity_data.get("type", "object")
            description = entity_data.get("description")

            if not name or not isinstance(name, str) or not name.strip():
                logging.warning(f"Пропуск сущности без имени или с некорректным именем: {entity_data}")
                continue
            name = name.strip()

            if not description or not isinstance(description, str):
                 description = ""
            else:
                 description = description.strip()

            matched = False
            for canonical_name in list(unique_entities.keys()):
                similarity = _get_similarity(name.lower(), canonical_name.lower())
                if similarity >= threshold:
                    logging.debug(f"Обнаружено слияние: '{name}' -> '{canonical_name}' (Схожесть: {similarity:.2f})")
                    if description:
                        unique_entities[canonical_name].descriptions.append(description)
                    if name != canonical_name:
                        name_mapping[name] = canonical_name
                    matched = True
                    break

            if not matched:
                logging.debug(f"Добавлена новая уникальная сущность: '{name}'")
                unique_entities[name] = Entity(name=name, type=e_type, descriptions=[desc for desc in [description] if desc])

        merged_count = sum(1 for old, new in name_mapping.items() if old != new)
        logging.info(f"Слияние сущностей завершено. Уникальных сущностей: {len(unique_entities)}. Обнаружено слияний: {merged_count}.")
        return unique_entities, name_mapping


    async def _summarize_description(self, entity_name: str, descriptions: List[str]) -> str:
        """Асинхронно суммирует список описаний для одной сущности."""
        if not descriptions:
            return ""
        unique_descriptions = sorted(list(set(filter(None, descriptions))))
        if not unique_descriptions:
             return ""
        if len(unique_descriptions) == 1:
            return unique_descriptions[0]

        logging.debug(f"Суммаризация {len(unique_descriptions)} описаний для '{entity_name}'...")
        try:
             system_prompt = self.config["prompts"]["summarization_system_prompt"]
             user_template = self.config["prompts"]["summarization_user_template"]
        except KeyError as e:
             logging.error(f"Отсутствует ключ промпта суммаризации в конфигурации: {e}")
             return ". ".join(unique_descriptions) # Fallback
        except TypeError:
             logging.error("Раздел 'prompts' в конфигурации имеет неверный формат.")
             return ". ".join(unique_descriptions) # Fallback


        prompt = user_template.format(system_message=system_prompt, text_to_summarize="\n---\n".join(unique_descriptions))

        try:
            summary_response = await self.llm_summarizer.ainvoke(prompt)

            if hasattr(summary_response, 'content'):
                summary = summary_response.content.strip()
            elif isinstance(summary_response, str):
                summary = summary_response.strip()
            else:
                logging.warning(f"Неожиданный тип ответа от LLM-суммаризатора для '{entity_name}': {type(summary_response)}")
                summary = ". ".join(unique_descriptions) # Fallback

            logging.debug(f"Описание для '{entity_name}' суммировано.")
            return summary or ". ".join(unique_descriptions)
        except Exception as e:
            logging.error(f"Ошибка при суммировании описания для '{entity_name}': {e}", exc_info=True)
            return ". ".join(unique_descriptions)


    async def extract_graph_from_path(self, data_path: str, save_json_path: Optional[str] = None, save: bool = True) -> Dict[str, List[Dict[str, Any]]]:
        """
        Основной метод для извлечения графа из документов в указанной директории.

        Args:
            data_path: Путь к директории с документами.
            save_json_path: Опциональный путь для сохранения результата в JSON.

        Returns:
            Словарь с ключами "entities" и "relationships".
        """
        if not data_path:
            logging.error("Не указан путь к данным (data_path).")
            return {"entities": [], "relationships": []}

        # 1. Загрузка и разбиение документов
        documents = self._load_documents(data_path)
        chunks = self._split_documents(documents)
        if not chunks:
            logging.warning("Нет чанков для обработки.")
            return {"entities": [], "relationships": []}

        # 2. Асинхронная обработка чанков
        logging.info(f"Начало асинхронной обработки {len(chunks)} чанков...")
        tasks = [self._process_chunk(chunk) for chunk in chunks]
        llm_results = await asyncio.gather(*tasks)
        logging.info("Обработка чанков завершена.")

        valid_results = [res for res in llm_results if res is not None]
        logging.info(f"Получено {len(valid_results)} валидных ответов от LLM.")
        if not valid_results:
             return {"entities": [], "relationships": []}

        # 3. Сбор всех сущностей и связей
        all_entities_raw: List[Dict[str, Any]] = []
        all_relationships_raw: List[Dict[str, Any]] = []
        for res in valid_results:
            entities = res.get("entities", [])
            relationships = res.get("relationships", [])
            if isinstance(entities, list):
                 all_entities_raw.extend(ent for ent in entities if isinstance(ent, dict) and ent.get("name"))
            if isinstance(relationships, list):
                 all_relationships_raw.extend(rel for rel in relationships if isinstance(rel, dict) and rel.get("source") and rel.get("target"))

        logging.info(f"Всего извлечено сущностей (до слияния): {len(all_entities_raw)}")
        logging.info(f"Всего извлечено связей (до обновления): {len(all_relationships_raw)}")

        # 3*. Лемматизация названий сущностей (приводим к единственному числу) 
        # Лемматизируем назавания сущностей в all_entities_raw
        for entity in all_entities_raw:
            entity["name"] = _preprocess_text(entity["name"])
            entity["description"] = remove_tokens(entity["description"]) if entity.get("description") else ""
        
        # Лемматизируем назавания сущностей в all_relationships_raw
        for relationship in all_relationships_raw:
            relationship["source"] = _preprocess_text(relationship["source"])
            relationship["target"] = _preprocess_text(relationship["target"])
        
        # 4. Слияние дубликатов сущностей
        try:
            threshold = self.config["processing"]["levenshtein_threshold"]
        except KeyError:
            logging.warning("Ключ 'levenshtein_threshold' не найден в config['processing']. Используется значение 0.85.")
            threshold = 0.85
        except TypeError:
            logging.warning("Раздел 'processing' в конфигурации имеет неверный формат. Используется threshold 0.85.")
            threshold = 0.85

        unique_entities_map, name_mapping = self._merge_entities(all_entities_raw, threshold)
        if not unique_entities_map:
             logging.warning("Не найдено уникальных сущностей после слияния.")
             return {"entities": [], "relationships": []}

        # 5. Асинхронная Суммаризация описаний
        logging.info("Начало суммирования описаний для сущностей...")
        summarization_tasks = [
            self._summarize_description(name, entity_obj.descriptions)
            for name, entity_obj in unique_entities_map.items()
        ]
        summarized_descriptions = await asyncio.gather(*summarization_tasks)
        logging.info("Суммирование описаний завершено.")

        # Формируем финальный список сущностей
        final_entities_list: List[Dict[str, Any]] = []
        entity_names_in_order = list(unique_entities_map.keys())
        for i, name in enumerate(entity_names_in_order):
            final_entities_list.append({
                "name": name,
                "type": unique_entities_map[name].type,
                "description": summarized_descriptions[i]
            })
        final_entity_names_set: Set[str] = set(entity_names_in_order)

        # 6. Обновление и фильтрация связей (с добавлением недостающих сущностей)
        logging.info("Обновление, фильтрация связей и добавление недостающих сущностей...")
        final_relationships_list: List[Dict[str, Any]] = []
        seen_relationships: Set[Tuple[str, str, str]] = set()
        added_entities_names: Set[str] = set() # Отслеживаем имена добавленных сущностей

        for rel_data in all_relationships_raw:
            original_source = rel_data.get("source", "").strip()
            original_target = rel_data.get("target", "").strip()
            description = rel_data.get("description", "").strip()

            if not original_source or not original_target:
                logging.warning(f"Пропуск связи с некорректным source или target: {rel_data}")
                continue

            # Получаем канонические имена, используя исходное имя как fallback, если его нет в карте
            final_source = name_mapping.get(original_source, original_source)
            final_target = name_mapping.get(original_target, original_target)

            # Пытаемся добавить сущности, если их нет в финальном списке
            entities_to_check = [final_source, final_target]
            for entity_name in entities_to_check:
                if entity_name not in final_entity_names_set and entity_name not in added_entities_names:
                    logging.info(f"Сущность '{entity_name}', упомянутая в связи, отсутствует. Добавление новой сущности с пустым описанием.")
                    final_entities_list.append({
                        "name": entity_name,
                        "type": "object",
                        "description": ""
                    })
                    final_entity_names_set.add(entity_name)
                    added_entities_names.add(entity_name)

            # Проверяем, существуют ли финальные source и target в множестве финальных сущностей
            if final_source in final_entity_names_set and final_target in final_entity_names_set:
                rel_tuple = (final_source, final_target, description)
                if rel_tuple not in seen_relationships:
                    final_relationships_list.append({
                        "source": final_source,
                        "target": final_target,
                        "description": description
                    })
                    seen_relationships.add(rel_tuple)
                else:
                    logging.debug(f"Пропуск дублирующей связи: {rel_tuple}")
            else:
                # Логируем только если имя было в name_mapping, но не попало в final_entity_names_set (что странно)
                # или если исходного имени нет в final_entity_names_set
                if (original_source in name_mapping and final_source not in final_entity_names_set) or \
                (original_target in name_mapping and final_target not in final_entity_names_set) or \
                (original_source not in name_mapping and final_source not in final_entity_names_set) or \
                (original_target not in name_mapping and final_target not in final_entity_names_set):
                    logging.warning(f"Пропуск связи из-за отсутствия source ('{final_source}' из '{original_source}') или target ('{final_target}' из '{original_target}') в финальном списке сущностей. Исходная связь: {rel_data}")

        initial_entity_names_set = set(entity_names_in_order)
        total_added_count = len(final_entity_names_set - initial_entity_names_set)
        if total_added_count > 0:
            logging.info(f"Всего добавлено {total_added_count} недостающих сущностей из связей.")

        logging.info(f"Финальное количество сущностей: {len(final_entities_list)}")
        logging.info(f"Финальное количество связей: {len(final_relationships_list)}")

        # 7. Формирование и возврат результата
        final_result = {
            "entities": final_entities_list,
            "relationships": final_relationships_list
        }

        if save:
            # 8. Сохранение в JSON (если указан путь)
            self.save_to_json(save_json_path, final_result)
        else:
            logging.info(f"Выбран режим без сохранения в JSON. Результаты не будут сохранены в {save_json_path}.")
        
        logging.info("Экстракция графа завершена.")
        # Возвращаем финальный результат
        return final_result
    
    def update(self, DATA_DIR_ENV = "DATA_PATH", save_to_json = True):
        """
        Метод для загрузки документов из директории, обработки их с помощью LLM и сохранения результатов в JSON файл.
        """

        logging.info("Запуск: извлечение файлов из директории...")

        data_directory = os.getenv(DATA_DIR_ENV)
        if not data_directory:
            logging.critical(f"Критическая ошибка: Переменная окружения '{DATA_DIR_ENV}' не установлена. Завершение работы.")
            exit(1) # Завершаем скрипт, если нет пути к данным

        try:
            # Получение пути для сохранения из конфига (если есть)
            output_file_path = self.config.get("output", {}).get("json_file", "./output.json")
            if not output_file_path:
                logging.warning("Путь для сохранения JSON не найден в конфигурации (output.json_file). Результат будет сохранен в ./output.json. ")

            # Запуск основного метода экстракции
            graph_data = asyncio.run(
                self.extract_graph_from_path(data_directory, save_json_path=output_file_path, save = save_to_json)
            )
            logging.info(f"Экстракция завершена. Найдено {len(graph_data.get('entities', []))} сущностей и {len(graph_data.get('relationships', []))} связей.")
            return graph_data
        
        except (ValueError, FileNotFoundError, RuntimeError, KeyError) as init_error:
            # Ловим ошибки инициализации или отсутствия ключей в конфиге
            logging.critical(f"Ошибка инициализации GraphExtractor: {init_error}. Завершение работы.")
            exit(1)

        except Exception as e:
            logging.critical(f"Непредвиденная ошибка во время выполнения: {e}", exc_info=True)
            exit(1)
        
        return None


# --- Блок для автономного запуска ---
if __name__ == "__main__":    
    # --- Конфигурация логирования ---
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 1. Инициализация экстрактора с путем к конфигу
    extractor = GraphExtractor(config_path= "./config.yaml")
    extractor.update()