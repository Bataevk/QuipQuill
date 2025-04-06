import os
import json
import asyncio
import logging
import re
from typing import List, Dict, Any, Tuple, Optional, Set, Union
from dataclasses import dataclass, field

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

# --- Конфигурация логирования ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Загрузка переменных окружения ---
load_dotenv()

# --- Датаклассы ---
@dataclass
class Entity:
    name: str
    descriptions: List[str] = field(default_factory=list)

@dataclass
class Relationship:
    source: str
    target: str
    description: str

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
        logger.info(f"Загрузка конфигурации из файла: {config_path}")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            if not isinstance(config, dict):
                 raise ValueError(f"Файл конфигурации {config_path} не содержит валидный YAML словарь.")
            logger.info(f"Конфигурация успешно загружена.")
            # TODO: Добавить валидацию наличия необходимых ключей в config (например, llm, prompts и т.д.)
            return config
        except FileNotFoundError:
            logger.error(f"Файл конфигурации не найден: {config_path}")
            raise # Передаем исключение выше
        except yaml.YAMLError as e:
            logger.error(f"Ошибка парсинга YAML файла {config_path}: {e}")
            raise ValueError(f"Некорректный формат YAML в файле {config_path}") from e
        except Exception as e:
            logger.error(f"Неожиданная ошибка при загрузке config файла {config_path}: {e}", exc_info=True)
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
            logger.info(f"LLM Extractor '{llm_base.model_name}' инициализирована с URL: {llm_base.openai_api_base}")

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
            logger.info(f"LLM Summarizer '{self.llm_summarizer.model_name}' инициализирована с URL: {self.llm_summarizer.openai_api_base}")

            # Создаем цепочку для экстрактора с очисткой и парсингом
            self.llm_extractor_chain = llm_base | RunnableLambda(self._strip_json_string) | SimpleJsonOutputParser()

        except KeyError as e:
             logger.error(f"Отсутствует необходимый ключ в конфигурации LLM: {e}")
             raise ValueError(f"Ошибка конфигурации LLM: отсутствует ключ {e}") from e
        except Exception as e:
            logger.error(f"Ошибка инициализации LLM: {e}", exc_info=True)
            # Можно перевыбросить специфичное исключение для инициализации
            raise RuntimeError("Не удалось инициализировать LLM модели.") from e

    @staticmethod
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
                logger.info(f"Результаты сохранены в файл: {save_json_path}")
            except IOError as e:
                logger.error(f"Не удалось сохранить результаты в файл {save_json_path}: {e}")
                return False
            except Exception as e:
                logger.error(f"Неожиданная ошибка при сохранении JSON в {save_json_path}: {e}", exc_info=True)
                return False
            return True
        return False # Если путь не указан, ничего не сохраняем

    @staticmethod
    def _load_documents(files_dir: str) -> List[Document]:
        """Загружает документы из указанной директории."""
        loaded_docs = []
        logger.info(f"Загрузка документов из директории: {files_dir}")
        if not os.path.isdir(files_dir):
            logger.error(f"Директория не найдена: {files_dir}")
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
                        logger.debug(f"Загрузка файла: {filename}")
                        loaded_docs.extend(loader.load())
                    except Exception as load_err:
                         logger.error(f"Ошибка загрузки файла {filename}: {load_err}", exc_info=True)
            logger.info(f"Загружено {len(loaded_docs)} документов.")
        except Exception as e:
            logger.error(f"Ошибка при доступе к директории {files_dir}: {e}", exc_info=True)
        return loaded_docs


    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """Разбивает документы на чанки согласно конфигурации."""
        # (Код метода без изменений, использует self.config)
        if not documents:
            logger.warning("Нет документов для разбиения на чанки.")
            return []
        try:
             # Проверка наличия ключей в конфиге
             chunk_size = self.config["processing"]["chunk_size"]
             chunk_overlap = self.config["processing"]["chunk_overlap"]
        except KeyError as e:
             logger.error(f"Отсутствует ключ в конфигурации processing: {e}. Используются значения по умолчанию (1000, 200).")
             chunk_size = 1000
             chunk_overlap = 200
        except TypeError:
             logger.error("Раздел 'processing' в конфигурации имеет неверный формат. Используются значения по умолчанию (1000, 200).")
             chunk_size = 1000
             chunk_overlap = 200


        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Документы разбиты на {len(chunks)} чанков.")
        return chunks

    async def _process_chunk(self, chunk: Document) -> Optional[Dict[str, Any]]:
        """Асинхронно обрабатывает один чанк текста с помощью LLM extractor chain."""
        try:
            system_prompt = self.config["prompts"]["extraction_system_prompt"]
            user_template = self.config["prompts"]["extraction_user_template"]
        except KeyError as e:
             logger.error(f"Отсутствует ключ промпта в конфигурации: {e}")
             return None # Не можем продолжить без промпта
        except TypeError:
             logger.error("Раздел 'prompts' в конфигурации имеет неверный формат.")
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
            logger.error(f"Ошибка парсинга JSON ответа LLM: {json_parse_error}", exc_info=False)
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

        logger.info(f"Начало слияния {len(all_entities)} извлеченных сущностей...")
        for entity_data in all_entities:
            if not isinstance(entity_data, dict):
                logger.warning(f"Пропуск некорректных данных сущности: {entity_data}")
                continue
            name = entity_data.get("name")
            description = entity_data.get("description")

            if not name or not isinstance(name, str) or not name.strip():
                logger.warning(f"Пропуск сущности без имени или с некорректным именем: {entity_data}")
                continue
            name = name.strip()

            if not description or not isinstance(description, str):
                 description = ""
            else:
                 description = description.strip()

            matched = False
            for canonical_name in list(unique_entities.keys()):
                similarity = ratio(name.lower(), canonical_name.lower())
                if similarity >= threshold:
                    logger.debug(f"Обнаружено слияние: '{name}' -> '{canonical_name}' (Схожесть: {similarity:.2f})")
                    if description:
                        unique_entities[canonical_name].descriptions.append(description)
                    if name != canonical_name:
                        name_mapping[name] = canonical_name
                    matched = True
                    break

            if not matched:
                logger.debug(f"Добавлена новая уникальная сущность: '{name}'")
                unique_entities[name] = Entity(name=name, descriptions=[desc for desc in [description] if desc])

        merged_count = sum(1 for old, new in name_mapping.items() if old != new)
        logger.info(f"Слияние сущностей завершено. Уникальных сущностей: {len(unique_entities)}. Обнаружено слияний: {merged_count}.")
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

        logger.debug(f"Суммаризация {len(unique_descriptions)} описаний для '{entity_name}'...")
        try:
             system_prompt = self.config["prompts"]["summarization_system_prompt"]
             user_template = self.config["prompts"]["summarization_user_template"]
        except KeyError as e:
             logger.error(f"Отсутствует ключ промпта суммаризации в конфигурации: {e}")
             return ". ".join(unique_descriptions) # Fallback
        except TypeError:
             logger.error("Раздел 'prompts' в конфигурации имеет неверный формат.")
             return ". ".join(unique_descriptions) # Fallback


        prompt = user_template.format(system_message=system_prompt, text_to_summarize="\n---\n".join(unique_descriptions))

        try:
            summary_response = await self.llm_summarizer.ainvoke(prompt)

            if hasattr(summary_response, 'content'):
                summary = summary_response.content.strip()
            elif isinstance(summary_response, str):
                summary = summary_response.strip()
            else:
                logger.warning(f"Неожиданный тип ответа от LLM-суммаризатора для '{entity_name}': {type(summary_response)}")
                summary = ". ".join(unique_descriptions) # Fallback

            logger.debug(f"Описание для '{entity_name}' суммировано.")
            return summary or ". ".join(unique_descriptions)
        except Exception as e:
            logger.error(f"Ошибка при суммировании описания для '{entity_name}': {e}", exc_info=True)
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
            logger.error("Не указан путь к данным (data_path).")
            return {"entities": [], "relationships": []}

        # 1. Загрузка и разбиение документов
        documents = self._load_documents(data_path)
        chunks = self._split_documents(documents)
        if not chunks:
            logger.warning("Нет чанков для обработки.")
            return {"entities": [], "relationships": []}

        # 2. Асинхронная обработка чанков
        logger.info(f"Начало асинхронной обработки {len(chunks)} чанков...")
        tasks = [self._process_chunk(chunk) for chunk in chunks]
        llm_results = await asyncio.gather(*tasks)
        logger.info("Обработка чанков завершена.")

        valid_results = [res for res in llm_results if res is not None]
        logger.info(f"Получено {len(valid_results)} валидных ответов от LLM.")
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

        logger.info(f"Всего извлечено сущностей (до слияния): {len(all_entities_raw)}")
        logger.info(f"Всего извлечено связей (до обновления): {len(all_relationships_raw)}")

        # 4. Слияние дубликатов сущностей
        try:
             threshold = self.config["processing"]["levenshtein_threshold"]
        except KeyError:
             logger.warning("Ключ 'levenshtein_threshold' не найден в config['processing']. Используется значение 0.85.")
             threshold = 0.85
        except TypeError:
             logger.warning("Раздел 'processing' в конфигурации имеет неверный формат. Используется threshold 0.85.")
             threshold = 0.85

        unique_entities_map, name_mapping = self._merge_entities(all_entities_raw, threshold)
        if not unique_entities_map:
             logger.warning("Не найдено уникальных сущностей после слияния.")
             return {"entities": [], "relationships": []}

        # 5. Асинхронная Суммаризация описаний
        logger.info("Начало суммирования описаний для сущностей...")
        summarization_tasks = [
            self._summarize_description(name, entity_obj.descriptions)
            for name, entity_obj in unique_entities_map.items()
        ]
        summarized_descriptions = await asyncio.gather(*summarization_tasks)
        logger.info("Суммирование описаний завершено.")

        # Формируем финальный список сущностей
        final_entities_list: List[Dict[str, Any]] = []
        entity_names_in_order = list(unique_entities_map.keys())
        for i, name in enumerate(entity_names_in_order):
            final_entities_list.append({
                "name": name,
                "description": summarized_descriptions[i]
            })
        final_entity_names_set: Set[str] = set(entity_names_in_order)




        # 6. Обновление и фильтрация связей
        logger.info("Обновление и фильтрация связей...")
        final_relationships_list: List[Dict[str, Any]] = []
        seen_relationships: Set[Tuple[str, str, str]] = set()

        for rel_data in all_relationships_raw:
            original_source = rel_data.get("source", "").strip()
            original_target = rel_data.get("target", "").strip()
            description = rel_data.get("description", "").strip()

            if not original_source or not original_target:
                logger.warning(f"Пропуск связи с некорректным source или target: {rel_data}")
                continue

            final_source = name_mapping.get(original_source, original_source)
            final_target = name_mapping.get(original_target, original_target)

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
                    logger.debug(f"Пропуск дублирующей связи: {rel_tuple}")
            else:
                if (original_source in name_mapping and final_source not in final_entity_names_set) or \
                (original_target in name_mapping and final_target not in final_entity_names_set) or \
                (original_source not in name_mapping and final_source not in final_entity_names_set) or \
                (original_target not in name_mapping and final_target not in final_entity_names_set):
                    logger.warning(f"Пропуск связи из-за отсутствия source ('{final_source}' из '{original_source}') или target ('{final_target}' из '{original_target}') в финальном списке сущностей. Исходная связь: {rel_data}")

        logger.info(f"Финальное количество сущностей: {len(final_entities_list)}")
        logger.info(f"Финальное количество связей: {len(final_relationships_list)}")

        # 7. Формирование и возврат результата
        final_result = {
            "entities": final_entities_list,
            "relationships": final_relationships_list
        }

        if save:
            # 8. Сохранение в JSON (если указан путь)
            self.save_to_json(save_json_path, final_result)
        else:
            logger.info(f"Выбран режим без сохранения в JSON. Результаты не будут сохранены в {save_json_path}.")
        
        logger.info("Экстракция графа завершена.")
        # Возвращаем финальный результат
        return final_result
    
    def update(self, DATA_DIR_ENV = "DATA_PATH", save_to_json = True):
        """
        Метод для загрузки документов из директории, обработки их с помощью LLM и сохранения результатов в JSON файл.
        """

        logger.info("Запуск: извлечение файлов из директории...")

        data_directory = os.getenv(DATA_DIR_ENV)
        if not data_directory:
            logger.critical(f"Критическая ошибка: Переменная окружения '{DATA_DIR_ENV}' не установлена. Завершение работы.")
            exit(1) # Завершаем скрипт, если нет пути к данным

        try:
            # Получение пути для сохранения из конфига (если есть)
            output_file_path = self.config.get("output", {}).get("json_file", "./output.json")
            if not output_file_path:
                logger.warning("Путь для сохранения JSON не найден в конфигурации (output.json_file). Результат будет сохранен в ./output.json. ")

            # Запуск основного метода экстракции
            graph_data = asyncio.run(
                self.extract_graph_from_path(data_directory, save_json_path=output_file_path, save = save_to_json)
            )
            logger.info(f"Экстракция завершена. Найдено {len(graph_data.get('entities', []))} сущностей и {len(graph_data.get('relationships', []))} связей.")
            return graph_data
        
        except (ValueError, FileNotFoundError, RuntimeError, KeyError) as init_error:
            # Ловим ошибки инициализации или отсутствия ключей в конфиге
            logger.critical(f"Ошибка инициализации GraphExtractor: {init_error}. Завершение работы.")
            exit(1)

        except Exception as e:
            logger.critical(f"Непредвиденная ошибка во время выполнения: {e}", exc_info=True)
            exit(1)
        
        return None


# --- Блок для автономного запуска ---
if __name__ == "__main__":
    # 1. Инициализация экстрактора с путем к конфигу
    extractor = GraphExtractor(config_path= "./config.yaml")
    extractor.update()