import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
import logging
from typing import List, Dict, Optional, Any
from utils import preprocess_text

from my_dataclasses import Entity

# --- Модуль для работы с ChromaDB и векторизацией текста ---
class VectorDB:
    """
    Модуль для взаимодействия с ChromaDB, управляющий коллекциями
    для имен и описаний сущностей.
    """
    DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
    DEFAULT_PERSIST_PATH = "./.chroma_db"
    COLLECTION_NAMES = "entity_names"
    COLLECTION_DESCRIPTIONS = "entity_descriptions"

    def __init__(self,
                 db_name: str = None,
                 persist_path: str = DEFAULT_PERSIST_PATH,
                 embedding_model_name: str = DEFAULT_EMBEDDING_MODEL):
        """
        Инициализирует модуль VectorDB.

        Args:
            persist_path: Путь для сохранения данных ChromaDB.
            embedding_model_name: Имя модели sentence-transformer для создания эмбеддингов.
        """
        self.db_name = db_name
        try:
            self.client = chromadb.PersistentClient(
                path=persist_path,
                    settings=Settings(
                        is_persistent=True,
                        persist_directory=".chroma_db",
                        anonymized_telemetry=False,   # <-- вот эта опция С БЕСПОЛЕЗНЫМИ СПАМ-сообщениями (отключает телеметрию)
                    )
                )
            logging.info(f"ChromaDB PersistentClient инициализирован по пути: {persist_path}")

            # Используем SentenceTransformer для эмбеддингов
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model_name
            )
            logging.info(f"Функция эмбеддингов инициализирована с моделью: {embedding_model_name}")

            # Получаем или создаем коллекции
            if db_name:
                en_name_col = f"{db_name}_{self.COLLECTION_NAMES}"
                en_desc_col = f"{db_name}_{self.COLLECTION_DESCRIPTIONS}"
            else:
                en_name_col = self.COLLECTION_NAMES
                en_desc_col = self.COLLECTION_DESCRIPTIONS
                

            self.names_collection = self.client.get_or_create_collection(
                name=en_name_col,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"} # Используем косинусное расстояние
            )
            logging.info(f"Коллекция '{self.COLLECTION_NAMES}' загружена/создана.")

            self.descriptions_collection = self.client.get_or_create_collection(
                name=en_desc_col,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            logging.info(f"Коллекция '{self.COLLECTION_DESCRIPTIONS}' загружена/создана.")

        except Exception as e:
            logging.error(f"Ошибка инициализации VectorDBModule: {e}", exc_info=True)
            raise RuntimeError("Не удалось инициализировать VectorDBModule") from e

    def add_entity(self, entity: Entity):
        """
        Добавляет или обновляет сущность в обеих коллекциях.

        Args:
            entity: Экземпляр класса Entity, содержащий имя, тип и описание.
        """
        if not entity.name or not isinstance(entity.name, str):
            logging.warning(f"Пропуск добавления сущности с некорректным именем: {entity.name}")
            return

        try:
            # Добавляем/обновляем имя
            self.names_collection.upsert(
                ids=[entity.name],
                documents=[ f"{entity.type}: {entity.name}" if entity.type else entity.name ], # Векторизуем само тип + имя
                metadatas=[{"entity_name": entity.name}],
            )
            logging.debug(f"Имя сущности '{entity.name}' добавлено/обновлено в коллекцию '{self.COLLECTION_NAMES}'.")

            # Подготавливаем текст для коллекции описаний
            # Встраиваем имя в описание для потенциально лучшего поиска
            description_text_for_embedding = f"Type: {entity.type}| Entity: {entity.name}| Description: {preprocess_text(entity.description)}"

            # Добавляем/обновляем описание
            self.descriptions_collection.upsert(
                ids=[entity.name],
                documents=[description_text_for_embedding],
                metadatas=[{"entity_name": entity.name, "original_description": entity.description}], # Сохраняем исходные данные
            )
            logging.debug(f"Описание сущности '{entity.name}' добавлено/обновлено в коллекцию '{self.COLLECTION_DESCRIPTIONS}'.")

        except Exception as e:
            logging.error(f"Ошибка при добавлении/обновлении сущности '{entity.name}': {e}", exc_info=True)

    def add_entities_batch(self, entities: List[Dict[str, str]]):
        """
        Добавляет или обновляет пакет сущностей в обеих коллекциях.

        Args:
            entities: Список словарей, где каждый словарь содержит ключи 'name' и 'description'.
        """
        if not entities:
            return

        names_ids = []
        names_docs = []
        names_metadatas = []
        desc_ids = []
        desc_docs = []
        desc_metadatas = []

        valid_entity_count = 0
        for entity in entities:
            name = entity.get("name")
            description = entity.get("description", "")
            etype = entity.get("type", None) # Получаем тип сущности, если есть

            if not name or not isinstance(name, str):
                logging.warning(f"Пропуск сущности с некорректным именем в батче: {name}")
                continue

            description = description or ""
            valid_entity_count += 1

            # Данные для коллекции имен
            names_ids.append(name)
            names_docs.append( f"{etype}: {name}" if etype else name )
            names_metadatas.append({"entity_name": name})

            # Данные для коллекции описаний
            description_text_for_embedding = f"Type: {etype}| Entity: {name}| Description: {preprocess_text(description)}"
            desc_ids.append(name)
            desc_docs.append(description_text_for_embedding)
            desc_metadatas.append({"entity_name": name, "original_description": description})

        if not valid_entity_count:
             logging.warning("В батче не найдено валидных сущностей для добавления.")
             return

        try:
            if names_ids:
                self.names_collection.upsert(
                    ids=names_ids,
                    documents=names_docs,
                    metadatas=names_metadatas,
                )
                logging.debug(f"Добавлено/обновлено {len(names_ids)} имен в '{self.COLLECTION_NAMES}'.")

            if desc_ids:
                self.descriptions_collection.upsert(
                    ids=desc_ids,
                    documents=desc_docs,
                    metadatas=desc_metadatas,
                )
                logging.debug(f"Добавлено/обновлено {len(desc_ids)} описаний в '{self.COLLECTION_DESCRIPTIONS}'.")

            logging.info(f"Батч из {valid_entity_count} сущностей обработан.")

        except Exception as e:
            logging.error(f"Ошибка при батчевом добавлении/обновлении сущностей: {e}", exc_info=True)

    def query_names(self, query_text: str, n_results: int = 5) -> Optional[Dict[str, Any]]:
        """
        Выполняет поиск по коллекции имен сущностей.

        Args:
            query_text: Текст запроса.
            n_results: Количество возвращаемых результатов.

        Returns:
            Результаты поиска ChromaDB или None в случае ошибки.
            Формат см. в документации ChromaDB (обычно включает ids, distances, metadatas, documents).
        """
        if not query_text:
            return None
        try:
            query_text = preprocess_text(query_text) # Предобработка текста
            # Выполняем поиск
            results = self.names_collection.query(
                query_texts=[query_text],
                n_results=n_results,
                include=['metadatas', 'distances'] # Включаем метаданные (там имя) и расстояния
            )
            logging.debug(f"Поиск по именам для '{query_text}' вернул {len(results.get('ids', [[]])[0])} результатов.")
            return results
        except Exception as e:
            logging.error(f"Ошибка при поиске по именам ('{query_text}'): {e}", exc_info=True)
            return None

    def query_descriptions(self, query_text: str, n_results: int = 5) -> Optional[Dict[str, Any]]:
        """
        Выполняет поиск по коллекции описаний сущностей.

        Args:
            query_text: Текст запроса.
            n_results: Количество возвращаемых результатов.

        Returns:
            Результаты поиска ChromaDB или None в случае ошибки.
            Формат см. в документации ChromaDB (обычно включает ids, distances, metadatas, documents).
        """
        if not query_text:
            return None
        try:
            description = preprocess_text(query_text) # Предобработка текста
            # Выполняем поиск
            results = self.descriptions_collection.query(
                query_texts=[description],
                n_results=n_results,
                include=['metadatas', 'distances'] # Включаем метаданные (имя, описание) и расстояния
            )
            logging.debug(f"Поиск по описаниям для '{query_text}' вернул {len(results.get('ids', [[]])[0])} результатов.")
            return results
        except Exception as e:
            logging.error(f"Ошибка при поиске по описаниям ('{query_text}'): {e}", exc_info=True)
            return None

    def delete_entity(self, name: str):
        """
        Удаляет сущность из обеих коллекций.

        Args:
            name: Имя сущности (ID).
        """
        if not name:
            return
        try:
            self.names_collection.delete(ids=[name])
            self.descriptions_collection.delete(ids=[name])
            logging.info(f"Сущность '{name}' удалена из коллекций.")
        except Exception as e:
            # ChromaDB может выдать ошибку, если ID не найден, обрабатываем это
            logging.warning(f"Ошибка или сущность не найдена при удалении '{name}': {e}")

    def get_entity_count(self) -> tuple[int, int]:
        """Возвращает количество сущностей в каждой коллекции."""
        try:
            names_count = self.names_collection.count()
            descriptions_count = self.descriptions_collection.count()
            return names_count, descriptions_count
        except Exception as e:
            logging.error(f"Ошибка при получении количества сущностей: {e}", exc_info=True)
            return -1, -1

# --- Пример использования ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    print("Запуск примера VectorDBModule...")

    # Инициализация
    vector_db = VectorDB(persist_path="./.test_chroma_db") # Используем тестовую директорию

    # from json import loads
    # with open("./output.json", "r", encoding="utf-8") as f:
    #     entities = loads(f.read())['entities']
    # 
    # # Добавление сущностей
    # print("\n--- Добавление сущностей ---")
    # vector_db.add_entities_batch(entities=entities)


    print(f"Количество сущностей (имена/описания): {vector_db.get_entity_count()}")

    # Поиск по именам
    print("\n--- Поиск по именам ('hotland') ---")
    name_results = vector_db.query_names("hotland", n_results=3)
    if name_results:
        print(f"IDs: {name_results.get('ids')}")
        print(f"Distances: {name_results.get('distances')}")
        print(f"Metadatas: {name_results.get('metadatas')}")

    # Поиск по описаниям
    print("\n--- Поиск по описаниям ('Location with the nucleus') ---")
    desc_results = vector_db.query_descriptions("Location with the nucleus", n_results=3)
    if desc_results:
        print(f"IDs: {desc_results.get('ids')}")
        print(f"Distances: {desc_results.get('distances')}")
        print(f"Metadatas: {desc_results.get('metadatas')}") # Здесь будет entity_name и original_description

    print("\nПример VectorDBModule завершен.")
