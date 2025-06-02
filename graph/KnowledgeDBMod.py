from GDBMod import GraphDB 
from VDBMod import VectorDB
from extractor import GraphExtractor
import logging
import os
import json
from typing import List, Dict, Optional, Any

from my_dataclasses import Entity, Relationship

class KnowledgeDB:
    """
    Общий модуль базы знаний, объединяющий векторный поиск (ChromaDB)
    и графовую базу данных (Neo4j).
    """
    DEFAULT_GRAPH_QUERY_RESULTS = 3

    def __init__(self,
            # Параметры для VectorDB
            chroma_persist_path: str = VectorDB.DEFAULT_PERSIST_PATH,
            embedding_model_name: str = VectorDB.DEFAULT_EMBEDDING_MODEL,
            # Параметры для GraphDB (из .env или передать явно)
            neo4j_uri: Optional[str] = None,
            neo4j_user: Optional[str] = None,
            neo4j_password: Optional[str] = None,
            # Параметры поиска
            n_results_graph_query: int = DEFAULT_GRAPH_QUERY_RESULTS, 
            db_name: Optional[str] = None,
            ):
        """
        Инициализирует KnowledgeDB.

        Args:
            chroma_persist_path: Путь для ChromaDB.
            embedding_model_name: Модель для эмбеддингов ChromaDB.
            neo4j_uri: URI для Neo4j (если не указан, берется из NEO4J_URI).
            neo4j_user: Имя пользователя Neo4j (если не указан, берется из NEO4J_USERNAME).
            neo4j_password: Пароль Neo4j (если не указан, берется из NEO4J_PASSWORD).
            n_results_graph_query: Количество топ-результатов векторного поиска,
                                    для которых выполняется запрос к графовой БД.
        """
        logging.info("Инициализация KnowledgeDB...")

        # Инициализация VectorDB
        try:
            self.vector_db = VectorDB(
                db_name=db_name,
                persist_path=chroma_persist_path,
                embedding_model_name=embedding_model_name
            )
        except Exception as e:
            logging.error(f"Ошибка инициализации VectorDB внутри KnowledgeDB: {e}", exc_info=True)
            # Решаем, что делать - падать или работать без векторной части? Пока падаем.
            raise RuntimeError("Не удалось инициализировать VectorDB") from e

        # Инициализация GraphDB
        uri = neo4j_uri or os.getenv("NEO4J_URI")
        user = neo4j_user or os.getenv("NEO4J_USERNAME")
        password = neo4j_password or os.getenv("NEO4J_PASSWORD")

        if not uri or not user:
            logging.warning("Параметры подключения к Neo4j (URI, USER) не найдены. GraphDB не будет инициализирован.")
            self.graph_db = None # Или можно создать заглушку
        else:
            try:
                self.graph_db = GraphDB(uri=uri, user=user, password=password, database_name=db_name)
                if not self.graph_db.ping():
                     logging.error("Не удалось подключиться к Neo4j. Проверьте параметры и доступность БД.")
                     # Можно установить self.graph_db = None или выбросить исключение
                     # raise ConnectionError("Не удалось подключиться к Neo4j")
                     self.graph_db = None # Работаем без графа, если не удалось подключиться
                else:
                     logging.info("GraphDB успешно инициализирован и подключен.")
            except Exception as e:
                logging.error(f"Ошибка инициализации GraphDB: {e}", exc_info=True)
                self.graph_db = None # Работаем без графа

        if not self.graph_db:
            logging.critical("GraphDB не инициализирован, работа с графом невозможна. Проверьте переменные окружения NEO4J_URI и NEO4J_USERNAME.")
            raise RuntimeError("GraphDB не инициализирован. Проверьте параметры подключения к Neo4j.")
        self.n_results_graph_query = n_results_graph_query
        logging.info(f"KnowledgeDB инициализирован. Количество результатов для графовых запросов: {self.n_results_graph_query}")

    def _get_top_entity_names(self, vector_results: Optional[Dict[str, Any]], n_top: Optional[int] = None, tresholder = None) -> List[str]:
        """
        Вспомогательная функция для извлечения имен сущностей из результатов векторного поиска.
        Args:
            vector_results: Результаты векторного поиска, полученные из VectorDB.
            n_top: Максимальное количество имен для извлечения (по умолчанию self.n_results_graph_query).
            treshholder: Пороговое значение для фильтрации результатов по расстоянию.
        """
        names = []
        if not vector_results or not vector_results.get('ids') or not vector_results['ids'][0]:
            return names

        limit = n_top if n_top is not None else self.n_results_graph_query


        # Результаты ChromaDB могут содержать метаданные в разном виде в зависимости от include
        ids = vector_results['ids'][0]
        metadatas = vector_results.get('metadatas', [[]])[0] # [[meta1, meta2,...]]
        
        if tresholder:
            # Фильтруем метаданные по tresholder
            distances = vector_results.get('distances', [[]])[0]
            ids = [ids[i] for i, dist in enumerate(distances) if dist > tresholder]
            metadatas = [metadatas[i] for i, dist in enumerate(distances) if dist > tresholder]

        if metadatas and len(metadatas) == len(ids):
             # Предполагаем, что имя хранится в 'entity_name'
             names = [meta.get('entity_name') for meta in metadatas[:limit] if meta and meta.get('entity_name')]
        elif ids:
             # Если метаданных нет, но есть ID (которые мы установили равными именам)
             names = ids[:limit]

        # Фильтруем None или пустые строки, если они как-то попали
        return [name for name in names if name]

    def load(self):
        '''Загрузка данных в KnowledgeDB из директории, генерация эмбеддингов и графа.'''
        # Загрузка данных в VectorDB
        json_data = GraphExtractor(config_path="./config.yaml").update()
        self.vector_db.add_entities_batch(json_data['entities'])
        # Загрузка данных в GraphDB
        if self.graph_db:
            self.graph_db.load_from_json(json_data)
        else:
            logging.warning("GraphDB не инициализирован, загрузка данных в граф невозможна.")

    def search_classic(self, query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Классический поиск: использует только векторную базу описаний сущностей.

        Args:
            query_text: Текст запроса.
            n_results: Количество возвращаемых результатов.

        Returns:
            Список словарей с метаданными найденных сущностей (включая имя и описание).
        """
        logging.info(f"Выполнение классического поиска: '{query_text}' (n_results={n_results})")
        results = self.vector_db.query_descriptions(query_text, n_results=n_results)
        if results and results.get('metadatas') and results['metadatas'][0]:
            # Возвращаем список словарей метаданных
            return results['metadatas'][0]
        else:
            logging.warning("Классический поиск не дал результатов или произошла ошибка.")
            return []

    def search_extended(self, query_text: str, n_vector_results: int = 10, tresholder = None) -> List[Optional[Dict[str, Any]]]:
        """
        Расширенный поиск: поиск по именам -> запрос полной информации из графа для топ-N.

        Args:
            query_text: Текст запроса для поиска имен.
            n_vector_results: Количество результатов, запрашиваемых у векторной БД.

        Returns:
            Список словарей, где каждый словарь представляет полную информацию
            об узле и его связях из графовой БД, или None, если узел не найден в графе.
        """
        logging.info(f"Выполнение расширенного поиска: '{query_text}' (n_vector={n_vector_results}, n_graph={self.n_results_graph_query})")
        if not self.graph_db:
             logging.error("Расширенный поиск невозможен: GraphDB не инициализирован.")
             return []

        vector_results = self.vector_db.query_names(query_text, n_results=n_vector_results)
        top_names = self._get_top_entity_names(vector_results, tresholder=tresholder) # Использует self.n_results_graph_query

        if not top_names:
            logging.warning("Расширенный поиск: не найдено релевантных имен в векторной БД.")
            return []

        graph_results = []
        logging.debug(f"Запрос к графовой БД для имен: {top_names}")
        for name in top_names:
            try:
                node_data = self.graph_db.get_node_with_relationships(name)
                graph_results.append(node_data) # Может быть None, если узел не найден
            except Exception as e:
                logging.error(f"Ошибка при запросе узла '{name}' из графовой БД: {e}", exc_info=True)
                graph_results.append(None) # Добавляем None в случае ошибки

        return graph_results
    
    def search_names(self, query_text: str, n_vector_results: int = 10, tresholder = None) -> List[Optional[Dict[str, Any]]]:
        """
        Поиск имен: поиск имен в векторной БД.
        """

        vector_results = self.vector_db.query_names(query_text, n_results=n_vector_results)
        top_names = self._get_top_entity_names(vector_results, tresholder = tresholder)

        if not top_names:
            logging.warning("Поиск имен: не найдено релевантных имен в векторной БД.")
            return []
        
        return top_names
    
    def search_lite(self, query_text: str, n_vector_results: int = 10, tresholder = None) -> List[Optional[Dict[str, Any]]]:
        """
        Краткий поиск: поиск по именам -> запрос частичной информации из графа для топ-N
                     (без описаний соседних узлов).

        Args:
            query_text: Текст запроса для поиска имен.
            n_vector_results: Количество результатов, запрашиваемых у векторной БД.

        Returns:
            Список словарей с частичной информацией об узле и его связях.
        """
        logging.info(f"Выполнение быстрого поиска: '{query_text}' (n_vector={n_vector_results}, n_graph={self.n_results_graph_query})")
        if not self.graph_db:
            logging.error("Быстрый поиск невозможен: GraphDB не инициализирован.")
            return []

        vector_results = self.vector_db.query_names(query_text, n_results=n_vector_results)
        top_names = self._get_top_entity_names(vector_results, tresholder=tresholder)

        if not top_names:
            logging.warning("Быстрый поиск: не найдено релевантных имен в векторной БД.")
            return []

        graph_results = []
        logging.debug(f"Запрос к графовой БД для имен (быстрый поиск): {top_names}")
        for name in top_names:
            try:
                node_data = self.graph_db.get_relationships_only(name)
                graph_results.append(node_data)
            except AttributeError:
                 logging.error(f"Метод get_node_with_relationships не найден в GraphDB или вернул некорректный тип.")
                 graph_results.append(None)
            except Exception as e:
                logging.error(f"Ошибка при запросе узла '{name}' из графовой БД (быстрый поиск): {e}", exc_info=True)
                graph_results.append(None)

        return graph_results

    def search_deep(self, query_text: str, n_vector_results: int = 10, tresholder =  None) -> List[Optional[Dict[str, Any]]]:
        """
        Глубокий поиск: поиск по описаниям -> извлечение имен -> запрос полной информации из графа для топ-N.

        Args:
            query_text: Текст запроса для поиска описаний.
            n_vector_results: Количество результатов, запрашиваемых у векторной БД.

        Returns:
            Список словарей с полной информацией об узле и его связях.
        """
        logging.info(f"Выполнение глубокого поиска: '{query_text}' (n_vector={n_vector_results}, n_graph={self.n_results_graph_query})")
        if not self.graph_db:
            logging.error("Глубокий поиск невозможен: GraphDB не инициализирован.")
            return []

        vector_results = self.vector_db.query_descriptions(query_text, n_results=n_vector_results)
        # Извлекаем имена из метаданных результатов поиска по описаниям
        top_names = self._get_top_entity_names(vector_results, tresholder=tresholder) # Использует self.n_results_graph_query

        if not top_names:
            logging.warning("Глубокий поиск: не найдено релевантных описаний/имен в векторной БД.")
            return []

        graph_results = []
        logging.debug(f"Запрос к графовой БД для имен (глубокий поиск): {top_names}")
        for name in top_names:
            try:
                node_data = self.graph_db.get_node_with_relationships(name)
                graph_results.append(node_data)
            except Exception as e:
                logging.error(f"Ошибка при запросе узла '{name}' из графовой БД (глубокий поиск): {e}", exc_info=True)
                graph_results.append(None)

        return graph_results

    # --------------------------------------
    def close_graph_db(self):
        """Закрывает соединение с графовой базой данных, если оно было открыто."""
        if self.graph_db and hasattr(self.graph_db, 'close'):
            try:
                self.graph_db.close()
                logging.info("Соединение с GraphDB закрыто.")
            except Exception as e:
                 logging.error(f"Ошибка при закрытии соединения с GraphDB: {e}", exc_info=True)


    # --------------------------------------
    def add_entity(self, entity: Entity):
        '''Add entity in all databases'''
        self.vector_db.add_entity(entity)
        self.graph_db.add_node(entity)
    
    def upsert_entity(self, entity: Entity):
        """Update or insert entity in all databases"""
        self.vector_db.add_entity(entity)
        self.graph_db.upsert_entity(entity)
    
    def add_relationship(self, rel: Relationship):
        """Add relationship to Graph Databse
        Args:
            rel: Relationship object containing source, target, type and description.
        """
        if not self.graph_db:
            logging.error("Добавление связи невозможно: GraphDB не инициализирован.")
            return
        
        try:
            self.graph_db.add_relationship(rel.source, rel.target, rel.type, rel.description)
        except Exception as e:
            logging.error(f"Ошибка при добавлении связи '{rel}': {e}", exc_info=True)
    
    def get_all_locations(self) -> List[str]:
        """
        Получение всех локаций из графовой БД.

        Returns:
            Список имен всех локаций.
        """
        if not self.graph_db:
            logging.error("Получение локаций невозможно: GraphDB не инициализирован.")
            return []
        
        results = self.graph_db._query("MATCH (e:Entity) WHERE e.type IN ['LOCATION', 'SPOT', 'BUILDING', 'COUNTRY'] RETURN e.name, e.description, e.type")

        if not results:
            logging.warning("Не найдено локаций в графовой БД.")
            return []
        
        return [ "{0} ({2}): {1}".format(*r.values()) for r in results]


# --- Пример использования ---
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG, 
        format='%(asctime)s - %(levelname)s - %(message)s', 
        filename='logs.log',
        encoding='utf-8'
        )

    # logging.getlogging("urllib3").setLevel(logging.ERROR)  
    # logging.getlogging("urllib3").disabled = True  # Отключаем urllib3, если он используется в других модулях

    logging.debug("DEBAG MODE")

    print("Запуск примера KnowledgeDB...")

    # Предполагается, что VectorDB уже содержит данные из примера vector_db_module.py
    # и GraphDB содержит соответствующие узлы и связи.

    # Инициализация KB
    kb = KnowledgeDB(db_name='staticdb')
    kb.load()  # Загрузка данных в VectorDB и GraphDB

    # Проверка подключения к графу
    if not kb.graph_db:
        print("\nWARNING: GraphDB не инициализирован, графовые поиски не будут работать.")
    else:
        query = input("Введите запрос для поиска в KnowledgeDB (или 'exit' для выхода): ")
        if query.lower() == 'exit':
            print("Выход из примера KnowledgeDB.")
            exit(0)
        print(f"\n--- Поиск Classic ({query}) ---")
        classic_res = kb.search_classic(query, n_results=2)
        print(json.dumps(classic_res, indent=2, ensure_ascii=False))

        print(f"\n--- Поиск Extended ({query}) ---")
        extended_res = kb.search_extended(query, n_vector_results=2)
        print(json.dumps(extended_res, indent=2, ensure_ascii=False)) # Печатаем результат из графа

        print(f"\n--- Поиск Lite ({query}) ---")
        fast_res = kb.search_lite(query, n_vector_results=2)
        print(json.dumps(fast_res, indent=2, ensure_ascii=False)) # Результат из графа без описаний соседей

        print(f"\n--- Поиск Deep ({query}) ---")
        deep_res = kb.search_deep(query, n_vector_results=2)
        print(json.dumps(deep_res, indent=2, ensure_ascii=False)) # Результат из графа

        # Закрытие соединения с графом
        kb.close_graph_db()

    print("\nПример KnowledgeDB завершен.")