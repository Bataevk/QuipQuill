import os
from neo4j import GraphDatabase
from typing import Dict, List, Any

class Neo4jSaver:
    """
    Класс для сохранения графа знаний в базу данных Neo4j.
    """

    def __init__(self, uri: str, user: str, password: str):
        """
        Инициализирует подключение к Neo4j.

        Args:
            uri: URI базы данных (например, "bolt://localhost:7687").
            user: Имя пользователя Neo4j.
            password: Пароль Neo4j.
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        """Закрывает соединение с Neo4j."""
        self.driver.close()

    def save_graph(self, graph_data: Dict[str, List[Dict[str, Any]]]):
        """
        Сохраняет граф (сущности и связи) в Neo4j.

        Args:
            graph_data: Словарь с ключами "entities" и "relationships".
        """
        with self.driver.session() as session:
            # Сохранение сущностей
            for entity in graph_data["entities"]:
                session.write_transaction(self._create_entity, entity)
            # Сохранение связей
            for relationship in graph_data["relationships"]:
                session.write_transaction(self._create_relationship, relationship)

    @staticmethod
    def _create_entity(tx, entity: Dict[str, Any]):
        """
        Создает или обновляет узел сущности в Neo4j.

        Args:
            tx: Транзакция Neo4j.
            entity: Словарь с полями "name" и "description".
        """
        query = (
            "MERGE (e:Entity {name: $name}) "
            "ON CREATE SET e.description = $description "
            "ON MATCH SET e.description = coalesce(e.description, $description)"
        )
        tx.run(query, name=entity["name"], description=entity["description"])

    @staticmethod
    def _create_relationship(tx, relationship: Dict[str, Any]):
        """
        Создает или обновляет отношение между узлами в Neo4j.

        Args:
            tx: Транзакция Neo4j.
            relationship: Словарь с полями "source", "target" и "description".
        """
        query = (
            "MATCH (source:Entity {name: $source}), (target:Entity {name: $target}) "
            "MERGE (source)-[r:RELATED_TO]->(target) "
            "ON CREATE SET r.description = $description "
            "ON MATCH SET r.description = coalesce(r.description, $description)"
        )
        tx.run(query, source=relationship["source"], target=relationship["target"], description=relationship["description"])

# Пример использования
if __name__ == "__main__":
    # Настройки подключения из переменных окружения
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")

    # Создание экземпляра
    saver = Neo4jSaver(uri, user, password)

    # Пример данных от GraphExtractor
    graph_data = {
        "entities": [
            {"name": "Игрок", "description": "Главный герой"},
            {"name": "Меч", "description": "Оружие"}
        ],
        "relationships": [
            {"source": "Игрок", "target": "Меч", "description": "взял"}
        ]
    }

    # Сохранение в Neo4j
    saver.save_graph(graph_data)
    saver.close()