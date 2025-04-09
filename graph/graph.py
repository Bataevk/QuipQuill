from neo4j import GraphDatabase

class GraphDB:
    """
    Класс для работы с графовой базой данных Neo4j.
    Поддерживает статический и динамический графы.
    """
    def __init__(self, uri, user, password):
        """Инициализация подключения к Neo4j."""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def __del__(self):
        """Закрытие соединения при удалении объекта."""
        if hasattr(self, 'driver'):
            self.driver.close()

    def close(self):
        """Закрытие соединения с Neo4j."""
        self.driver.close()

    def load_from_json(self, json_data):
        """Загрузка данных из JSON в Neo4j."""
        with self.driver.session() as session:
            # Добавление узлов
            for entity in json_data["entities"]:
                session.run(
                    "MERGE (e:Entity {name: $name}) "
                    "ON CREATE SET e.description = $description",
                    name=entity["name"], description=entity["description"]
                )
            # Добавление связей
            for rel in json_data["relationships"]:
                session.run(
                    "MATCH (source:Entity {name: $source}), (target:Entity {name: $target}) "
                    "MERGE (source)-[r:RELATED_TO]->(target) "
                    "ON CREATE SET r.description = $description",
                    source=rel["source"], target=rel["target"], description=rel["description"]
                )

    def get_node_by_id(self, node_id):
        """Получение узла по ID (name)."""
        with self.driver.session() as session:
            result = session.run("MATCH (e:Entity {name: $name}) RETURN e", name=node_id)
            record = result.single()
            return record["e"] if record else None

    def add_node(self, entity):
        """Добавление нового узла."""
        with self.driver.session() as session:
            session.run(
                "CREATE (e:Entity {name: $name, description: $description})",
                name=entity["name"], description=entity["description"]
            )

    def update_node(self, node_id, new_description):
        """Редактирование узла."""
        with self.driver.session() as session:
            session.run(
                "MATCH (e:Entity {name: $name}) SET e.description = $description",
                name=node_id, description=new_description
            )

    def delete_node(self, node_id):
        """Удаление узла и всех его связей."""
        with self.driver.session() as session:
            session.run(
                "MATCH (e:Entity {name: $name}) DETACH DELETE e",
                name=node_id
            )

    def add_relationship(self, source_id, target_id, description):
        """Добавление новой связи."""
        with self.driver.session() as session:
            session.run(
                "MATCH (source:Entity {name: $source}), (target:Entity {name: $target}) "
                "CREATE (source)-[r:RELATED_TO {description: $description}]->(target)",
                source=source_id, target=target_id, description=description
            )

    def update_relationship(self, source_id, target_id, new_description):
        """Обновление связи."""
        with self.driver.session() as session:
            session.run(
                "MATCH (source:Entity {name: $source})-[r:RELATED_TO]->(target:Entity {name: $target}) "
                "SET r.description = $description",
                source=source_id, target=target_id, description=new_description
            )

    def delete_relationship(self, source_id, target_id):
        """Удаление связи."""
        with self.driver.session() as session:
            session.run(
                "MATCH (source:Entity {name: $source})-[r:RELATED_TO]->(target:Entity {name: $target}) "
                "DELETE r",
                source=source_id, target=target_id
            )

    def get_node_with_relationships(self, node_id):
        """Получение узла и связанных с ним узлов вместе со связями."""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (e:Entity {name: $name}) "
                "OPTIONAL MATCH (e)-[r]->(related:Entity) "
                "RETURN e, collect({relationship: r, related: related}) as relationships",
                name=node_id
            )
            record = result.single()
            if record:
                node = record["e"]
                relationships = record["relationships"]
                return {"node": node, "relationships": relationships}
            return None
        
    def ping(self):
        """Проверка соединения с базой данных."""
        with self.driver.session() as session:
            try:
                session.run("RETURN 1")
                return True
            except Exception as e:
                print(f"Error pinging Neo4j: {e}")
                return False