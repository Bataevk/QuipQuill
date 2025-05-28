from neo4j import GraphDatabase
from extractor import Entity

class GraphDB:
    """
    Класс для работы с графовой базой данных Neo4j.
    Поддерживает статический и динамический графы.
    """
    def __init__(self, uri, user, password, database_name = None):
        """Инициализация подключения к Neo4j."""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database_name = database_name

    def __del__(self):
        """Закрытие соединения при удалении объекта."""
        if hasattr(self, 'driver'):
            self.driver.close()

    def close(self):
        """Закрытие соединения с Neo4j."""
        self.driver.close()

    def load_from_json(self, json_data):
        """Загрузка данных из JSON в Neo4j."""
        with self.driver.session(database=self.database_name) as session:
            # Добавление узлов
            for entity in json_data["entities"]:
                session.run(
                    "MERGE (e:Entity {name: $name}) "
                    "ON CREATE SET e.description = $description, e.type = $type",
                    name=entity["name"], description=entity["description"], type=entity["type"]
                )
            # Добавление связей
            for rel in json_data["relationships"]:
                # Используем MERGE для создания связи
                # session.run(
                #     "MATCH (source:Entity {name: $source}), (target:Entity {name: $target}) "
                #     "MERGE (source)-[r:RELATED_TO]->(target) "
                #     "ON CREATE SET r.description = $description",
                #     source=rel["source"], target=rel["target"], description=rel["description"]
                # )

                # Используем MERGE для создания или обновления связи в случае повторяющихся узлов
                # session.run(
                #     "MERGE (a:Entity {name: $source})"
                #     "MERGE (b:Entity {name: $target})"
                #     "MERGE (a)-[r:RELATED_TO]->(b)"
                #     "SET r.description = coalesce(r.description + '; ', '') + $description;",
                #     source=rel["source"], target=rel["target"], description=rel["description"]
                # )
                # Используем MERGE для создания связи, даже если узлы уже существуют
                session.run(
                    "MERGE (a:Entity {name: $source}) "
                    "MERGE (b:Entity {name: $target}) "
                    "CREATE (a)-[r:RELATED_TO {description: $description, type: $type}]->(b)",
                    source=rel["source"], target=rel["target"], description=rel["description"], type=rel.get("type", "related_to")
                )

    def get_node_by_id(self, node_id):
        """Получение узла по ID (name)."""
        with self.driver.session(database=self.database_name) as session:
            result = session.run("MATCH (e:Entity {name: $name}) RETURN e", name=node_id)
            record = result.single()
            return record["e"] if record else None

    def get_node_description(self, node_id):
        """Получение описания узла по ID (name)."""
        node = self.get_node_by_id(node_id)
        if node:
            return node.get("description", "")
        return None

    def add_node(self, entity):
        """Добавление нового узла."""
        with self.driver.session(database=self.database_name) as session:
            session.run(
                "CREATE (e:Entity {name: $name, description: $description})",
                name=entity["name"], description=entity["description"]
            )

    def update_node(self, node_id, new_description):
        """Редактирование узла."""
        with self.driver.session(database=self.database_name) as session:
            session.run(
                "MATCH (e:Entity {name: $name}) SET e.description = $description",
                name=node_id, description=new_description
            )
    def add_or_update_entity(self, entity: Entity):
        """
        Добавление новой сущности или обновление существующей.
        Использует MERGE для поиска или создания узла.
        """
        with self.driver.session(database=self.database_name) as session:
            session.run(
                """
                MERGE (e:Entity {name: $name})
                ON CREATE SET e.description = $description
                ON MATCH SET e.description = $description
                """,
                name=entity["name"], description=entity["description"]
            )
    
    def check_node_exists(self, node_id):
        """Проверка существования узла по ID (name)."""
        with self.driver.session(database=self.database_name) as session:
            result = session.run("MATCH (e:Entity {name: $name}) RETURN count(e) > 0", name=node_id)
            record = result.single()
            return record[0] if record else False

    def delete_node(self, node_id):
        """Удаление узла и всех его связей."""
        with self.driver.session(database=self.database_name) as session:
            session.run(
                "MATCH (e:Entity {name: $name}) DETACH DELETE e",
                name=node_id
            )

    def add_relationship(self, source_id, target_id, description):
        """Добавление новой связи."""
        with self.driver.session(database=self.database_name) as session:
            session.run(
                "MATCH (source:Entity {name: $source}), (target:Entity {name: $target}) "
                "CREATE (source)-[r:RELATED_TO {description: $description}]->(target)",
                source=source_id, target=target_id, description=description
            )

    def update_relationship(self, source_id, target_id, new_description):
        """Обновление связи."""
        with self.driver.session(database=self.database_name) as session:
            session.run(
                "MATCH (source:Entity {name: $source})-[r:RELATED_TO]->(target:Entity {name: $target}) "
                "SET r.description = $description",
                source=source_id, target=target_id, description=new_description
            )

    def delete_relationship(self, source_id, target_id):
        """Удаление связи."""
        with self.driver.session(database=self.database_name) as session:
            session.run(
                "MATCH (source:Entity {name: $source})-[r:RELATED_TO]->(target:Entity {name: $target}) "
                "DELETE r",
                source=source_id, target=target_id
            )

    def _format_entity_relationships(self, data):
        """
        Форматирует результат из get_node_with_relationships по следующей структуре:
        
        # MC
        {имя главной сущности} - {описание главной сущности}
        
        # Others
        - {имя связанной сущности} - {описание, если есть}
        - ...
        
        # Relationships
        - {источник} {описание связи} {цель}
        - ...
        """
        if not data:
            return "Узел не найден"

        main_node = data["node"]
        main_name = main_node.get("name", "")
        main_desc = main_node.get("description", self.get_node_description(main_name)).strip() if main_name else "unknown name"

        # Формируем строку для главной сущности
        output = "# MC\n"
        output += f"{main_name} - {main_desc}\n\n"

        # Собираем связанные сущности и связи
        others = {}
        rels = []

        for item in data["relationships"]:
            if not item:
                continue
            rel = item.get("relationship", {})
            related = item.get("related", {})
            relation_desc = rel.get("description", "").strip() if rel else "linked"
            related_name = related.get("name", "") if related else None
            related_desc = related.get("description", "").strip() if related else None
            # Запоминаем информацию о связанных сущностях (если сущность повторяется, оставляем одно описание)
            if related_desc and related_name:
                others[related_name] = related_desc
            # Формируем строку для связи
            rels.append(f"{main_name} {relation_desc} {related_name}")

        # Формируем раздел связанных сущностей
        output += "# Others\n"
        for name, desc in others.items():
            line = f"- {name}"
            if desc:
                line += f" - {desc}"
            output += line + "\n"

        # Формируем раздел связей
        output += "\n# Relationships\n"
        for r in rels:
            output += f"- {r}\n"

        return output
    
    def _format_relationships_only(self, data):
        """
        Форматирует результат запроса, возвращая только список связей в следующем виде:
        - {источник} {описание связи} {цель}
        """
        if not data:
            return "Узел не найден"
        main_node = data["node"]
        main_name = main_node.get("name", "")
        rels = []
        for item in data["relationships"]:
            if not item:
                continue
            rel = item.get("relationship", {})
            related = item.get("related", {})
            relation_desc = rel.get("description", "").strip() if rel else "linked"
            related_name = related.get("name", "") if related else ""
            rels.append(f"- {main_name} {relation_desc} {related_name}")
        return "\n".join(rels)
    
    def _get_relationships(self, node_id):
        """Получение всех связей узла."""
        with self.driver.session(database=self.database_name) as session:
            # Односторонний поиск
            # result = session.run(
            #     "MATCH (e:Entity {name: $name}) "
            #     "OPTIONAL MATCH (e)-[r]->(related:Entity) "
            #     "RETURN e, collect({relationship: r, related: related}) as relationships",
            #     name=node_id
            # )
            # Двусторонний поиск
            result = session.run(  
                "MATCH (e:Entity {name: $name}) "  
                "OPTIONAL MATCH (e)-[r]->(related:Entity) "  
                "OPTIONAL MATCH (e)<-[r2]-(related_from:Entity) "  
                "RETURN e, "  
                "collect({relationship: r, related: related}) as relationships_to, "  
                "collect({relationship: r2, related: related_from}) as relationships_from",  
                name=node_id  
            )  
            record = result.single()
            if record:
                node = record["e"]
                relationships = record["relationships_to"] + record["relationships_from"]
                return {"node": node, "relationships": relationships}
            return None

    def get_node_with_relationships(self, node_id, to_str=True):
        """Получение узла и связанных с ним узлов вместе со связями."""
        data = self._get_relationships(node_id)
        if not data:
            return "Узел не найден" if to_str else None
        if to_str:
            return self._format_entity_relationships(data)
        return data
        
    def get_relationships_only(self, node_id, to_str=True):
        """Получение только связей узла."""
        data = self._get_relationships(node_id)
        if not data:
            return "Узел не найден" if to_str else None
        if to_str:
            return self._format_relationships_only(data)
        return data

    def ping(self):
        """Проверка соединения с базой данных."""
        with self.driver.session(database=self.database_name) as session:
            try:
                session.run("RETURN 1")
                return True
            except Exception as e:
                print(f"Error pinging Neo4j: {e}")
                return False
    
    def get_linked_nodes(self, node_id, relationship_type=None):
        """
        Получение связанных узлов для заданного узла.
        Если relationship_type не указан, возвращает все связанные узлы.
        """
        with self.driver.session(database=self.database_name) as session:
            if relationship_type:
                result = session.run(
                    "MATCH (e:Entity {name: $name})-[r]->(related:Entity) "
                    "WHERE e.type = $relationship_type "
                    "RETURN e",
                    name=node_id, relationship_type=relationship_type
                )
            else:
                result = session.run(
                    "MATCH (e:Entity {name: $name})-[r]->(related:Entity) "
                    "RETURN e",
                    name=node_id
                )
            return [record["name"] for record in result]
    
    def _query(self, query: str, parameters: dict = None):
        """
        Выполнение произвольного запроса к базе данных.
        Возвращает список записей.
        """
        with self.driver.session(database=self.database_name) as session:
            result = session.run(query, parameters or {})
            return [record for record in result]