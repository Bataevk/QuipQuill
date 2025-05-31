from KnowledgeDBMod import KnowledgeDB
from DynDBMod import DynDBMod
import logging
from my_dataclasses import Entity
from typing import List

class Manager:
    def __init__(self, load = False, static_db_name = "staticdb", dynamic_db_name = "dynamicdb"):
        self.static_database = KnowledgeDB(db_name=static_db_name)
        self.dynamic_database = DynDBMod(db_name=dynamic_db_name)
        # Состояние, которое будет хранить текущее главного объекта (быстрый доступ к данным, которые хранятся в динамическом графе)
        # Например, текущее местоположение игрока, инвентарь и т.д.
        self.state = {}

        # Загрузка базы знаний из файла
        if load:
            self.load()
    
    def load(self):
        self.static_database.load()

    def close(self):
        if hasattr(self, 'static_database') and self.static_database:
            self.static_database.close_graph_db()
        if hasattr(self, 'dynamic_database') and self.dynamic_database:
            self.dynamic_database.close_graph_db()
    
    def __del__(self):
        self.close()

    
    def initalize_agent(self, agent_name: str = "player") -> str:
        """
        Initializes the agent in the dynamic graph.
        If a agent name is provided, it uses that name; otherwise, it uses the default agent name.
        """
        agent_name = agent_name.lower()
        return self.dynamic_database.add_agent(agent_name)

    def _remove_duplicate_nodes(self, old_entities: List[Entity], new_entities: List[Entity], uniques_types =  ["LOCATED_IN", "HAS_ITEM"]) -> str:
        """
        Removes duplicate nodes from the old_entities list based on the new_entities list.
        If a node in old_entities has the same name as a node in new_entities, it is removed.
        """
        if not old_entities:
            return new_entities
        if not new_entities:
            return old_entities
        

        # Создаем множество для хранения существующих сущностей из new_entities
        existing_entities = set()
        for actual_rel in new_entities:
            existing_entities.add((actual_rel.type, actual_rel.source, actual_rel.target))

        # Удаляем дубликаты из old_entities, которые уже есть в existing_entities
        dedup_old_entities = []
        for rel in old_entities:
            if (rel.type, rel.source, rel.target) not in existing_entities:
                dedup_old_entities.append(rel)
                existing_entities.add((rel.type, rel.source, rel.target)) # Добавляем новый элемент в множество
        
        # Удаляем дубликаты, где type и target совпадают и type совпадает с "LOCATED_IN" или "HAS_ITEM" (сохраняем из new_entities)
        dedup_old_entities = [
            rel for rel in dedup_old_entities 
            if not any(
                (rel.type == new_rel.type and rel.target == new_rel.target and new_rel.type in uniques_types)
                for new_rel in new_entities
            )
        ]
        
        return new_entities + dedup_old_entities  # Возвращаем объединенный список новых и уникальных старых сущностей

    def update(self):
        """
        Updates the dynamic database - removes not linked nodes.
        """

        self.dynamic_database.delete_orphaned_nodes()


    # TOOLS METHODS
    def get_agent_state(self, agent_name: str = "player") -> str:
        """
        Returns the current state of the agent, including location and inventory.
        """
        agent_name = agent_name.lower()
        locations, inventory = self.dynamic_database.get_agent_state(agent_name)
        if not locations and not inventory:
            return f"AGENT - '{agent_name}' has no known state."
        state_description = f"AGENT - '{agent_name}' current state:\n"
        state_description += f"Current location(s): {', '.join([location.name for location in locations])}\n"
        if inventory:
            str_inventory = ',\n'.join(list(map( lambda e: f"name: {e.name} - decription: {e.description}", inventory)))
            state_description += f"Inventory:\n{str_inventory}\n"
        return state_description.strip()  # Удаляем лишние пробелы в конце строки

    def get_agent_inventory(self, agent_name: str = "player") -> str:
        """
        Returns the current inventory of the agent.
        """
        agent_name = agent_name.lower()
        inventory = self.dynamic_database.get_agent_inventory(agent_name)

        if not inventory:
            return f"AGENT - '{agent_name}' has no items in their inventory."
        
        # Форматируем список предметов в строку
        str_inventory = ',\n'.join(list(map( lambda e: f"name: {e.name} - decription: {e.description}", inventory)))

        return f"AGENT - '{agent_name}' has the following items in their inventory:\n{str_inventory}."
    
    def add_item_to_inventory(self, item_name: str, agent_name: str = "player") -> str:
        """
        Adds an item to the agent's inventory.
        If a agent name is provided, it uses that name; otherwise, it uses the default agent name.
        """
        item_name = item_name.lower()
        if not self.static_database.graph_db.has_node(item_name):
            # Проверяем, существует ли предмет в статической базе данных
            return f"Item '{item_name}' does not exist in the static database."
        
        if not self.dynamic_database.graph_db.has_node(item_name):
            # Если предмет не существует в динамической базе, добавляем его
            self.dynamic_database.upsert_entity(self.static_database.graph_db.get_node_by_id(item_name))
        
        agent_name = agent_name.lower()
        return self.dynamic_database.add_item_to_inventory(item_name, agent_name)
    
    def move_item_from_inventory(self, item_name, agent_name=None):
        """
        Deletes an item from the agent's inventory.
        Put the item in the agent's current location.
        If a agent name is provided, it uses that name; otherwise, it uses the default agent name.
        """
        if agent_name is None:
            agent_name = self.agent_name
        
        agent_name = agent_name.lower()
        item_name = item_name.lower()

        # Ensure agent name is in lowercase for consistency
        if not self.dynamic_database.graph_db.has_node(agent_name):
            return f"AGENT - '{agent_name}' does not exist in the dynamic graph."
        
        # Проверяем, существует ли предмет
        if not self.dynamic_database.graph_db.has_node(item_name):
            return f"Item '{item_name}' does not exist in the dynamic graph."
        
        # Проверяем, есть ли связь между агентом и предметом
        if not self.dynamic_database.graph_db.has_relationship(agent_name, item_name, "HAS_ITEM"):
            return f"AGENT - '{agent_name}' does not have the item '{item_name}' in their inventory."
        
        # Получаем текущее местоположение агента
        current_location = self.dynamic_database.get_agent_location(agent_name)
        if not current_location:
            return f"AGENT - '{agent_name}' has no known location to put the item '{item_name}'."
        


        # Удаляем связь между агентом и предметом
        self.dynamic_database.graph_db.delete_relationship(agent_name, item_name, "HAS_ITEM")

        # Добавляем связь между предметом и текущим местоположением агента
        self.dynamic_database.graph_db.add_relationship(item_name, current_location[0].name, f"Located in {current_location[0].name}", "LOCATED_IN")


        return f'AGENT - {agent_name} has moved the item "{item_name}" from their inventory to their current location: {current_location}.'
    
    def describe_entity(self, entity_name: str) -> str:
        """
        Returns a description of the entity.
        If the entity is not found, it returns an error message.
        """
        entity_name = entity_name.lower()
        main_entity = self.dynamic_database.graph_db.get_node_by_id(entity_name)
        if not main_entity:
            # Если сущность не найдена в динамической базе, проверяем в статической базе
            main_entity = self.static_database.graph_db.get_node_by_id(entity_name)

        if not main_entity:
            return f"Entity '{entity_name}' not found in the static database."

        related_entities = self.static_database.graph_db.get_linked_nodes_by_type(entity_name)

        actual_related_entities = self.dynamic_database.graph_db.get_linked_nodes_by_type(entity_name)

        results_entity = self._remove_duplicate_nodes(related_entities, actual_related_entities)

        return f"Entity '{main_entity.name}' - {main_entity.description}\n" + \
                "Related entities:\n" + \
                "\n".join([f"{rel.name} - {rel.description}" for rel in results_entity]) if results_entity else "No related entities found."
    
    def edit_entity(self, entity_name: str, new_description: str) -> str:
        """
        Edits the description of the entity.
        If the entity is not found, it returns an error message.
        """
        entity_name = entity_name.lower()
        main_entity = self.static_database.graph_db.get_node_by_id(entity_name)

        if not main_entity:
            return f"Entity '{entity_name}' not found in the static database."

        # Обновляем описание сущности
        main_entity.description = new_description
        self.dynamic_database.upsert_entity(main_entity)

        return f"Entity '{main_entity.name}' has been updated with new description: {new_description}."

    def get_agent_location(self, agent_name: str = "player") -> str:
        """
        Returns the current location of the agent.
        """
        agent_name = agent_name.lower()
        locations = self.dynamic_database.get_agent_location(agent_name)
        
        if not locations:
            return f"AGENT - '{agent_name}' has no known location."
        
        if len(locations) == 1:
            return f"AGENT - '{agent_name}' is currently located in: {locations[0].name}"
        
        return f"AGENT - '{agent_name}' is currently located in: {', '.join([location.name for location in locations])}."
    
    def move_agent(self, new_location, agent_name=None):
        """
        Moves the agent to a new location.
        If a agent name is provided, it uses that name; otherwise, it uses the default agent name.
        """
        if agent_name is None:
            agent_name = self.agent_name

        agent_name = agent_name.lower()  # Ensure agent name is in lowercase for consistency
        new_location = new_location.lower()  # Ensure new location is in lowercase for consistency
        
        # Проверяем, существует ли новая локация
        if not self.static_database.graph_db.has_node(new_location):
            return f"Location '{new_location}' does not exist in the static database."
        
        if not self.dynamic_database.graph_db.has_node(agent_name):
            return f"AGENT - '{agent_name}' does not exist in the dynamic graph."
        
        
        # Проверяем, существует ли новая локация в динамическом графе
        if not self.dynamic_database.graph_db.has_node(new_location):
            # Если новая локация не существует в динамической базе, добавляем ее
            self.dynamic_database.upsert_entity(self.static_database.graph_db.get_node_by_id(new_location))
        
        # Удаляем старую локацию игрока
        self.dynamic_database.graph_db.delete_relationships_by_type(agent_name, "LOCATED_IN")
        
        # Добавляем новую локацию
        self.dynamic_database.graph_db.add_relationship(agent_name, new_location, f"Located in {new_location}", "LOCATED_IN")
        
        return f"AGENT - '{agent_name}' moved to {new_location}."

    def get_all_locations(self) -> str:
        """
        This function returns all locations in the game.
        """
        locations = self.static_database.get_all_locations()
        if not locations:
            return 'No locations found in the database.'
        # Форматируем список локаций в строку
        locations = ";\n".join([str(l) for l in locations])
        return f'Available locations: \n{locations}'
    
    def search_deep(self, query: str, n_names = 5, tresholder = 0.8) -> str:
        """
        Searches for a query in the static database and returns the results.
        """
        names = self.static_database.search_names(query, n_vector_results=n_names, tresholder=tresholder)

        if not names:
            return f"No results found for query '{query}'."
        
        results = "Search results for query '{}':\n".format(query)
        for name in names:
            results += self.describe_entity(name) + "\n"

        return results.strip()  # Удаляем лишние пробелы в конце строки


    

    




if __name__ == "__main__":
    # Загрузка переменных окружения из .env файла
    from dotenv import load_dotenv
    load_dotenv()

    # Логирование 
    logging.basicConfig(
        level=logging.DEBUG, 
        format='%(asctime)s - %(levelname)s - %(message)s', 
        filename='logs.log',
        encoding='utf-8'
    )

    # Инициализация менеджера базы знаний
    manager = Manager(
        load=True
        )
    
    # Пример использования методов менеджера
    # print(manager.get_all_locations())

    # # Пример добавления узла (для динамического графа)
    # new_entity = {"name": "Игрок", "description": "Главный герой"}
    # graph_module.add_node(new_entity)

    # # Пример добавления связи (обновление состояния мира)
    # graph_module.add_relationship("Игрок", "Меч", "Имеет")

    # # Пример получения узла и связей
    # node_with_rels = graph_module.get_node_with_relationships("semyon")
    # print(node_with_rels)

