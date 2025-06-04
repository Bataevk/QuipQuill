import logging
from typing import List

from src.storymind.databeses.KnowledgeDBMod import KnowledgeDB
from src.storymind.databeses.DynDBMod import DynDBMod
from src.storymind.databeses.DelRelDBMod import DelRelDBM
from src.storymind.my_dataclasses import Entity, Relationship
from scripts.utils import preprocess_text, remove_tokens

class Manager:
    """
    Manager class to manage the game state and interactions with the dynamic and static databases.
    This class provides methods to initialize the agent, manage the game state, and interact with the databases.
    It allows for adding entities, moving agents, managing inventory, and describing entities.
    It also provides methods to load the databases and close the connections.
    The manager can be used to create a game world, manage the player's state, and interact with the game environment.

    Attributes:
        static_database (KnowledgeDB): The static database containing the game world knowledge.
        dynamic_database (DynDBMod): The dynamic database containing the current game state.
        agent_name (str): The name of the agent, default is "player".
        del_rel_db (DelRelDBM): Database for storing deleted relationships.
        state (dict): The current state of the game, such as the agent's location and inventory.
    """
    def __init__(self, load = False, static_db_name = "staticdb", dynamic_db_name = "dynamicdb"):
        """
        Initializes the Manager with the static and dynamic databases.
        Args:
            load (bool): Whether to export the entities from txt files. Default is False.
            static_db_name (str): The name of the static database. Default is "staticdb".
            dynamic_db_name (str): The name of the dynamic database. Default is "dynamicdb".
        """
        self.static_database = KnowledgeDB(db_name=static_db_name)
        self.dynamic_database = DynDBMod(db_name=dynamic_db_name)
        self.agent_name = "player"  # Имя агента по умолчанию, можно изменить при инициализации агента
        self.del_rel_db = DelRelDBM()  # База данных для хранения удаленных связей
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

    def restart(self) -> str:
        """
        Restarts the dynamic graph, clearing all nodes and relationships.
        """
        self.dynamic_database.graph_db._query("MATCH (n) DETACH DELETE n")  # Удаляем все узлы и связи в динамическом графе
        self.del_rel_db.clear_deleted_relationships()  # Очищаем базу удаленных связей

    def initalize_agent(self, agent_name: str = "player", start_location = None) -> str:
        """
        Initializes the agent in the dynamic graph.
        If a agent name is provided, it uses that name; otherwise, it uses the default agent name.
        """
        self.agent_name = agent_name.lower()
        callback = ""
        callback +=  self.dynamic_database.add_agent(self.agent_name)
        if start_location:
            start_location = start_location.lower()
            callback2 = self.dynamic_database.add_location(start_location)
            if callback2:
                # Перемещаем агента в начальную локацию
                callback += "\n" + callback2
                callback += "\n" + self.move_agent(start_location, agent_name=self.agent_name)
        return callback.strip()  # Удаляем лишние пробелы в конце строки

    def _filter_duplicate_rels(self, old_rels: List[Relationship], new_rels: List[Relationship], uniques_types =  ["LOCATED_IN", "HAS_ITEM"]) -> List[Relationship]:
        """
        Removes duplicate nodes from the old_rels list based on the new_rels list.
        If a node in old_rels has the same name as a node in new_rels, it is removed.
        """
        if not old_rels:
            return new_rels
        if not new_rels:
            return old_rels
        
        # Удаляем связи из old_rels, которые уже есть в удаленных связях (del_rel_db)
        old_rels = [rel for rel in old_rels if not self.del_rel_db.has_deleted_relationship(rel)]

        # Создаем множество для хранения существующих сущностей из new_rels
        existing_rels = set()
        for actual_rel in new_rels:
            existing_rels.add((actual_rel.type, actual_rel.source, actual_rel.target))

        # Удаляем дубликаты из old_rels, которые уже есть в existing_rels
        dedup_old_rels = []
        for rel in old_rels:
            if (rel.type, rel.source, rel.target) not in existing_rels:
                dedup_old_rels.append(rel)
                existing_rels.add((rel.type, rel.source, rel.target)) # Добавляем новый элемент в множество
        
        # Удаляем дубликаты, где type и target совпадают и type совпадает с "LOCATED_IN" или "HAS_ITEM" (сохраняем из new_rels)
        dedup_old_rels = [
            rel for rel in dedup_old_rels 
            if not any(
                (rel.type == new_rel.type and rel.target == new_rel.target and new_rel.type in uniques_types)
                for new_rel in new_rels
            )
        ]
        
        return new_rels + dedup_old_rels  # Возвращаем объединенный список новых и уникальных старых сущностей

    def update(self):
        """
        Updates the dynamic database - removes not linked nodes.
        """

        self.dynamic_database.delete_orphaned_nodes()


    # TOOLS METHODS
    def get_agent_state(self, agent_name: str = None) -> str:
        """
        Returns the current state of the agent, including location and inventory.
        """
        if not agent_name:
            agent_name = self.agent_name
        else:
            agent_name = agent_name.lower()

        # Проверяем существует ли агент в динамическом графе
        if not self.dynamic_database.graph_db.has_node(agent_name):
            return f"AGENT - '{agent_name}' does not exist in the dynamic graph."
        
        locations, inventory = self.dynamic_database.get_agent_state(agent_name)
        if not locations and not inventory:
            return f"AGENT - '{agent_name}' has no known state."
        state_description = f"AGENT - '{agent_name}' current state:\n"
        state_description += f"Current location(s): {', '.join([location.name for location in locations])}\n" if locations else "No known locations.\n"
        if inventory:
            str_inventory = ',\n'.join(list(map( lambda e: f"name: {e.name} - decription: {e.description}", inventory)))
            state_description += f"Inventory:\n{str_inventory}\n"
        
        if locations:
            state_description += f"Additional information about the current location(s):\n{self.describe_entity(locations[0].name)}\n" if locations else ""

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
            item = self.static_database.graph_db.get_node_by_id(item_name)
            self.dynamic_database.upsert_entity(item)
        
        agent_name = agent_name.lower()

        if agent_name is None:
            agent_name = self.agent_name
        
        agent_name = agent_name.lower()
        item_name = item_name.lower()

        # Ensure agent name is in lowercase for consistency
        if not self.dynamic_database.graph_db.has_node(agent_name):
            return f"AGENT - '{agent_name}' does not exist in the dynamic graph."

        # Проверяем, есть ли уже связь между агентом и предметом
        if self.dynamic_database.graph_db.has_relationship(agent_name, item_name, "HAS_ITEM"):
            return f"AGENT - '{agent_name}' already has the item '{item_name}' in their inventory."

        # Проверяем, что предмет не находится в другой локации
        agent_location = self.dynamic_database.get_agent_location(agent_name)
        if not agent_location:
            return f"AGENT - '{agent_name}' has no known location to add the item '{item_name}' to their inventory."
        
        has_r_l = self.static_database.graph_db.has_relationship(item_name, agent_location[0].name, "LOCATED_IN")
        has_r_h = self.static_database.graph_db.has_relationship(agent_location[0].name, item_name, "HAS_ITEM")
        if not (agent_location and  (has_r_l or has_r_h)):
            return f"Item '{item_name}' is currently located in another location and cannot be added to the inventory."

        # Добавляем связь между агентом и предметом
        self.dynamic_database.graph_db.add_relationship(agent_name, item_name, description=f"{agent_name} has {item_name}", relationship_type="HAS_ITEM")

        # Удаляем связь между предметом и локацией, если она существует
        self.dynamic_database.graph_db.delete_relationships_by_type(item_name, "LOCATED_IN")
        self.dynamic_database.graph_db.delete_relationship(agent_location[0].name, item_name, "HAS_ITEM")

        # ДОБАВЛЯЕМ В СПИСОК УДАЛЕННЫХ СВЯЗЕЙ
        if has_r_l:
            self.del_rel_db.add_deleted_relationship(Relationship(source=agent_location[0].name, target=item_name, type="HAS_ITEM"))
        if has_r_h:
            self.del_rel_db.add_deleted_relationship(Relationship(source=item_name, target=agent_location[0].name, type="LOCATED_IN"))

        return f"Item '{item_name}' has been successfully added to AGENT - '{agent_name}' inventory.\nIMPORTANT: Check the description of the item needs to be edited? (you can to use `edit_entity` to do it).\nCurrent item description: {item.description}\n'"
    
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

        # Проверяем в списке удаленных, на наличие связи между предметом и текущим местоположением агента
        has_r_l = self.del_rel_db.has_deleted_relationship(Relationship(source=item_name, target=current_location[0].name, type="LOCATED_IN"))
        has_r_h = self.del_rel_db.has_deleted_relationship(Relationship(source=current_location[0].name, target= item_name, type="HAS_ITEM"))

        # Удаялем связь между предметом и текущим местоположением агента
        if has_r_l:
            # УДАЛЯЕМ ИЗ СПИСКА УДАЛЕННЫХ СВЯЗЕЙ
            self.del_rel_db.remove_deleted_relationship(Relationship(source=item_name, target=current_location[0].name, type="LOCATED_IN"))
        elif has_r_h:
            # УДАЛЯЕМ ИЗ СПИСКА УДАЛЕННЫХ СВЯЗЕЙ
            self.del_rel_db.remove_deleted_relationship(Relationship(source=current_location[0].name, target=item_name, type="HAS_ITEM"))
        else:
            self.dynamic_database.graph_db.add_relationship(current_location[0].name, item_name, f"Located in {current_location[0].name}", "HAS_ITEM")

        # Get item description from static database
        item_description = self.dynamic_database.graph_db.get_node_by_id(item_name).description
        

        return f'AGENT - {agent_name} has been moved the item "{item_name}" from their inventory to their current location: {current_location}.\n WARNING: THE ITEM DESCRIPTION MIGHT NEED EDITING.\nCurrent item description: {item_description}\n'
    
    def describe_entity(self, entity_name: str, NAME_LEAD_LINK = "LEADS_TO") -> str:
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

        relations = self.static_database.graph_db.get_linked_rel_by_type(entity_name)

        actual_relations = self.dynamic_database.graph_db.get_linked_rel_by_type(entity_name)

        results_rels = self._filter_duplicate_rels(relations, actual_relations)

        leads_to_entities = [ r for r in results_rels if r.type == NAME_LEAD_LINK ]

        results_rels = [ r for r in results_rels if r.type != NAME_LEAD_LINK ]

        return str(f"Entity '{main_entity.name}' - {main_entity.description}\n" + \
                "Related entities:\n" + \
                "\n".join([f"{rel.target} - {rel.description}" for rel in results_rels]) if results_rels else "No related entities found.") + \
                str(
                    "\nLeads to entities:\n" + \
                    "\n".join(
                        [f"{lead.target} - {lead.description}" for lead in leads_to_entities])
                ) if leads_to_entities else ""

    
    def edit_entity(self, entity_name: str, new_description: str) -> str:
        """
        Edits the description of the entity.
        If the entity is not found, it returns an error message.
        """
        entity_name = entity_name.lower()
        main_entity = self.dynamic_database.graph_db.get_node_by_id(entity_name)

        if not main_entity:
            # Если сущность не найдена в динамической базе, проверяем в статической базе
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
    
    def move_agent(self, new_location, agent_name=None) -> str:
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

    def add_entity(self, name, description, type, location = None) -> str:
        """
        Adds a new entity to the dynamic graph.
        """
        entity = Entity(name=name.replace("_"," ").lower() , description=remove_tokens(description), type=preprocess_text(type).upper())
        if not isinstance(entity, Entity):
            return "Invalid entity. Please provide an instance of Entity."

        if not location:
            # Если локация не указана, используем текущее местоположение агента
            location = self.dynamic_database.get_agent_location(self.agent_name)
            if not location:
                return "No current location found for the agent. Please specify a location."
            location = location[0].name

        # Проверяем, существует ли сущность в динамической базе

        if self.dynamic_database.graph_db.has_node(entity.name) or self.static_database.graph_db.has_node(entity.name):
            # Если сущность уже существует, Сообщаем об этом и обновляем ее описание
            return f"Entity '{entity.name}' already exists. USE `edit_entity` to update its description."
        
        # Если сущность не существует, добавляем ее
        self.dynamic_database.upsert_entity(entity)

        # Добавляем связь между локацией и entity
        self.dynamic_database.add_relationship(Relationship(entity.name, location, "LOCATED_IN", f'{entity.name} located in {location}'))

        return f"Entity '{entity.name}' added to the dynamic graph."

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

