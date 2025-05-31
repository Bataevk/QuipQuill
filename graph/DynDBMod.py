from KnowledgeDBMod import KnowledgeDB
from my_dataclasses import Entity as E
from typing import List
import logging

class DynDBMod(KnowledgeDB):
    """
    This class is a modification of KnowledgeDB that is used for dynamic graph operations.
    It inherits from KnowledgeDB and adds functionality specific to dynamic graphs.
    """

    def __init__(self, *params, db_name="dynamicdb", agent_name="Player"):
        super().__init__(*params, db_name=db_name)
        self.agent_name = agent_name.lower()  # Ensure agent name is stored in lowercase for consistency

        self.add_agent(self.agent_name)  # Ensure the agent node is added upon initialization
    
    def delete_entity(self, name):
        """Delete entity from all databases"""
        name = name.lower()  # Ensure entity name is in lowercase for consistency
        self.vector_db.delete_entity(name)
        self.graph_db.delete_node(name)


    def add_agent(self, agent_name=None):
        """
        Adds a agent node to the dynamic graph if it does not already exist.
        If a agent name is provided, it uses that name; otherwise, it uses the default agent name.
        """
        if agent_name is None:
            agent_name = self.agent_name
        
        if not self.graph_db.has_node(agent_name):
            agent = E(name=agent_name, type="AGENT", description="Default agent node")
            self.graph_db.upsert_entity(agent)  # Ensure agent name is in lowercase for consistency
            return f"AGENT - '{agent_name}' added to the dynamic graph."
        else:
            return f"AGENT - '{agent_name}' already exists in the dynamic graph."
    
    def add_location(self, location_name):
        """
        Adds a location node to the dynamic graph if it does not already exist.
        The location name is converted to lowercase for consistency.
        """
        location_name = location_name.lower()
        if not self.graph_db.has_node(location_name):
            # Создаем узел локации
            location = E(name=location_name, type="LOCATION", description="Default location node")
            self.graph_db.upsert_entity(location)
            return f"Location '{location_name}' added to the dynamic graph."

    def get_agent_location(self, agent_name=None) -> List[E]:
        """
        Returns the current location of the agent.
        If a agent name is provided, it uses that name; otherwise, it uses the default agent name.
        """
        if agent_name is None:
            agent_name = self.agent_name
        
        agent_name = agent_name.lower()  # Ensure agent name is in lowercase for consistency

        if not self.graph_db.has_node(agent_name):
            logging.error(f"AGENT - '{agent_name}' does not exist in the dynamic graph.")
            return []
        
        # Получаем связанные узлы для игрока с отношением 'LOCATED_IN' через метод get_related_nodes
        return self.graph_db.get_linked_nodes_by_type(agent_name, relationship_type="LOCATED_IN")

    

    def get_agent_inventory(self, agent_name=None) -> List[E]:
        """
        Returns the agent's inventory.
        If a agent name is provided, it uses that name; otherwise, it uses the default agent name.
        """
        if agent_name is None:
            agent_name = self.agent_name
        
        agent_name = agent_name.lower()  # Ensure agent name is in lowercase for consistency

        if not self.graph_db.has_node(agent_name):
            return f"agent '{agent_name}' does not exist in the dynamic graph."
        
        # Получаем связанные узлы для игрока с отношением 'HAS_ITEM' через метод get_related_nodes
        return self.graph_db.get_linked_nodes_by_type(agent_name, relationship_type="HAS_ITEM")

    
    def add_item_to_inventory(self, item_name, agent_name=None):
        """
        Adds an item to the agent's inventory.
        If a agent name is provided, it uses that name; otherwise, it uses the default agent name.
        """
        if agent_name is None:
            agent_name = self.agent_name
        
        agent_name = agent_name.lower()
        item_name = item_name.lower()

        # Ensure agent name is in lowercase for consistency
        if not self.graph_db.has_node(agent_name):
            return f"AGENT - '{agent_name}' does not exist in the dynamic graph."
        # Проверяем, существует ли предмет
        if not self.graph_db.has_node(item_name):
            return f"Item '{item_name}' does not exist in the dynamic graph."
        
        # Проверяем, есть ли уже связь между агентом и предметом
        if self.graph_db.has_relationship(agent_name, item_name, "HAS_ITEM"):
            return f"AGENT - '{agent_name}' already has the item '{item_name}' in their inventory."

        # Добавляем связь между агентом и предметом
        self.graph_db.add_relationship(agent_name, item_name, description=f"{agent_name} has {item_name}", relationship_type="HAS_ITEM")

        # Удаляем связь между предметом и локацией, если она существует
        self.graph_db.delete_relationships_by_type(item_name, "LOCATED_IN")

        return f"Item '{item_name}' added to AGENT - '{agent_name}' inventory."



    def get_agent_state(self, agent_name='player') -> tuple[List[E], List[E]]:
        """
        Returns the current state of the agent, including location and inventory.
        """
        agent_name = self.agent_name.lower()
        if not self.graph_db.has_node(agent_name):
            return f"AGENT - '{agent_name}' does not exist in the dynamic graph."
        location = self.get_agent_location(agent_name)
        inventory = self.get_agent_inventory(agent_name)

        return location, inventory

    # Удаляем все узлы без связей
    def delete_orphaned_nodes(self):
        """
        Deletes all nodes in the dynamic graph that do not have any relationships.
        This helps to keep the graph clean and free of orphaned nodes.
        """
        orphaned_nodes = self.graph_db._query("""
            MATCH (n)
            WHERE NOT (n)-[:*]->()
            RETURN n.name AS name
        """)
        if not orphaned_nodes:
            return "No orphaned nodes found in the dynamic graph."
        
        for node in orphaned_nodes:
            self.graph_db.delete_node(node['name'])
            self.vector_db.delete_entity(node['name'])
        
        return f"Deleted {len(orphaned_nodes)} orphaned nodes from the dynamic graph."

    def update(self):
        """
        Updates the dynamic graph by deleting orphaned nodes.
        This method can be called periodically to keep the graph clean.
        """
        return self.delete_orphaned_nodes()