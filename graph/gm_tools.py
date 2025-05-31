from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.runnables import RunnableConfig
from typing import Annotated

def get_toolpack():
    """
    This function returns a list of tools available in this file.
    """
    return [
        get_agent_state,
        get_agent_location,
        get_agent_inventory,
        add_item_to_inventory,
        delete_item_from_inventory,
        move_agent,
        get_all_locations,
        describe_entity,
        search_about
    ]

@tool
def describe_entity(location_name: str, config: RunnableConfig) -> str:
    """
    This function describes a specific entity in the game.
    """
    gm = config.get("configurable",{}).get('gm', None)
    if not gm:
        return 'Game manager not initialized.'
    
    return gm.describe_entity(location_name)

@tool
def search_about(query: str, config: RunnableConfig) -> str:
    """
    This function searches for information about a specific entity in the game.
    """
    gm = config.get("configurable",{}).get('gm', None)
    if not gm:
        return 'Game manager not initialized.'
    
    return gm.search_deep(query)

@tool
def get_all_locations(config: RunnableConfig,) -> str:
    """
    This function returns all locations in the game.
    """
    gm = config.get("configurable",{}).get('gm', None)
    if not gm:
        return 'Game manager not initialized.'
    return gm.get_all_locations()

@tool
def get_agent_state(config: RunnableConfig) -> str:
    """
    Returns the current state of the agent, including location and inventory.
    """
    gm = config.get("configurable",{}).get('gm', None)
    if not gm:
        return 'Game manager not initialized.'
    
    agent_name = config.get("agent_name", "player").lower()
    return gm.get_agent_state(agent_name)

@tool
def get_agent_location(config: RunnableConfig) -> str:
    """
    Returns the current location of the agent.
    """
    gm = config.get("configurable",{}).get('gm', None)
    if not gm:
        return 'Game manager not initialized.'
    
    agent_name = config.get("agent_name", "player").lower()
    return gm.get_agent_location(agent_name)

@tool
def get_agent_inventory(config: RunnableConfig) -> str:
    """
    Returns the current inventory of the agent.
    """
    gm = config.get("configurable",{}).get('gm', None)
    if not gm:
        return 'Game manager not initialized.'
    
    agent_name = config.get("agent_name", "player").lower()
    return gm.get_agent_inventory(agent_name)

@tool
def add_item_to_inventory(item_name: str, config: RunnableConfig) -> str:
    """
    Adds an item to the agent's inventory.
    """

    gm = config.get("configurable",{}).get('gm', None)
    if not gm:
        return 'Game manager not initialized.'
    
    agent_name = config.get("agent_name", "player").lower()
    return gm.add_item_to_inventory(item_name, agent_name)


@tool
def delete_item_from_inventory(item_name: str, config: RunnableConfig) -> str:
    """
    Deletes an item from the agent's inventory.
    """
    gm = config.get("configurable",{}).get('gm', None)
    if not gm:
        return 'Game manager not initialized.'
    
    agent_name = config.get("agent_name", "player").lower()
    return gm.move_item_from_inventory(item_name, agent_name)


@tool
def move_agent(location_name: str, config: RunnableConfig) -> str:
    """
    Moves the agent to a specified location.
    """
    gm = config.get("configurable",{}).get('gm', None)
    if not gm:
        return 'Game manager not initialized.'
    
    agent_name = config.get("agent_name", "player").lower()
    return gm.move_agent(location_name, agent_name)
