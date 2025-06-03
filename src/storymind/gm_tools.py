from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.runnables import RunnableConfig


def get_toolpack():
    """
    This function returns a list of tools available in this file.
    """
    return [
        get_agent_state,
        get_agent_location,
        get_agent_inventory,
        get_all_locations,
        move_agent,
        add_item_to_inventory,
        delete_item_from_inventory,
        describe_entity,
        edit_entity,
        search_about,
        add_new_entity_to_game,
        update_user_agent
    ]

@tool
def describe_entity(location_name: str, config: RunnableConfig) -> str:
    """
    Use this:
        * to give a detailed description of a specific entity (location, creature, item).
        * to describe the location when the player wants to look around
        * when the player asks, “Describe the Torch Corridor” or “What is this object?”.
    Insert the result directly into your response, explaining the details.
    """
    gm = config.get("configurable",{}).get('gm', None)
    if not gm:
        return 'Game manager not initialized.'
    
    return gm.describe_entity(location_name)

@tool
def search_about(query: str, config: RunnableConfig) -> str:
    """
    Use this for a “deep” search in the game’s lore - if the player asks a question beyond the immediate location (for example, “Tell me the history of the Dark Paladin”).
    Then present the found information to the player while maintaining the Game Master’s style.
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
    Use it whenever you need to know which location the player (agent) is currently in.
    Based on the result, describe the surroundings and offer possible actions.
    """
    gm = config.get("configurable",{}).get('gm', None)
    if not gm:
        return 'Game manager not initialized.'
    
    agent_name = config.get("agent_name", "player").lower()
    return gm.get_agent_location(agent_name)

@tool
def get_agent_inventory(config: RunnableConfig) -> str:
    """
    Call this when the player asks about their inventory or after obtaining/removing items.
    Structure your response so the player clearly understands what they have in their bag.
    """
    gm = config.get("configurable",{}).get('gm', None)
    if not gm:
        return 'Game manager not initialized.'
    
    agent_name = config.get("agent_name", "player").lower()
    return gm.get_agent_inventory(agent_name)

@tool
def add_item_to_inventory(item_name: str, config: RunnableConfig) -> str:
    """
    Use these when the player picks up an item or spends/gives it away.
    After adding or removing an item, always check the inventory and report the operation’s success.
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
    Use this when the player decides to change location (e.g., go north/south/east/west, etc.).
    After performing the move, immediately use function named "get_agent_location" to confirm the new area and describe it.
    """
    gm = config.get("configurable",{}).get('gm', None)
    if not gm:
        return 'Game manager not initialized.'
    
    agent_name = config.get("agent_name", "player").lower()
    return gm.move_agent(location_name, agent_name)


@tool
def edit_entity(entity_name: str, description: str, config: RunnableConfig) -> str:
    """
    Edits the description of a specific entity in the game.
    Use this when you need to update the description of a location, creature, or item.
    """
    gm = config.get("configurable",{}).get('gm', None)
    if not gm:
        return 'Game manager not initialized.'
    
    return gm.edit_entity(entity_name, description)

@tool
def add_new_entity_to_game(entity_name: str, description: str, type: str, config: RunnableConfig) -> str:
    """
    Adds a new entity to the current player location.

    Use this to introduce a NEW, previously non-existent entity into the game world when the story you are narrating requires it, or when the player's actions plausibly result in the creation of something entirely new. This tool makes the entity exist in the game world.

    This could be:
    - A unique item the player crafts, discovers, or that appears due to an event (e.g., "glowing_shard_from_meteorite", "makeshift_splint", "enchanted_dust_pouch").
    - A new character or creature that appears or is summoned (e.g., "shadow_wolf_conjured", "lost_child_hiding", "ethereal_guardian").
    - A distinct environmental feature, magical effect, or interactable object that manifests or is created (e.g., "rune_etched_on_wall", "temporal_rift_crackling", "barricade_of_debris").

    You MUST provide:
    - `entity_name`: A unique and descriptive name for this new entity (use underscores for spaces, e.g., "ancient_stone_altar", "rickety_rope_bridge"). This name will be used to refer to the entity later.
    - `description`: A detailed narrative description of what the entity is, what it looks like, its current state, and any immediate relevant properties or effects from a story perspective.
    - `type`: The type of entity you are creating, which must be one of the following:
        You are required to use only these types of entities:
        * Location:
            * Spot
            * Building
            * Country
        * Agent:
            * Person
            * Organization
            * Creature
        * Item:
            * Tool
            * Vehicle
            * Document
        * Event:
            * Action
            * Occasion
    IMPORTANT:
    - Use this ONLY for creating genuinely new things that are not part of the pre-defined game world, known item list, or existing characters.
    - Do NOT use this to add a standard, pre-existing item type to the player's inventory (use `add_item_to_inventory` for that, which handles items from a known list).
    - Do NOT use this to change or update an already existing entity (use `edit_entity` for that).
    - After successfully using `add_entity` to create something (e.g., a new item on the ground), if the player then wants to pick it up, you would typically follow up with `add_item_to_inventory("newly_created_item_name")`.
    """
    gm = config.get("configurable",{}).get('gm', None)
    if not gm:
        return 'Game manager not initialized.'
    
    return gm.add_entity(entity_name, description, type)

@tool
def update_user_agent(
    destination: str,
    config: RunnableConfig
) -> str:
    """
    Updates the state of the user agent in the game.
    This tool is used to modify the user agent's state based on the current game context.
    """
    gm = config.get("configurable",{}).get('gm', None)
    if not gm:
        return 'Game manager not initialized.'
    
    return gm.edit_entity(gm.agent_name, destination)