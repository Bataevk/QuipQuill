from langchain_core.tools import tool
from langchain_core.tools import InjectedToolCallId # Используется для указания параметров, которые не должны передаваться в инструменты
from typing import Annotated
from langchain_core.tools import tool

# Define the tools for the agent to use
@tool
def move_to(string: str):
    """
    This function moves the player to the given location.

    """
    return 'User has moved to: ШКОЛА'

@tool
def get_player_inventory() -> str:
    """
    This function returns the player's inventory.
    """
    return 'Пенал, Печеньки, Ключи, Карта, Карандаш, Линейка, Ластик, Книга о волшебстве'

@tool
def get_location():
    """
    This function returns the player's current location.
    """
    return 'Школа'

@tool
def search_object(string: str):
    """
    This function searches for the given item in the current location.
    """
    return f'{string} - найден под лестницей !'