from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.runnables import RunnableConfig
from typing import Annotated

def get_toolpack():
    """
    This function returns a list of tools available in this file.
    """
    return [get_all_locations]

@tool
def get_all_locations(config: RunnableConfig,) -> str:
    """
    This function returns all locations in the game.
    """
    gm = config.get("configurable",{}).get('gm', None)
    if not gm:
        return 'Game manager not initialized.'
    return gm.get_all_locations()