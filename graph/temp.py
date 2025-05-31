# сторонние импорты
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI

from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Notice we're using an in-memory checkpointer. This is convenient for our tutorial (it saves it all in-memory). 
# In a production application, you would likely change this to use SqliteSaver or PostgresSaver and connect to your own DB.
from langgraph.checkpoint.memory import MemorySaver # Очень удобно для тестов, но не для продакшена

# Оказывается реально крутая штука
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict
from typing import Annotated

# Для обновления состояния внутри инструмента
from langgraph.types import Command, interrupt

from dotenv import load_dotenv
import os

import logging


# Tools
from gm_tools import get_toolpack

# Импортируем менеджер игры
from manager import Manager as gm

# -----------------------------------------------------------------------------------------------------------

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    filename='game_master.log',
    filemode='w',  # 'w' для перезаписи файла при каждом запуске, 'a' для добавления в конец
    encoding='utf-8'  # Кодировка для поддержки кириллицы
    )

# Загружаем инструменты из gm_tools.py
tools = get_toolpack()

# tools = [ctls.get_location, ctls.get_player_inventory, ctls.move_to, ctls.search_object] # ТЕСТОВЫЕ ИНСТРУМЕНТЫ

# Загрузка переменных окружения из .env файла
load_dotenv()


# Инициализируйте модель Gemini 2.0 Flash через Langchain  
llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

# Инициализируйте модель Mistral через Langchain
# llm = init_chat_model("mistral-large-latest", model_provider="mistralai")

# # Инициализируйте модель OpenAI через Langchain, но с другими параметрами baseurl
# llm = ChatOpenAI(
#     model=os.getenv("RAG_LLM_MODEL"),
#     base_url=os.getenv('RAG_BASE_URL'),
#     temperature=0.8, # Используем .get с дефолтом для необязательных параметров
#     top_p=0.9,
#     max_tokens=2048,
#     timeout=600,
#     openai_api_key=os.getenv('OPENAI_API_KEY')
# )

llm_with_tools = llm.bind_tools(tools)


# Каждый node может получать текущее State в качестве входных данных и выводить обновлённое состояние.
# Обновления для messages будут добавляться к существующему списку, а не перезаписывать его, благодаря 
# встроенной функции add_messages с синтаксисом Annotated

# Общее сосотояние для графа
class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)
    


# Память 

memory = MemorySaver()

#  Посмотреть create react agent https://github.com/langchain-ai/langmem

# Создаём систему с помощью LangGraph, которая будет управлять взаимодействием с игроком

# Поганали мутить узлы!
def chatbot(state: State):
    return {
        "messages": [
            llm_with_tools.invoke( 
                    [
                        {
                            "role": "system",
                            "content": 
                            "You are a Game Master (GM) for a text-based role-playing game. "
                            "You are forbidden to deviate from the state of the game and you are obliged to navigate only on the information available to you"
                            "CURRENT STATE: " + game_manager.get_agent_state(state.get("agent_name", "player"))
                        }
                    ] + state["messages"]
                )
            ]
        }



# Make graph
graph_builder.add_node("chatbot", chatbot)

# Add tools
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")

graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile(checkpointer=memory)



def stream_graph_updates(user_input: str):
    for event in graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config,
        stream_mode='values'):
        
        event["messages"][-1].pretty_print()
        
        # for value in event.values():
        #     print("Assistant:", value["messages"][-1].content)


# Chat config 
game_manager = gm(
    load=False
)  # Инициализация менеджера игры
agent_name = "player"  # Имя агента

game_manager.initalize_agent(agent_name)  # Инициализация агента в динамическом графе

config = {"configurable": {"thread_id": "1"}, 'gm': game_manager, "agent_name": agent_name}

# Start chat
while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break

    stream_graph_updates(user_input)
    game_manager.update()

