# Importing chat models
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI

# from langgraph.prebuilt import create_react_agent
# from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Notice we're using an in-memory checkpointer. This is convenient for our tutorial (it saves it all in-memory). 
# In a production application, you would likely change this to use SqliteSaver or PostgresSaver and connect to your own DB.
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict
from typing import Annotated
import logging
from dotenv import load_dotenv
import os

# from langmem.short_term import SummarizationNode
# from langchain_core.messages.utils import count_tokens_approximately

from langchain_core.messages.utils import (
    trim_messages,
    count_tokens_approximately
)

# -----------------------------------------------------------------------------------------------------------

# Tools
from gm_tools import get_toolpack

# Импортируем менеджер игры
from manager import Manager as gm

# INIT CONFIG .yaml
from utils import load_config


# -----------------------------------------------------------------------------------------------------------
# Создаем папку для логов, если она не существует
if not os.path.exists('logs'):
    os.makedirs('logs')

# -----------------------------------------------------------------------------------------------------------
# Настройка логирования
# Создаем корневой логгер
logger = logging.getLogger()
logger.setLevel(logging.DEBUG) # Устанавливаем минимальный уровень для корневого логгера

# Создаем обработчик для INFO-сообщений
info_handler = logging.FileHandler('./logs/INFO.log', mode='a', encoding='utf-8')
info_handler.setLevel(logging.INFO)
info_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
info_handler.setFormatter(info_formatter)
logger.addHandler(info_handler)

# Создаем обработчик для DEBUG-сообщений
debug_handler = logging.FileHandler('./logs/DEBUG.log', mode='a', encoding='utf-8')
debug_handler.setLevel(logging.DEBUG)
debug_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
debug_handler.setFormatter(debug_formatter)
logger.addHandler(debug_handler)


# -----------------------------------------------------------------------------------------------------------
def get_master_agent(tools, provider: str = "google_genai") -> PromptTemplate:
    """Функция для получения главного агента с инструментами."""
    match provider:
        case "google_genai":
            # Инициализируйте модель Gemini 2.0 Flash через Langchain
            llm = init_chat_model("gemini-2.5-flash-preview-05-20", model_provider="google_genai")
            # llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
            # llm = init_chat_model("gemini-2.0-flash-lite", model_provider="google_genai")

        case "mistral":
            # Инициализируйте модель Mistral через Langchain
            llm = init_chat_model("mistral-large-latest", model_provider="mistralai")
        case _:
            # Инициализируйте модель OpenAI через Langchain, но с другими параметрами baseurl
            llm = ChatOpenAI(
                model=os.getenv("RAG_LLM_MODEL"),
                base_url=os.getenv('RAG_BASE_URL'),
                temperature=0.8, 
                top_p=0.9,
                max_tokens=16384,  # Увеличиваем лимит токенов для больших ответов
                timeout=600,
                openai_api_key=os.getenv('OPENAI_API_KEY')
            )
    llm_with_tools = llm.bind_tools(tools)
    master_agent = PromptTemplate.from_template(system_prompt) | llm_with_tools 
    return master_agent, llm

def create_graph(llm_provider = 'google_genai', max_tokens_trim = 2048) -> StateGraph:
    """Функция для создания графа взаимодействия с игроком."""
    # Каждый node может получать текущее State в качестве входных данных и выводить обновлённое состояние.
    # Обновления для messages будут добавляться к существующему списку, а не перезаписывать его, благодаря 
    # встроенной функции add_messages с синтаксисом Annotated

    # Общее сосотояние для графа
    class State(TypedDict):
        messages: Annotated[list, add_messages]

    # Создаем главного агента
    def chatbot(state: State, config: RunnableConfig) -> Command[State]:
        """Функция для обработки сообщений от игрока."""

        # Trim messages 
        messages = trim_messages(
            state["messages"],
            strategy="last",
            token_counter=count_tokens_approximately,
            max_tokens=max_tokens_trim,  # Устанавливаем лимит на количество токенов
            start_on="human",
            end_on="human",
            include_system = False
        )

        return {
            "messages": [
                llm_agent.invoke(
                    {
                        "messages": messages, 
                        "game_state": config['configurable']['gm'].get_agent_state()
                    }),
                ]
            }
    
    # Инициализация LLM с инструментами
    tools = get_toolpack()  
    llm_agent, _ = get_master_agent(tools=tools, provider=llm_provider) 


    # Создание графа состояний
    graph_builder = StateGraph(State)
        
    # Память 
    memory = InMemorySaver()

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

    # graph_builder.add_node("summarization", summarization_node)
    # graph_builder.add_edge("chatbot", "summarization")

    graph_builder.set_entry_point("chatbot")

    # Хранимим граф в памяти в оперативной памяти
    # return graph_builder.compile(checkpointer=memory)

    # Хранимим граф в памяти на длительный срок
    return graph_builder.compile(store=memory)


def stream_graph_updates(graph, config, user_input: str, role: str = "user"):
    for event in graph.stream(
        {"messages": [{"role": role, "content": user_input}]},
        config=config,
        stream_mode='values'):
        
        event["messages"][-1].pretty_print()
        
        # for value in event.values():
        #     print("Assistant:", value["messages"][-1].content)

# -----------------------------------------------------------------------------------------------------------
# Инициализация конфигурации
app_config = load_config("./config.yaml")
system_prompt = app_config.get("prompts", {}).get("system")
if not system_prompt:
    logging.critical("System prompt not found in config.yaml. Please check your configuration file. Will use default system prompt instead.")
    system_prompt = "You are a Game Master (GM) for a text-based role-playing game. You are forbidden to deviate from the state of the game and you are obliged to navigate only on the information available to you."

# -----------------------------------------------------------------------------------------------------------
# Загрузка переменных окружения из .env файла
load_dotenv()


# -----------------------------------------------------------------------------------------------------------
# Импортируем менеджер игры
game_manager = gm(
    load=False
)

game_manager.restart()  # Перезапускаем менеджер игры

agent_name = "player" 
start_location = "entrance rune hall"

# Инициализация агента в динамическом графе
game_manager.initalize_agent(agent_name, start_location=start_location) 

# Настройка конфигурации для запуска
run_config = {"configurable": {"thread_id": "1"}, 'gm': game_manager, "agent_name": agent_name}

# Создание графа взаимодействия с игроком
graph = create_graph()  # Используем OpenAI в качестве провайдера LLM

# Start chat
print("Welcome to the text-based RPG! Type 'quit' or 'exit' to end the game.")

# Strart message by the system
stream_graph_updates(graph, run_config,'You need to welcome the player!', role="system")

while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break

    stream_graph_updates(graph, run_config, user_input)
    game_manager.update()



snapshot = graph.get_state(run_config)
print("Snapshot:", snapshot)
