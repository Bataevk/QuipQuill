from src.storymind.agentgraph import *

def stream_graph_updates(graph, config, user_input: str, role: str = "user"):
    for event in graph.stream(
        {"messages": [{"role": role, "content": user_input}]},
        config=config,
        stream_mode='values'):
        event["messages"][-1].pretty_print()

        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

# Инициализация конфигурации
app_config = load_config("./config.yaml")
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
run_config = {"configurable": {"thread_id": "1"}, 'gm': game_manager, "agent_name": agent_name, "generated_mode": True}

# Создание графа взаимодействия с игроком
graph = create_graph(
    app_config=app_config, 
)

# Start chat
print("Welcome to the text-based RPG! Type 'quit', 'q' or 'exit' to end the game.")

# Start message

init_message = {'role': 'system', 'content': 'You need to welcome the player!'}

stream_graph_updates(graph, run_config, init_message['content'], role=init_message['role'])

# Игровой цикл
while True:
    # Получаем запрос пользователя 
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break

    stream_graph_updates(graph, run_config, user_input)
    game_manager.update()



snapshot = graph.get_state(run_config)

# СОХРАНЯЕМ СООБЩЕНИЯ
from datetime import date

current_date = date.today()
# Save messages to log
with open('./logs/MESSAGES.log', 'w', encoding="utf-8") as f:
    for m in snapshot.values["messages"]:
        f.write(f"{current_date} - {type(m).__name__} - {m.content}\n")