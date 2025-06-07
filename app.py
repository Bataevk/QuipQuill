import os
import sys
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from datetime import date

import logging
# -----------------------------------------------------------------------------------------------------------
# Настройка логирования
# 1) Create and configure your handlers (you already have these):
from logging import FileHandler, StreamHandler, Formatter, getLogger

# INFO handler → writes INFO+ to ./logs/INFO.log
info_handler = FileHandler("./logs/INFO.log", mode="w", encoding="utf-8")
info_handler.setLevel("INFO")
info_formatter = Formatter("%(asctime)s - %(levelname)s - %(message)s")
info_handler.setFormatter(info_formatter)

# DEBUG handler → writes DEBUG+ to ./logs/DEBUG.log
debug_handler = FileHandler("./logs/DEBUG.log", mode="w", encoding="utf-8")
debug_handler.setLevel("DEBUG")
debug_formatter = Formatter("%(asctime)s - %(levelname)s - %(message)s")
debug_handler.setFormatter(debug_formatter)

# (Optional) Console handler so you see logs in the terminal:
console_handler = StreamHandler(sys.stdout)
console_handler.setLevel("INFO")
console_handler.setFormatter(debug_formatter)

# 2) Pick up Flask’s internal loggers:
flask_logger = getLogger("flask.app")     # Flask‐specific messages
werkzeug_logger = getLogger("werkzeug")    # HTTP request logging

# Elevate their log level so they emit INFO/DEBUG
flask_logger.setLevel("INFO")
werkzeug_logger.setLevel("INFO")

# 3) Attach your handlers to them:
for h in (info_handler, debug_handler, console_handler):
    flask_logger.addHandler(h)
    werkzeug_logger.addHandler(h)

# 4) (Optional) If you also want root‐level messages to go to the same files/console,
#    attach them to the root logger too:
root_logger = getLogger()
root_logger.setLevel("DEBUG")
for h in (info_handler, debug_handler, console_handler):
    root_logger.addHandler(h)

# --- Загрузка переменных окружения ---
load_dotenv()

# --- КРИТИЧЕСКИ ВАЖНЫЕ ИМПОРТЫ из вашего проекта ---
# Если эти импорты не удаются, приложение не должно запускаться.
try:
    from src.storymind.agentgraph import *
except ImportError as e:
    logging.critical(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось импортировать основные компоненты из 'src.storymind.agentgraph': {e}")
    sys.exit(1) # Завершение работы, если базовые компоненты недоступны
# --- Конец критических импортов ---

app = Flask(__name__)
app.secret_key = os.urandom(24) # Для управления сессиями Flask (здесь не используется активно)

# Глобальное хранилище состояний игр. Ключ - thread_id.
# Значение: {'gm': game_manager, 'graph': graph, 'run_config': run_config}
# Сообщения теперь не хранятся здесь, а извлекаются из графа.
game_states = {}

# Инициализация конфигурации приложения (загружается один раз)
try:
    app_config_global = load_config("./config.yaml")
except Exception as e:
    logging.critical(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить ./config.yaml: {e}")
    sys.exit(1)


def stream_graph_updates_web(graph, config, user_input: str, role: str = "user"):
    """
    Обрабатывает поток от графа и возвращает последнее сообщение ассистента.
    Предполагается, что граф сам управляет историей сообщений на основе thread_id в config.
    """
    assistant_response_content = "Мастер не ответил." # Ответ по умолчанию
    thread_id = config.get("configurable", {}).get("thread_id", 0)

    messages_for_stream = [{"role": role, "content": user_input}]
    logging.info(f"Отправка в граф для thread_id {thread_id}: {messages_for_stream}")

    try:
        for event in graph.stream(
            {"messages": messages_for_stream},
            config=config,
            stream_mode='values' 
        ):
            last_msg = event["messages"][-1]
            # Структура 'event' зависит от графа LangGraph
            if "messages" in event and event["messages"]:
                if type(last_msg).__name__ == "AIMessage" and last_msg.content != "":
                    assistant_response_content = last_msg.content
                    break # Нашли подходящее сообщение
        logging.info(f"Ответ от графа для thread_id {thread_id}: {assistant_response_content}")
    except Exception as e:
        logging.error(f"Ошибка во время stream_graph_updates_web для thread_id {thread_id}: {e}", exc_info=True)
        assistant_response_content = f"Произошла ошибка при обработке вашего запроса: {e}"

    return assistant_response_content

def save_messages_to_log(thread_id: str, graph_instance, run_config_instance):
    """
    Сохраняет историю сообщений для данного thread_id в лог-файл,
    извлекая сообщения из состояния графа.
    """
    if not graph_instance or not run_config_instance:
        logging.warning(f"Попытка сохранить лог для {thread_id} без графа или конфигурации.")
        return

    try:
        snapshot = graph_instance.get_state(run_config_instance)
        # Структура snapshot.values["messages"] должна соответствовать вашему графу
        messages_to_log = snapshot.values.get("messages", [])
        if not messages_to_log:
            logging.info(f"Нет сообщений для логирования для thread_id: {thread_id}")
            return

        current_date_str = date.today().strftime("%Y-%m-%d")
        log_file_path = f'./logs/MESSAGES_{thread_id}.log'
        os.makedirs('./logs', exist_ok=True)

        with open(log_file_path, 'w', encoding="utf-8") as f:
            for m in messages_to_log:
                if hasattr(m, 'content'):
                    content = m.content
                    role = type(m).__name__ # HumanMessage, AIMessage, SystemMessage, ToolMessage
                    f.write(f"{current_date_str} - {role} - {content}\n")

        logging.info(f"Сообщения для thread_id {thread_id} сохранены в {log_file_path}")

    except Exception as e:
        logging.error(f"Ошибка при сохранении лога для thread_id {thread_id}: {e}", exc_info=True)


@app.route('/')
def index_route():
    return render_template('index.html')

@app.route('/config', methods=['POST'])
def configure_game():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Нет данных"}), 400

    agent_name = data.get('agent_name')
    start_location = data.get('start_location')
    thread_id = data.get('thread_id') # thread_id должен быть строкой для согласованности

    if not all([agent_name, start_location, thread_id]):
        return jsonify({"error": "Отсутствуют обязательные поля: agent_name, start_location, thread_id"}), 400

    # Приводим thread_id к строке, если это не так, для консистентности ключей
    thread_id_str = str(thread_id)

    logging.info(f"Конфигурация для thread_id {thread_id_str}: Agent={agent_name}, Location={start_location}")

    try:
        # Используем импортированный класс/фабрику для создания game_manager
        game_manager_instance.restart()
        game_manager_instance.initalize_agent(agent_name, start_location=start_location)

        run_config_instance = {
            "configurable": {"thread_id": thread_id_str},
            'gm': game_manager_instance, # Передаем экземпляр менеджера игры
            "agent_name": agent_name,
            "generated_mode": app_config_global.get('agentgraph',{}).get("generated_mode", False)
        }

        game_states[thread_id_str] = {
            'run_config': run_config_instance
        }

        # Отправка приветственного сообщения
        init_message_content = 'You need to welcome the player!'
        welcome_message = stream_graph_updates_web(
            graph_instance,
            run_config_instance,
            init_message_content,
            role="system" # Или "user", если ваш граф лучше реагирует так для приветствия
        )

        # Логирование после инициализации (граф должен содержать приветствие)
        save_messages_to_log(thread_id_str, graph_instance, run_config_instance)

        return jsonify({
            "message": "Игра успешно настроена.",
            "thread_id": thread_id_str,
            "initial_message": welcome_message
        })

    except Exception as e:
        logging.error(f"Ошибка при конфигурации игры для thread_id {thread_id_str}: {e}", exc_info=True)
        return jsonify({"error": f"Внутренняя ошибка сервера: {e}"}), 500


@app.route('/chat', methods=['POST'])
def chat_message():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Нет данных"}), 400

    user_input = data.get('message')
    thread_id = data.get('thread_id')

    if not user_input or not thread_id:
        return jsonify({"error": "Отсутствуют 'message' или 'thread_id'"}), 400

    thread_id_str = str(thread_id)
    if thread_id_str not in game_states:
        return jsonify({"error": f"Игровая сессия для thread_id '{thread_id_str}' не найдена. Сначала настройте игру."}), 404

    current_game_state = game_states[thread_id_str]
    run_config_instance = current_game_state['run_config']

    logging.info(f"Сообщение от пользователя для thread_id {thread_id_str}: {user_input}")

    try:
        assistant_response = stream_graph_updates_web(
            graph_instance,
            run_config_instance,
            user_input,
            role="user"
        )

        if hasattr(game_manager_instance, 'update'):
            game_manager_instance.update()

        # Логирование после каждого хода (граф должен содержать все сообщения)
        save_messages_to_log(thread_id_str, graph_instance, run_config_instance)

        return jsonify({"reply": assistant_response})

    except Exception as e:
        logging.error(f"Ошибка в /chat для thread_id {thread_id_str}: {e}", exc_info=True)
        return jsonify({"error": f"Внутренняя ошибка сервера при обработке сообщения: {e}"}), 500

if __name__ == '__main__':
    # Для "продакшена" debug=False и используйте production-grade сервер типа Gunicorn или uWSGI
    # Запуск с debug=True удобен для разработки.
    # host='0.0.0.0' делает сервер доступным извне (например, по IP в локальной сети)
    game_manager_instance: gm = gm(load=app_config_global.get('agentgraph',{}).get("load_entities",False))
    graph_instance = create_graph(app_config=app_config_global)

    app.run(host='0.0.0.0', port=5000, debug=False)