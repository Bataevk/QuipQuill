from extractor import GraphExtractor
from graph import GraphDB
import os







if __name__ == "__main__":
    # Загрузка переменных окружения из .env файла
    from dotenv import load_dotenv
    load_dotenv()

    # NEO4J_URI
    # NEO4J_USERNAME
    # NEO4J_PASSWORD
    # Проверка наличия необходимых переменных окружения
    required_env_vars = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]
    for var in required_env_vars:
        if var not in os.environ:
            raise ValueError(f"Отсутствует переменная окружения: {var}")

    # Инициализация GraphDB
    graph_module = GraphDB(uri=os.getenv("NEO4J_URI"), user=os.getenv("NEO4J_USERNAME"), password=os.getenv("NEO4J_PASSWORD"))
    if not graph_module.ping():  # Проверка соединения с Neo4j
        print("Не удалось подключиться к Neo4j. Проверьте параметры подключения.")
        exit(1)

    
    # Инициализация GraphExtractor
    extractor = GraphExtractor(config_path="./config.yaml")
    graph_data = extractor.update()  # Получение данных в формате JSON

    # Загрузка данных в Neo4j (для статического графа)
    graph_module.load_from_json(graph_data)

    # # Пример добавления узла (для динамического графа)
    # new_entity = {"name": "Игрок", "description": "Главный герой"}
    # graph_module.add_node(new_entity)

    # # Пример добавления связи (обновление состояния мира)
    # graph_module.add_relationship("Игрок", "Меч", "Имеет")

    # Пример получения узла и связей
    node_with_rels = graph_module.get_node_with_relationships("Игрок")
    print(node_with_rels)

    # Закрытие соединения
    graph_module.close()
