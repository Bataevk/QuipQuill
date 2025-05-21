from extractor import GraphExtractor
from graph import GraphDB
import os
from knowledgeBase_pre import KnowledgeBaseModule
import logging

class Manager:
    def __init__(self, load = False, static_db_name = "staticdb", dynamic_db_name = "dynamicdb"):
        self.static_database = KnowledgeBaseModule(db_name=static_db_name)
        self.dynamic_database = KnowledgeBaseModule(db_name=dynamic_db_name)

        # Загрузка базы знаний из файла
        if load:
            self.load()
    
    def load(self):
        self.static_database.load()

    def close(self):
        if hasattr(self, 'static_database') and self.static_database:
            self.static_database.close()
        if hasattr(self, 'dynamic_database') and self.dynamic_database:
            self.dynamic_database.close()
    
    def __del__(self):
        self.close()






if __name__ == "__main__":
    # Загрузка переменных окружения из .env файла
    from dotenv import load_dotenv
    load_dotenv()

    # Логирование 
    logging.basicConfig(
        level=logging.DEBUG, 
        format='%(asctime)s - %(levelname)s - %(message)s', 
        filename='logs.log',
        encoding='utf-8'
    )

    # Инициализация менеджера базы знаний
    manager = Manager(load=True)

    # # Пример добавления узла (для динамического графа)
    # new_entity = {"name": "Игрок", "description": "Главный герой"}
    # graph_module.add_node(new_entity)

    # # Пример добавления связи (обновление состояния мира)
    # graph_module.add_relationship("Игрок", "Меч", "Имеет")

    # # Пример получения узла и связей
    # node_with_rels = graph_module.get_node_with_relationships("semyon")
    # print(node_with_rels)

