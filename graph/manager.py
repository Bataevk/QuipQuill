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
            self.static_database.close_graph_db()
        if hasattr(self, 'dynamic_database') and self.dynamic_database:
            self.dynamic_database.close_graph_db()
    
    def __del__(self):
        self.close()


    # TOOLS METHODS
    def change_location_for_player(self, location: str):
        """
        This function moves the player to the given location.

        """

        # 1. Поиск у игрока связного узла с типом "LOCATION" в dynamic graph и поиск связных локаций с текущей локацией в статическом графе сохраняем их в переменной link_locations
        # 2. Поиск похожего узла по переменной location в статическом графе через метод search_names
        # 3. Если узел не найден, то возвращаем сообщение об ошибке
        # 4. Если узел найден и он есть в link locations, то добавляем его в динамический граф и создаем связь между игроком и локацией, со связью "located_in", а старую свзязь удаляем
        # 5. Если узел найден и его нет в link locations, то возвращаем сообщение об ошибке


        return 'User has moved to: ШКОЛА'
        
        
        
        link_locations = self.dynamic_database.graph_db.get_linked_nodes("Player", "LOCATION")








        new_locations = self.static_database.search_names(location, n_vector_results=5, threshholder=0.9)
        # new_locations = self.static_database.search_lite

        if not new_locations:
            return f'Location "{location}" not found!'
        
        # if len(new_locations) > 1:
        #     return f'Multiple locations found: {", ".join(new_locations)}. Please specify further.'

        if not self.dynamic_database.graph_db.check_node_exists("Player"):
            self.dynamic_database.add_entity(name="Player", description="Main character")
        if self.dynamic_database.graph_db.check_node_exists(new_locations[0]):
            self.dynamic_database.add_entity(name=new_locations[0], description="Main character")
            

        self.dynamic_database.add_entity(
            name='Player',
            description=f'Игрок переместился в {new_locations[0]}'
        )
        self.dynamic_database.add_relationship(
            subject='Player',
            object=new_locations[0],
            relationship='находится_в'
        )



    def get_player_inventory(self) -> str:
        """
        This function returns the player's inventory.
        """
        return 'Пенал, Печеньки, Ключи, Карта, Карандаш, Линейка, Ластик, Книга о волшебстве'

    def get_location(self):
        """
        This function returns the player's current location.
        """
        return 'Школа'

    def search_object(self, string: str):
        """
        This function searches for the given item in the current location.
        """
        return f'{string} - найден под лестницей !'






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
    manager = Manager(
        load=True
        )

    # # Пример добавления узла (для динамического графа)
    # new_entity = {"name": "Игрок", "description": "Главный герой"}
    # graph_module.add_node(new_entity)

    # # Пример добавления связи (обновление состояния мира)
    # graph_module.add_relationship("Игрок", "Меч", "Имеет")

    # # Пример получения узла и связей
    # node_with_rels = graph_module.get_node_with_relationships("semyon")
    # print(node_with_rels)

