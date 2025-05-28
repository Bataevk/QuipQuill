from langchain_core.tools import tool
from langchain_core.tools import InjectedToolCallId # Используется для указания параметров, которые не должны передаваться в инструменты
from typing import Annotated
from langchain_core.tools import tool
# from graph.manager import Manager



@tool
def move_to(location: str):
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