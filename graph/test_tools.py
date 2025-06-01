import os
import sys
import pytest
import logging

# Вставляем в sys.path путь к текущей папке, чтобы "import manager" брал именно graph/manager.py
current_dir = os.path.dirname(__file__)
sys.path.insert(0, current_dir)

from manager import Manager as Gm  # теперь гарантированно импортируем graph/manager.py


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='test_game_manager.log',
    filemode='w',
    encoding='utf-8'
)


@pytest.fixture(scope="module")
def gm():
    gm = Gm()
    logging.info("Initializing Game Manager")
    # Инициализация ИГРОКА test_player
    gm.initalize_agent("test_player")
    yield gm
    # Метод close() в Manager есть, поэтому можно оставить
    gm.close()


def test_get_all_locations(gm: Gm):
    result = gm.get_all_locations()
    logging.info(f"get_all_locations result: {result}")
    assert isinstance(result, str)
    assert ("Available locations:" in result) or ("No locations found" in result)


def test_get_agent_state(gm: Gm):
    result = gm.get_agent_state("test_player")
    logging.info(f"get_agent_state result: {result}")
    assert isinstance(result, str)
    assert "AGENT - 'test_player'" in result

def test_add_item_to_inventory(gm: Gm):
    result = gm.add_item_to_inventory("copper torch", "test_player")
    logging.info(f"add_item_to_inventory result: {result}")
    assert isinstance(result, str)
    assert "copper torch" in result

def test_get_agent_inventory(gm: Gm):
    result = gm.get_agent_inventory("test_player")
    logging.info(f"get_agent_inventory result: {result}")
    assert isinstance(result, str)
    assert "copper torch" in result



def test_move_agent_without_current_location(gm: Gm):
    result = gm.move_agent("flood hall", "test_player")
    logging.info(f"move_agent result: {result}")
    assert isinstance(result, str)
    assert "moved to flood hall" in result

def move_item_from_inventory(gm: Gm):
    result = gm.move_item_from_inventory("copper torch", "test_player")
    logging.info(f"move_item_from_inventory result: {result}")
    assert isinstance(result, str)
    assert "has moved" in result

def test_search_deep(gm: Gm):
    result = gm.search_deep("test_player", n_names=1, tresholder=0.5)
    logging.info(f"search_deep result: {result}")
    assert isinstance(result, str)
    assert len(result) > 0

def test_delete_node(gm: Gm):
    # Удаляем узел, который точно есть в графе
    gm.dynamic_database.delete_entity("test_player")
    
    # Проверяем, что узел удален
    result = gm.dynamic_database.graph_db.get_node_by_id("test_player")
    logging.info(f"delete_node result: {result}")
    assert result is None, "Node 'test_player' should be deleted from the graph."

def test_delete_orphaned_nodes(gm: Gm):
    # Удаляем все узлы без связей
    result = gm.dynamic_database.delete_orphaned_nodes()
    
    # Проверяем, что результат содержит информацию об удалении
    logging.info(f"delete_orphaned_nodes result: {result}")
    assert isinstance(result, str)
    assert "Deleted" in result
