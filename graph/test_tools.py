import os
import sys
import pytest

# Вставляем в sys.path путь к текущей папке, чтобы "import manager" брал именно graph/manager.py
current_dir = os.path.dirname(__file__)
sys.path.insert(0, current_dir)

from manager import Manager as Gm  # теперь гарантированно импортируем graph/manager.py

@pytest.fixture(scope="module")
def gm():
    gm = Gm()

    # Restart
    gm.restart()

    # Инициализация ИГРОКА test_player
    gm.initalize_agent("test_player","entrance rune hall")
    
    assert gm.dynamic_database.graph_db.has_node("test_player"), "Agent 'test_player' should be initialized in the graph."
    assert gm.agent_name == "test_player", "Agent name should be 'test_player'."
    yield gm
    # Метод close() в Manager есть, поэтому можно оставить
    gm.close()

    gm.restart()  # Перезапускаем менеджер игры после тестов


def test_get_all_locations(gm: Gm):
    result = gm.get_all_locations()
    assert isinstance(result, str)
    assert ("Available locations:" in result) or ("No locations found" in result)

def test_get_agent_state(gm: Gm):
    result = gm.get_agent_state("test_player")
    assert isinstance(result, str)
    assert "AGENT - 'test_player'" in result

def test_add_item_to_inventory(gm: Gm):
    result = gm.add_item_to_inventory("copper torch", "test_player")
    assert isinstance(result, str)
    assert "copper torch" in result

def test_get_agent_inventory(gm: Gm):
    result = gm.get_agent_inventory("test_player")
    assert isinstance(result, str)
    assert "copper torch" in result

def test_move_agent_without_current_location(gm: Gm):
    result = gm.move_agent("flood hall", "test_player")
    assert isinstance(result, str)
    assert "moved to flood hall" in result

def move_item_from_inventory(gm: Gm):
    result = gm.move_item_from_inventory("copper torch", "test_player")
    assert isinstance(result, str)
    assert "has moved" in result

def test_search_deep(gm: Gm):
    result = gm.search_deep("test_player", n_names=1, tresholder=0.5)
    assert isinstance(result, str)
    assert len(result) > 0

def test_add_entity(gm: Gm) -> str:
    # This function adds a new entity to the game.
    result = gm.add_entity('test_entity', 'description of test entity','ITEM', 'location_name')

    assert isinstance(result, str)
    assert "added to the dynamic graph" in result, "The entity should be added successfully."

def test_update_user(gm: Gm) -> str:
    # This function updates the user in the game, such as moving them to a new location.
    result = gm.edit_entity(gm.agent_name, 'NEW AGENT DESCRIPTION')
    assert isinstance(result, str)
    assert "has been updated with new description" in result, "The agent should be updated successfully."

def test_delete_node(gm: Gm):
    # Удаляем узел, который точно есть в графе
    gm.dynamic_database.delete_entity("test_player")
    
    # Проверяем, что узел удален
    result = gm.dynamic_database.graph_db.get_node_by_id("test_player")
    assert result is None, "Node 'test_player' should be deleted from the graph."

def test_delete_orphaned_nodes(gm: Gm):
    # Удаляем все узлы без связей
    result = gm.dynamic_database.delete_orphaned_nodes()
    
    # Проверяем, что результат содержит информацию об удалении
    assert isinstance(result, str)
    assert "Deleted" in result
