from quipquill_builder.models import Entity, Graph, Relationship
from quipquill_builder.story import build_story, slugify, story_type


def edges(graph: Graph) -> set[tuple[str, str, str]]:
    return {(r.src, r.type, r.dst) for r in build_story(graph).relations}


def test_types_and_ids(graph: Graph) -> None:
    story = build_story(graph)
    by_name = {e.name: e for e in story.entities}

    assert by_name["The Entrance Rune Hall"].id == "loc:entrance-rune-hall"
    assert by_name["copper torch"].id == "item:copper-torch"
    assert by_name["Skeleton Guardian"].id == "npc:skeleton-guardian"
    assert by_name["Skeleton Guardian"].type == "CREATURE"
    assert by_name["Skeleton Guardian"].subtype == "Creature"
    # Untyped, but the end of a path → a location.
    assert by_name["Whispering Woods"].type == "LOCATION"
    assert by_name["Whispering Woods"].subtype is None


def test_relations_are_normalized(graph: Graph) -> None:
    result = edges(graph)

    # Location HAS_ITEM / HAS_AGENT → LOCATED_IN.
    assert ("item:copper-torch", "LOCATED_IN", "loc:entrance-rune-hall") in result
    assert ("npc:skeleton-guardian", "LOCATED_IN", "loc:torchlight-corridor") in result
    assert not any(t in ("HAS_AGENT",) for _, t, _ in result)
    # An NPC's HAS_ITEM is inventory and stays.
    assert ("npc:skeleton-guardian", "HAS_ITEM", "item:rusty-key") in result


def test_paths_get_reverse_edges_unless_one_way(graph: Graph) -> None:
    result = edges(graph)

    assert ("loc:torchlight-corridor", "LEADS_TO", "loc:entrance-rune-hall") in result
    assert ("loc:whispering-woods", "LEADS_TO", "loc:torchlight-corridor") in result
    assert ("loc:torchlight-corridor", "LEADS_TO", "loc:collapsed-shaft") in result
    assert ("loc:collapsed-shaft", "LEADS_TO", "loc:torchlight-corridor") not in result


def test_reverse_edge_keeps_original_description() -> None:
    graph = Graph(
        entities=[Entity("A", "Spot"), Entity("B", "Spot")],
        relationships=[
            Relationship("A", "B", "LEADS_TO", "North"),
            Relationship("B", "A", "LEADS_TO", "South"),
        ],
    )
    descriptions = {(r.src, r.dst): r.description for r in build_story(graph).relations}
    assert descriptions == {("loc:a", "loc:b"): "North", ("loc:b", "loc:a"): "South"}


def test_id_collisions_get_suffixes() -> None:
    graph = Graph(entities=[Entity("Old Key", "Tool"), Entity("old-key", "Tool")])
    assert [e.id for e in build_story(graph).entities] == ["item:old-key", "item:old-key-2"]


def test_slugify_and_type_mapping() -> None:
    assert slugify("Café de l'Ombre") == "cafe-de-lombre"
    assert slugify("!!!") == "entity"
    assert story_type("Person") == "NPC"
    assert story_type("weird") == "OTHER"
