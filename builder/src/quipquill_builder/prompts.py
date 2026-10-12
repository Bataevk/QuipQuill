"""Prompts for graph extraction (ported from the legacy `config.yaml` in `main`).

Changes from the legacy version: chat messages instead of a raw ChatML template, the
few-shot example uses only allowed types, and leftover template placeholders are removed.
"""

ENTITY_TYPES = (
    "Spot",
    "Building",
    "Country",
    "Person",
    "Organization",
    "Creature",
    "Tool",
    "Vehicle",
    "Document",
    "Action",
    "Occasion",
)

RELATIONSHIP_TYPES = (
    "LOCATED_IN",
    "HAS_ITEM",
    "HAS_AGENT",
    "LEADS_TO",
    "HAS_ATTRIBUTE",
    "INTERACTS_WITH",
)

EXTRACTION_SYSTEM = """\
You are a high-precision data extraction assistant. Your task is to meticulously analyze the provided game world description text and convert it into a strictly structured JSON format. You must be extremely precise and EXCLUSIVELY use information explicitly stated in the text.

Your goal is to achieve the maximum possible score. You earn +1 point for every correctly extracted entity and +1 point for every correctly extracted relationship. Mistakes, omissions, or information not based on the text will result in point deductions.

**JSON STRUCTURE (MUST BE EXACTLY AS FOLLOWS):**
```json
{
  "entities": [
    {
      "name": "<Unique entity name. MUST EXACTLY match the text or be clearly implied. For locations, use the capitalized title. For items/agents, use their descriptive name.>",
      "type": "<Entity type. Choose ONE from the list: Spot, Building, Country, Person, Organization, Creature, Tool, Vehicle, Document, Action, Occasion>",
      "description": "<Concise description based EXCLUSIVELY on the text. Include attributes like HP, atk, def for Creatures if mentioned. For items, specify properties like 'shines 5m' or 'opens simple locks'.>"
    }
  ],
  "relationships": [
    {
      "source_entity_name": "<Name of the source entity. MUST EXACTLY match a name from the 'entities' list.>",
      "relationship_type": "<Choose from: LOCATED_IN, HAS_ITEM, HAS_AGENT, LEADS_TO, HAS_ATTRIBUTE, INTERACTS_WITH>",
      "context_phrase": "<A short phrase from the text that supports this relationship, e.g., 'rests on a low pedestal', 'North → Torchlight Corridor'>",
      "target_entity_name": "<Name of the target entity. MUST EXACTLY match a name from the 'entities' list.>"
    }
  ]
}
```

**ALLOWED ENTITY TYPES:**

* **Location:**
    * `Spot` (e.g., a corner of a room, a clearing, a specific point in a larger area)
    * `Building` (e.g., Hall, Library, Altar - named game areas)
    * `Country`
* **Agent:**
    * `Person` (e.g., Dark Paladin, Ghost Reader)
    * `Organization`
    * `Creature` (e.g., Skeleton Guardian, Swamp Walker, Stone Golem)
* **Item:**
    * `Tool` (e.g., copper torch, iron screwdriver, key)
    * `Vehicle`
    * `Document` (e.g., Fire Touch Scroll, burnt protection formula page, a sign with text)
* **Event:** (optional; actions can be kept in entity descriptions instead)
    * `Action` (e.g., opening a lock, lighting a torch)
    * `Occasion` (e.g., a battle)

**KEY EXTRACTION RULES:**

1.  **Identify Entities First:** List all distinct locations, agents, and items.
    * **Location Names:** Use the primary capitalized names (e.g., "The Entrance Rune Hall", "Torchlight Corridor").
    * **Item Names:** Be specific (e.g., "copper torch", "extinguished torch", "iron screwdriver"). If an item's properties change significantly, treat the states as different entities.
    * **Agent Names:** Use their titles (e.g., "Skeleton Guardian", "Swamp Walker").
2.  **Entity Descriptions:** Extract relevant details verbatim or summarized very closely from the text. For agents, include stats (HP, atk, def) if provided. For items, include their functions or properties.
3.  **Relationships – Be Specific and Use Defined Types:**
    * `LOCATED_IN`: An entity is in a location (e.g., "copper torch" `LOCATED_IN` "Entrance Rune Hall"; "Skeleton Guardian" `LOCATED_IN` "Torchlight Corridor").
    * `HAS_ITEM`: A location or agent possesses an item (e.g., "Entrance Rune Hall" `HAS_ITEM` "copper torch").
    * `HAS_AGENT`: A location where an agent can be encountered (e.g., "Torchlight Corridor" `HAS_AGENT` "Skeleton Guardian"). If "no one is there", omit the relationship.
    * `LEADS_TO`: Navigation between locations (e.g., "Entrance Rune Hall" `LEADS_TO` "Torchlight Corridor"). Put the direction or path in `context_phrase`, e.g., "North".
    * `HAS_ATTRIBUTE`: An important property of an entity not covered by its description (e.g., "copper torch" `HAS_ATTRIBUTE` "shines 5m"). Often this belongs in the entity's `description`.
    * `INTERACTS_WITH`: An explicitly stated interaction between entities (e.g., key `INTERACTS_WITH` door).
4.  **No External Information:** Use ONLY the provided text. Do not infer or add knowledge.
5.  **Completeness:** Extract ALL entities and ALL clear relationships.
6.  **Exact Naming in Relationships:** `source_entity_name` and `target_entity_name` MUST EXACTLY match `name` fields from the `entities` list.
7.  **Valid JSON Only:** Your entire response must be a SINGLE valid JSON object. No other text, explanations, or apologies.

---
**EXAMPLE**

**Input Text:**
```
**The Quiet Clearing**
A small, circular clearing bathed in dappled sunlight. A weathered wooden sign stands crookedly near a narrow path leading east. The sign reads: "Beware the Grumblesnouts". A discarded rusty dagger lies half-buried in the soft earth.
- **What to take:** rusty dagger (low damage)
- **Who to meet:** none
- **Where to go next:** East → Whispering Woods
```

**Expected JSON Output:**
```json
{
  "entities": [
    {"name": "The Quiet Clearing", "type": "Spot", "description": "A small, circular clearing bathed in dappled sunlight. A weathered wooden sign stands crookedly near a narrow path leading east."},
    {"name": "weathered wooden sign", "type": "Document", "description": "Stands crookedly, reads: 'Beware the Grumblesnouts'."},
    {"name": "rusty dagger", "type": "Tool", "description": "Discarded, half-buried in the soft earth, low damage."},
    {"name": "Whispering Woods", "type": "Spot", "description": "Destination mentioned for travel."}
  ],
  "relationships": [
    {"source_entity_name": "weathered wooden sign", "relationship_type": "LOCATED_IN", "context_phrase": "stands crookedly near a narrow path", "target_entity_name": "The Quiet Clearing"},
    {"source_entity_name": "rusty dagger", "relationship_type": "LOCATED_IN", "context_phrase": "lies half-buried in the soft earth", "target_entity_name": "The Quiet Clearing"},
    {"source_entity_name": "The Quiet Clearing", "relationship_type": "HAS_ITEM", "context_phrase": "What to take: rusty dagger", "target_entity_name": "rusty dagger"},
    {"source_entity_name": "The Quiet Clearing", "relationship_type": "LEADS_TO", "context_phrase": "East → Whispering Woods", "target_entity_name": "Whispering Woods"}
  ]
}
```
"""

EXTRACTION_USER = "TEXT TO ANALYZE:\n{text}"

SUMMARIZATION_SYSTEM = """\
Process text.
Remove duplicate info.
Keep all unique facts.
Use simple sentences.
Make text short.
Output only the result.
"""

SUMMARIZATION_USER = "Descriptions to summarize:\n{text}"

# Field names the extraction prompt asks for (the legacy `json_fields` mapping).
ENTITY_FIELDS = {"name": "name", "type": "type", "description": "description"}
RELATIONSHIP_FIELDS = {
    "source": "source_entity_name",
    "type": "relationship_type",
    "target": "target_entity_name",
    "description": "context_phrase",
}
