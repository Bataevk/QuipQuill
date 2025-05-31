from typing import List
from dataclasses import dataclass, field

# --- Датаклассы ---
@dataclass
class Entity:
    name: str
    type: str = "UNKNOWN"
    description: str = ""

@dataclass
class Relationship:
    source: str
    target: str
    type: str = "UNKNOWN"
    description: str = ""