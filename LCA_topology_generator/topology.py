# LCA_topology_generator/topology.py

LCA_TOPOLOGY = {
    "LMCA": {
        "id": 0,
        "parent": None,
        "children": ["LAD", "LCX"],
        "status": "mandatory",
        "description": "Left main coronary artery root trunk",
    },
    "LAD": {
        "id": 1,
        "parent": "LMCA",
        "children": ["D1", "D2", "D3_optional", "Septal_optional"],
        "status": "mandatory",
        "description": "Left anterior descending artery",
    },
    "LCX": {
        "id": 2,
        "parent": "LMCA",
        "children": ["OM1", "OM2", "OM3_optional", "Left_PLV_optional", "Left_PDA_optional"],
        "status": "mandatory",
        "description": "Left circumflex artery",
    },
    "RI": {
        "id": 3,
        "parent": "LMCA",
        "children": [],
        "status": "future_optional",
        "description": "Ramus intermedius for trifurcation pattern",
    },
}


def get_active_priority1_branches():
    """
    First active LCA structure.
    Later we will add RI, D branches, OM branches.
    """
    return ["LMCA", "LAD", "LCX"]