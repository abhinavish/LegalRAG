from .entities import *
from .relations import *

ENTITY_TYPES = {
    "LegalCase": LegalCase,
    "Court": Court,
    "LegalConcept": LegalConcept,
    "LegalIssue": LegalIssue,           
    "ProceduralPosture": ProceduralPosture,  
    "StatuteReference": StatuteReference,
}

EDGE_TYPES = {
    "CitesCase": CitesCase,              
    "DecidedBy": DecidedBy,
    "InvolvesConcept": InvolvesConcept,
    "Addresses": Addresses,            
    "HasPosture": HasPosture,            
    "ReferencesStatute": ReferencesStatute,
}

EDGE_TYPE_MAP = {
    ("LegalCase", "LegalCase"): ["CitesCase"],
    ("LegalCase", "Court"): ["DecidedBy"],
    ("LegalCase", "LegalConcept"): ["InvolvesConcept"],
    ("LegalCase", "LegalIssue"): ["Addresses"],
    ("LegalCase", "ProceduralPosture"): ["HasPosture"],
    ("LegalCase", "StatuteReference"): ["ReferencesStatute"],
}