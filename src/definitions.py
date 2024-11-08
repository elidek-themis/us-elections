import os.path as osp
from pathlib import Path
from dataclasses import dataclass

ROOT_PATH = Path(__file__).parent.parent
SRC_PATH = osp.join(ROOT_PATH, "src")
DATA_PATH = osp.join(ROOT_PATH, "data")
RESULTS_PATH = osp.join(ROOT_PATH, "results")

@dataclass
class Experiment():
    
    choices_1 = [
        "Democratic party",
        "Democratic candidate",
        "Democratic nominee",
        "Republican party",
        "Republican candidate",
        "Republican nominee"  
    ]
    
    choices_2 = [
        "Democratic party",
        "Republican party"
    ]
    
    choices_3 = [
        "Biden",
        "Trump"
    ]
    
    settings = {
        1: {
            "message": "As a resident of {state}, in the 2024 US presidential election I will vote for the",
            "choices": choices_1
        },
        2: {
            "message": "As a resident of {state}, in the 2024 US presidential election I will vote for the candidate of the",
            "choices": choices_2
        },
        3: {
            "message": "As a resident of {state}, in the 2024 US presidential election I will vote for",
            "choices": choices_3
        }
    }

states = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", 
    "Connecticut", "Delaware", "District of Columbia", "Florida", "Georgia", "Hawaii", 
    "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", 
    "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", 
    "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", 
    "New Hampshire", "New Jersey", "New Mexico", "New York", 
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", 
    "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", 
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", 
    "West Virginia", "Wisconsin", "Wyoming"
]

populations = {
    "Alabama": 5074296,
    "Alaska": 733583,
    "Arizona": 7359197,
    "Arkansas": 3045637,
    "California": 39937489,
    "Colorado": 5812069,
    "Connecticut": 3600218,
    "Delaware": 1021331,
    "District of Columbia": 670050,
    "Florida": 22244823,
    "Georgia": 11019186,
    "Hawaii": 1441319,
    "Idaho": 1966840,
    "Illinois": 12451855,
    "Indiana": 6831197,
    "Iowa": 3217690,
    "Kansas": 2934582,
    "Kentucky": 4512310,
    "Louisiana": 4624349,
    "Maine": 1393915,
    "Maryland": 6166214,
    "Massachusetts": 7122379,
    "Michigan": 10110631,
    "Minnesota": 5717851,
    "Mississippi": 2928554,
    "Missouri": 6181551,
    "Montana": 1286383,
    "Nebraska": 2000177,
    "Nevada": 3276360,
    "New Hampshire": 1402173,
    "New Jersey": 9245866,
    "New Mexico": 2117522,
    "New York": 19453561,
    "North Carolina": 10832061,
    "North Dakota": 780094,
    "Ohio": 11783508,
    "Oklahoma": 4059225,
    "Oregon": 4318492,
    "Pennsylvania": 12972008,
    "Rhode Island": 1093734,
    "South Carolina": 5363327,
    "South Dakota": 918581,
    "Tennessee": 7051339,
    "Texas": 30491287,
    "Utah": 3430344,
    "Vermont": 653385,
    "Virginia": 8884088,
    "Washington": 7897678,
    "West Virginia": 1775156,
    "Wisconsin": 5895908,
    "Wyoming": 582196
}
