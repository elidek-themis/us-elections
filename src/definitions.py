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
        "Democrats",
        "Democratic party",
        "Democratic candidate",
        "Democratic nominee",
        "Republicans",
        "Republican party",
        "Republican candidate",
        "Republican nominee"  
    ]
    
    choices_2 = [
        "Democrats",
        "Democratic party",
        "Republicans",
        "Republican party"
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
            "message": [
                {"role": "system", "content": "Assume the identity of a {state} resident."},
                {"role": "assistant", "content": "In the 2024 US presidential election I will vote for the"}
            ],
            "choices": choices_1
        },
        4: {
            "message": [
                {"role": "system", "content": "Assume the identity of a {state} resident."},
                {"role": "assistant", "content": "In the 2024 US presidential election I will vote for the candidate of the"}
            ],
            "choices": choices_2
        },
        5: {
            "message": [
                {"role": "system", "content": "Assume the identity of a {state} resident."},
                {"role": "user", "content": "What will you vote for in the 2024 US presidential election?"},
                {"role": "assistant", "content": "In the 2024 US presidential election I will vote for the"}
            ],
            "choices": choices_1
        },
        6: {
            "message": [
                {"role": "system", "content": "Assume the identity of a {state} resident."},
                {"role": "user", "content": "What will you vote for in the 2024 US presidential election?"},
                {"role": "assistant", "content": "In the 2024 US presidential election I will vote for the candidate of the"}
            ],
            "choices": choices_2
        },
        7: {
            "message": [
                {"role": "system", "content": (
                    "You are an AI assistant used to create exit polls.\n"
                    "The validitity of your responses is critical.\n"
                    "Assume the identity of a {state} resident."
                    )},
                {"role": "user", "content": "What will you vote for in the 2024 US presidential election?"},
                {"role": "assistant", "content": "In the 2024 US presidential election I will vote for the"}
            ],
            "choices": choices_1
        },
        "7v2": {
            "message": [
                {"role": "system", "content": (
                    "You are a knowledgeable, efficient, and direct AI assistant used to create exit polls.\n"
                    "Provide concise answers, focusing on the key information needed. The validitity of your responses is critical.\n"
                    "Assume the identity of a {state} resident."
                    )},
                {"role": "user", "content": "What will you vote for in the 2024 US presidential election?"},
                {"role": "assistant", "content": "In the 2024 US presidential election I will vote for the"}
            ],
            "choices": choices_1
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

results_2020 = {
    'Alabama': -0.2578361982,
    'Alaska': -0.1050988554,
    'Arizona': 0.003045685279,
    'Arkansas': -0.2839506173,
    'California': 0.2985685072,
    'Colorado': 0.1387461459,
    'Connecticut': 0.2040609137,
    'Delaware': 0.1926977688,
    'District of Columbia': 0.8902439024,
    'Florida': -0.03329969728,
    'Georgia': 0.002024291498,
    'Hawaii': 0.3,
    'Idaho': -0.3175257732,
    'Illinois': 0.1743119266,
    'Indiana': -0.1641182467,
    'Iowa': -0.08350305499,
    'Kansas': -0.1554192229,
    'Kentucky': -0.2634791455,
    'Louisiana': -0.1890243902,
    'Maine': 0.09371781668,
    'Maryland': 0.3401221996,
    'Massachusetts': 0.3421588595,
    'Michigan': 0.02845528455,
    'Minnesota': 0.07346938776,
    'Mississippi': -0.1671732523,
    'Missouri': -0.1568228106,
    'Montana': -0.1671794872,
    'Nebraska': -0.1950970378,
    'Nevada': 0.0245398773,
    'New Hampshire': 0.07520325203,
    'New Jersey': 0.1610942249,
    'New Mexico': 0.1104294479,
    'New York': 0.1582150101,
    'North Carolina': -0.01417004049,
    'North Dakota': -0.3449691992,
    'Ohio': -0.08113590264,
    'Oklahoma': -0.3387922211,
    'Oregon': 0.1659836066,
    'Pennsylvania': 0.01214574899,
    'Rhode Island': 0.2121827411,
    'South Carolina': -0.1187817259,
    'South Dakota': -0.2689938398,
    'Tennessee': -0.2375127421,
    'Texas': -0.05679513185,
    'Utah': -0.2137643379,
    'Vermont': 0.366255144,
    'Virginia': 0.1034482759,
    'Washington': 0.1991786448,
    'West Virginia': -0.3957273652,
    'Wisconsin': 0.007106598985,
    'Wyoming': -0.4500514933
}










