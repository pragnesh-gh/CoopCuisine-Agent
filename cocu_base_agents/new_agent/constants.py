# constants.py

# Kitchen positions
KITCHEN_POSITIONS = [
    # Stove
    {'x': 1, 'y': 0, 'type': 'stove'},

    # Fryer
    {'x': 4, 'y': 0, 'type': 'fryer'},

    # Counters along the top
    *[{'x': x, 'y': 0, 'type': 'counter'} for x in range(0, 15) if x not in (1, 4)],

    # Counters along rows 1-4
    *[{'x': x, 'y': y, 'type': 'counter'} for y in range(1, 5) for x in range(0, 15)],

    # Plate Dispensers
    {'x': 2, 'y': 8, 'type': 'plate_dispenser'},
    {'x': 3, 'y': 8, 'type': 'plate_dispenser'},

    # Sinks
    {'x': 8, 'y': 8, 'type': 'sink'},
    {'x': 9, 'y': 8, 'type': 'sink'},
]

# Task positions for agents to claim
# constants.py

# constants.py
# constants.py


TASK_POSITIONS = {
    "GET_BUN": (10, 0),
    "GET_TOMATO": (7, 0),
    "GET_LETTUCE": (9, 0),
    "GET_MEAT": (11, 1),
    "CUTTING_BOARD_1": (0, 5),
    "PAN": (1, 0),
    "PLATE_DISPENSER": (2, 8),
    "SERVING_WINDOW": (0, 2),
}


