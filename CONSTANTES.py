# enum logs
LOGS = {
    "CAM": "[EVENT_CAMERA]",
    "CONNECT": "[EVENT_CONNECT]",
    "COORD": "[EVENT_COORDS]",
    "OBJECT": "[EVENT_OBJECT]",
    "FRAME": "[EVENT_FRAME]"
}

REFERENCES = {
    #FRAMES
    "B1_S1": 'Centro B1 S1',
    "B1_S2": 'Centro B1 S2',
    "B1_S3": 'Centro B1 S3',
    "B2_S1": 'Centro B2 S1',
    "B2_S2": 'Centro B2 S3',
    "B2_S3": 'Centro B2 S3',
    "B3_S1": 'Centro B3 S1',
    "B3_S2": 'Centro B3 S2',
    "B3_S3": 'Centro B3 S3',
    #OBJECTS
    "CENTER_TABLE": 'Frame_Centro_Mesa_UP',
    "BLOCK_GREY": 'Cubo Cinza',
    "BLOCK_BLACK": 'Cubo Negro',
    "ROBOT": 'Staubli TS60 FL 200',
}

BLOCKS = {
    "WHITE": "WHITE_BLOCK",
    "GREY": "GREY_BLOCK",
    "BLACK": "BLACK_BLOCK"
}

BLOCKS_COLOR = {
    "WHITE": [1, 1, 1, 1],
    "BLACK": [0, 0, 0, 1],
    "GREY": [0.5, 0.5, 0.5, 1]
}


BLOCKS_POSITION = { # this positions base are valid just for frame on center table
    "TABLE": {
        "x_pos": 0,
        "y_pos": 0,
        "z_pos": 60, #todo: validate this, verify with G.
        "x_rot": 0,
        "y_rot": 0,
        "z_rot": 180,
    },
    "REST": {
        "x_pos": 0,
        "y_pos": 0,
        "z_pos": 0, #todo: validate this, verify with G.
        "x_rot": 0,
        "y_rot": 0,
        "z_rot": 180,
    }
}
# robomath.PosePP(
#     100,
#     200,
#     100,
#     0,
#     0,
#     0
# )   