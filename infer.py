from classes.Frame.frame import Frame, IMAGE, MODEL_BLOCKS, MODEL_TABLE
from CONSTANTES import LOGS
from classes.Objects.object import Objects


# Infer // todo: fix this infers, move constantes
table_frames = Frame.infer_frames(IMAGE, MODEL_TABLE, 'table')
blocks_frames = Frame.infer_frames(IMAGE, MODEL_BLOCKS, 'blocks')
reason_pixels_mm = Frame.get_proportionality()

Frame.draw_frames(blocks_frames, table_frames[0], True, IMAGE, 'name.png', reason_pixels_mm['average'])

#plot items on robodk
for frame in blocks_frames:
    coords = Frame.get_coords(table_frames[0], frame, reason_pixels_mm['average'])
    print(f"{LOGS['CONNECT']} {coords}")
    Objects().insert(coords=[coords['x_mm'], coords['y_mm']])

