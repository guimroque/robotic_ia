from ultralytics import YOLO
from classes.frame.frame import Frame, IMAGE, MODEL_BLOCKS, MODEL_TABLE
import cv2

# Infer
table_frames = Frame.infer_frames(IMAGE, MODEL_TABLE, 'table')
blocks_frames = Frame.infer_frames(IMAGE, MODEL_BLOCKS, 'blocks')
reason_pixels_mm = Frame.get_proportionality()

Frame.draw_frames(blocks_frames, table_frames[0], True, IMAGE, 'name.png', reason_pixels_mm['average'])
