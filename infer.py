from ultralytics import YOLO
from PIL import Image
from classes.frame.frame import Frame
import math
import cv2

MODEL = './treinos/train5/weights/best.pt'
IMAGE = './mocks/test_image/20240216_19_05_23_Pro.jpg'

IMAGE_RESOLUTION = (1920, 1080)

# Inferência
model = YOLO(MODEL)
results = model(IMAGE)

# Acessa as coordenadas das caixas delimitadoras
xyxy = results[0].boxes.xyxy

# Carrega a imagem
image = cv2.imread(IMAGE)
frames = []

# Desenha os boxes na imagem
for box in xyxy:
    frames.append(Frame.yolov8_infer(box))

Frame.draw_frames(frames, Frame.full(IMAGE_RESOLUTION, "FULL_FRAME"), True, IMAGE)
