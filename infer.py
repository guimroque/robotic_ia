from ultralytics import YOLO
import torch
from PIL import Image, ImageDraw
from utils import euclidean_distance


MODEL = './treinos/train5/weights/best.pt'
IMAGE = './mocks/test_image/20240216_19_09_34_Pro.jpg'


#inferencia
model = YOLO(MODEL)
results = model(IMAGE)


# Acessa os centros das caixas delimitadoras
xyxy = results[0].boxes.xyxy #todo: make an looping to get boxes of all files
centers = torch.stack([(box[:2] + box[2:4]) / 2 for box in xyxy])

# Calcula as distâncias entre todos os pares de centros
distances = []
for i in range(len(centers)):
    for j in range(i+1, len(centers)):
        dist = euclidean_distance(centers[i], centers[j])
        distances.append((i, j, dist.item()))

# Carrega a imagem original
image = Image.open(IMAGE)
draw = ImageDraw.Draw(image)

# Desenha as distâncias na imagem
for i, j, dist in distances:
    # Ponto médio entre os centros para desenhar o texto
    midpoint = ((centers[i][0] + centers[j][0]) / 2, (centers[i][1] + centers[j][1]) / 2)
    # Desenha uma linha entre os centros
    draw.line((centers[i][0].item(), centers[i][1].item(), centers[j][0].item(), centers[j][1].item()), fill='yellow', width=2)
    # Desenha o texto da distância
    draw.text(midpoint, f"{dist:.2f}", fill='red')

# Salva a imagem com as anotações
image.save('annotated_image.jpg')
image.show()