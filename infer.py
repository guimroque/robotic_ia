import torch
from PIL import Image

# Carrega o modelo pré-treinado personalizado
model = torch.hub.load('ultralytics/yolov5', 'custom', path='caminho/para/seu/modelo.pt')  # Substitua 'caminho/para/seu/modelo.pt' pelo caminho do seu modelo

# Carrega uma imagem para inferência
img = 'caminho/para/sua/imagem.jpg'  # Substitua 'caminho/para/sua/imagem.jpg' pelo caminho da sua imagem

# Realiza a inferência
results = model(img)

# Exibe os resultados
results.show()

# Para salvar ou manipular os resultados, você pode fazer:
results.save()  # Salva a imagem com as detecções
detections = results.xyxy[0]  # Resultados como um tensor PyTorch
print(detections)