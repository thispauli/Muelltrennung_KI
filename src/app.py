import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

# Klassen (muss gleiche Reihenfolge wie Training haben)
classes = ["Bio", "Glas", "Papier", "Plastik", "Rest", "Sonder"]

# Modell laden
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)

    return classes[predicted.item()]

# Test
image_path = "test.jpg"  # dein Testbild
result = predict(image_path)

print("Vorhersage:", result)