import os
import json
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# =========================
# Pfade
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # src/
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))   # Projektwurzel
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pth")
CLASSES_PATH = os.path.join(MODELS_DIR, "classes.json")

# =========================
# Parameter
# =========================
IMG_SIZE = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Klassen laden
# =========================
with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

num_classes = len(class_names)

# =========================
# Modell laden
# =========================
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# =========================
# Transform
# =========================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# =========================
# Grad-CAM Hook Variablen
# =========================
activations = None
gradients = None

def forward_hook(module, inp, out):
    global activations
    activations = out.detach()

def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0].detach()

# Letzte Convolution-Schicht für Grad-CAM
target_layer = model.layer4[-1].conv2
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# =========================
# Hilfsfunktionen
# =========================
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    orig_size = image.size  # (width, height)
    tensor = transform(image).unsqueeze(0).to(device)
    return image, tensor, orig_size

def predict(image_tensor):
    with torch.enable_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()
    return output, pred_idx, confidence

def generate_gradcam(output, pred_idx):
    model.zero_grad()
    score = output[:, pred_idx]
    score.backward(retain_graph=True)

    # gradients: [1, C, H, W]
    # activations: [1, C, H, W]
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])   # [C]
    activation_map = activations[0]                            # [C, H, W]

    for i in range(activation_map.shape[0]):
        activation_map[i, :, :] *= pooled_gradients[i]

    cam_map = torch.sum(activation_map, dim=0)
    cam_map = torch.relu(cam_map)
    cam_map = cam_map.cpu().numpy()

    # normalisieren
    cam_map -= cam_map.min()
    if cam_map.max() != 0:
        cam_map /= cam_map.max()

    return cam_map

def overlay_heatmap_on_image(pil_image, cam_map, alpha=0.4):
    # CAM auf Originalgröße skalieren
    cam_img = Image.fromarray(np.uint8(cam_map * 255)).resize(pil_image.size, Image.Resampling.BILINEAR)
    cam_array = np.array(cam_img) / 255.0

    # Colormap anwenden
    heatmap = cm.jet(cam_array)[:, :, :3]  # RGB
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap)

    # Originalbild
    original = pil_image.convert("RGB")

    # Overlay
    blended = Image.blend(original, heatmap_pil, alpha=alpha)

    return original, heatmap_pil, blended

def save_visualization(original, heatmap, overlay, image_path, pred_class, confidence):
    file_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(OUTPUT_DIR, f"{file_name}_gradcam.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(heatmap)
    axes[1].set_title("Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay\nKlasse: {pred_class} | Confidence: {confidence:.2%}")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    return save_path

# =========================
# Hauptfunktion
# =========================
def run_inference(image_path):
    if not os.path.exists(image_path):
        print(f"Fehler: Datei nicht gefunden: {image_path}")
        return

    pil_image, image_tensor, _ = preprocess_image(image_path)
    output, pred_idx, confidence = predict(image_tensor)

    pred_class = class_names[pred_idx]
    cam_map = generate_gradcam(output, pred_idx)
    original, heatmap, overlay = overlay_heatmap_on_image(pil_image, cam_map)

    save_path = save_visualization(
        original=original,
        heatmap=heatmap,
        overlay=overlay,
        image_path=image_path,
        pred_class=pred_class,
        confidence=confidence
    )

    print("\n===== Ergebnis =====")
    print(f"Bild:        {image_path}")
    print(f"Kategorie:   {pred_class}")
    print(f"Confidence:  {confidence:.2%}")
    print(f"Visualisierung gespeichert unter:")
    print(save_path)

    print("\nHinweis zur Begründung:")
    print("Die Heatmap zeigt, welche Bildbereiche für die Entscheidung wichtig waren.")
    print("Das ist eine visuelle technische Erklärung, keine natürliche Sprachbegründung.")

# =========================
# Einstieg
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Einzelbild-Klassifikation mit Grad-CAM")
    parser.add_argument("image_path", type=str, help="Pfad zum Bild")
    args = parser.parse_args()

    run_inference(args.image_path)