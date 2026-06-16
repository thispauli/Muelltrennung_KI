import os
import json
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import gradio as gr

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

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activation_map = activations[0]

    for i in range(activation_map.shape[0]):
        activation_map[i, :, :] *= pooled_gradients[i]

    cam_map = torch.sum(activation_map, dim=0)
    cam_map = torch.relu(cam_map)
    cam_map = cam_map.cpu().numpy()

    cam_map -= cam_map.min()
    if cam_map.max() != 0:
        cam_map /= cam_map.max()

    return cam_map

def overlay_heatmap_on_image(pil_image, cam_map, alpha=0.4):
    cam_img = Image.fromarray(np.uint8(cam_map * 255)).resize(pil_image.size, Image.Resampling.BILINEAR)
    cam_array = np.array(cam_img) / 255.0

    heatmap = cm.jet(cam_array)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap)

    original = pil_image.convert("RGB")
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
# Interface Hauptfunktion
# =========================
def run_inference_ui(image_path):
    if image_path is None:
        return None, "<p>Bitte lade ein Bild hoch.</p>"

    try:
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
        
        icon_map = {
            "Rest": "🗑️",
            "Plastik": "🟡",   
            "Papier": "📦",   
            "Bio": "🍎",      
            "Glas": "🍾"      
        }
        icon = icon_map.get(pred_class, "♻️")
        
        result_html = f"""
        <div style="display: flex; align-items: center; justify-content: center; gap: 50px; background: #ffffff; border: 2px solid #e2e8f0; border-radius: 15px; padding: 40px; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1); height: 100%; min-height: 320px; box-sizing: border-box;">
            <div style="font-size: 130px; line-height: 1; filter: drop-shadow(0px 4px 6px rgba(0,0,0,0.2));">
                {icon}
            </div>
            <div style="text-align: left;">
                <div style="font-size: 20px; color: #64748b; text-transform: uppercase; letter-spacing: 2px; font-weight: 600;">Erkannte Kategorie</div>
                <div style="font-size: 64px; font-weight: 800; color: #1e293b; margin-top: -5px; margin-bottom: 15px;">{pred_class}</div>
                <div style="display: inline-block; background: #dcfce7; color: #166534; padding: 10px 20px; border-radius: 30px; font-size: 24px; font-weight: bold;">
                    🎯 {confidence:.2%} Sicherheit
                </div>
            </div>
        </div>
        """
        
        return save_path, result_html
        
    except Exception as e:
        return None, f"<p style='color:red;'>Fehler bei der Bildverarbeitung: {str(e)}</p>"

# =========================
# Gradio UI Start (Mit Accordion)
# =========================
if __name__ == "__main__":
    with gr.Blocks(title="Müll-Klassifizierung", theme=gr.themes.Soft()) as demo:
        gr.Markdown("<h1 style='text-align: center; margin-bottom: 10px;'>♻️ Müll-Klassifizierung mit KI</h1>")
        gr.Markdown("<p style='text-align: center; font-size: 16px; color: #666;'>Lade ein Bild von Abfall hoch. Das KI-Modell analysiert den Müll und ordnet ihn einer Kategorie zu.</p>")
        
        # Obere Reihe: Upload (Links) & Großes Ergebnis (Rechts)
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                input_image = gr.Image(type="filepath", label="Bild hochladen", height=320)
                analyze_btn = gr.Button("🔍 Müll analysieren", variant="primary", size="lg")
                
            with gr.Column(scale=2):
                output_html = gr.HTML(label="KI-Ergebnis")

        # --- NEU: Einklappbarer Bereich für die Heatmap ---
        # open=False sorgt dafür, dass es beim Start eingeklappt ist.
        with gr.Accordion("📊 Detailanalyse (Heatmap) einblenden", open=False):
            gr.Markdown("Hier kannst du sehen, **welche Bildbereiche** für die Entscheidung der KI ausschlaggebend waren (rot = sehr wichtig).")
            # show_label=False macht es noch etwas cleaner, da die Überschrift schon im Accordion steht
            output_image = gr.Image(type="filepath", show_label=False, height=450, interactive=False)

        # Aktion beim Klick auf den Button
        analyze_btn.click(
            fn=run_inference_ui,
            inputs=input_image,
            outputs=[output_image, output_html]
        )

    # Starte die App lokal
    demo.launch(inbrowser=True)