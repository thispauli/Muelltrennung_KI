import os
import json
import numpy as np
from PIL import Image
import time
import shutil

import torch
import torch.nn as nn
from torchvision import transforms, models
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import gradio as gr

# =====================================================================
# Pfade & Konfiguration
# =====================================================================
# Definiere absolute Pfade, damit das Skript von überall ausgeführt werden kann,
# ohne dass es zu Pfad-Problemen kommt.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # src/ (Verzeichnis dieses Skripts)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))   # Eine Ebene höher (Projektwurzel)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")              # Ordner für Modellgewichte & Klassen
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")             # Ordner für generierte Grad-CAM Bilder

# Pfad zum dedizierten Training-Ordner für Korrekturen aus dem Admin-Bereich
TRAINING_DIR = os.path.join(PROJECT_ROOT, "Training")                  

# Erstelle die Ausgabe- und Trainingsordner, falls sie noch nicht existieren
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TRAINING_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pth")
CLASSES_PATH = os.path.join(MODELS_DIR, "classes.json")

# Einfaches Passwort für den Admin-Bereich (in Produktion besser Hashes oder Umgebungsvariablen nutzen)
ADMIN_PASSWORD = "admin"

# =====================================================================
# Modell- und Hardware-Parameter
# =====================================================================
IMG_SIZE = 224 # Standard-Eingabegröße für ResNet
# Automatische Erkennung, ob eine Grafikkarte (GPU) verfügbar ist, ansonsten CPU nutzen
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================================
# Klassen laden
# =====================================================================
# Lädt das Mapping der Klassen-IDs zu den sprechenden Namen (z.B. 0: "Plastik")
with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

num_classes = len(class_names)

# =====================================================================
# Modell initialisieren und laden
# =====================================================================
# Initialisiere die ResNet18-Architektur (ohne vortrainierte ImageNet-Gewichte, da wir eigene laden)
model = models.resnet18(weights=None)
# Passe die letzte vollvernetzte (Fully Connected) Schicht an unsere Anzahl an Klassen an
model.fc = nn.Linear(model.fc.in_features, num_classes)
# Lade die trainierten Gewichte in das Modell (map_location stellt sicher, dass es auch auf CPU läuft)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval() # Setze das Modell in den Evaluierungsmodus (deaktiviert Dropout, Batch-Norm etc.)

# =====================================================================
# Bild-Transformationen (Preprocessing)
# =====================================================================
# Definiere, wie hochgeladene Bilder verarbeitet werden sollen, bevor sie ins Modell gehen
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), # Skaliere auf 224x224
    transforms.ToTensor(),                   # Konvertiere PIL-Bild in PyTorch-Tensor (Werte von 0 bis 1)
])

# =====================================================================
# Grad-CAM Setup (Hooks)
# =====================================================================
# Globale Variablen, um die Feature-Maps (Activations) und die Gradienten zwischenzuspeichern
activations = None
gradients = None

# Forward-Hook: Wird während des Vorwärtsdurchlaufs (Inferenz) aufgerufen
def forward_hook(module, inp, out):
    global activations
    activations = out.detach() # Speichert die Ausgabe (Feature Maps) der Schicht

# Backward-Hook: Wird während des Rückwärtsdurchlaufs (Backpropagation) aufgerufen
def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0].detach() # Speichert die Gradienten, die zu dieser Schicht fließen

# Ziel-Schicht für Grad-CAM festlegen (Die letzte Faltungs-Schicht im ResNet18)
target_layer = model.layer4[-1].conv2
# Registriere die Hooks an der Ziel-Schicht, damit sie bei Inferenz und Backprop feuern
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# =====================================================================
# Hilfsfunktionen für Inference & Erklärbarkeit
# =====================================================================
def preprocess_image(image_path):
    # Lädt das Bild, wandelt es in RGB um und wendet die Transformationen an.
    image = Image.open(image_path).convert("RGB")
    orig_size = image.size  # (width, height) speichern für spätere Visualisierungen
    # Füge eine Batch-Dimension hinzu (aus [C, H, W] wird [1, C, H, W]) und verschiebe auf CPU/GPU
    tensor = transform(image).unsqueeze(0).to(device)
    return image, tensor, orig_size

def predict(image_tensor):
    # Führt die Vorhersage durch. enable_grad() ist hier wichtig für das spätere Grad-CAM!
    with torch.enable_grad(): # Gradientenberechnung aktivieren, da wir sie für Grad-CAM brauchen
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1) # Wandle rohe Modell-Outputs in Wahrscheinlichkeiten um
        pred_idx = torch.argmax(probs, dim=1).item() # Index der Klasse mit höchster Wahrscheinlichkeit
        confidence = probs[0, pred_idx].item() # Konfidenz (Sicherheit) der Vorhersage
    return output, pred_idx, confidence

def generate_gradcam(output, pred_idx):
    # Erzeugt die Grad-CAM Heatmap, die zeigt, worauf das Modell geachtet hat.
    model.zero_grad() # Setze alte Gradienten zurück
    score = output[:, pred_idx] # Nimm den Vorhersage-Wert der ermittelten Klasse
    # Backpropagation: Berechne die Gradienten des Scores in Bezug auf die Feature Maps
    score.backward(retain_graph=True) 

    # Berechne das globale Durchschnitts-Pooling (Global Average Pooling) über die Gradienten
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activation_map = activations[0] # Die gespeicherten Feature Maps aus dem Forward-Pass

    # Gewichte die Kanäle der Feature Map mit den entsprechenden Gradienten
    for i in range(activation_map.shape[0]):
        activation_map[i, :, :] *= pooled_gradients[i]

    # Summiere alle Kanäle auf, um eine einzelne 2D-Heatmap zu erhalten
    cam_map = torch.sum(activation_map, dim=0)
    cam_map = torch.relu(cam_map) # ReLU: Verwirf negative Werte (nur positive Einflüsse zählen)
    cam_map = cam_map.cpu().numpy()

    # Normalisiere die Heatmap auf Werte zwischen 0 und 1
    cam_map -= cam_map.min()
    if cam_map.max() != 0:
        cam_map /= cam_map.max()

    return cam_map

def overlay_heatmap_on_image(pil_image, cam_map, alpha=0.4):
    # Legt die generierte Heatmap transparent über das Originalbild.
    # Skaliere die kleine Heatmap auf die Originalbildgröße hoch
    cam_img = Image.fromarray(np.uint8(cam_map * 255)).resize(pil_image.size, Image.Resampling.BILINEAR)
    cam_array = np.array(cam_img) / 255.0

    # Wende eine Farbskala (Jet-Colormap: Blau=kalt, Rot=heiß) an
    heatmap = cm.jet(cam_array)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap)

    # Mische (blende) das Originalbild mit der farbigen Heatmap
    original = pil_image.convert("RGB")
    blended = Image.blend(original, heatmap_pil, alpha=alpha)

    return original, heatmap_pil, blended

def save_visualization(original, heatmap, overlay, image_path, pred_class, confidence):
    # Speichert eine kombinierte Ansicht aus Original, Heatmap und Overlay als Bilddatei.
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
    plt.close() # Gib den Speicher für den Plot wieder frei

    return save_path

# =====================================================================
# Gradio Callback-Funktionen (UI-Logik)
# =====================================================================
def run_inference_ui(image_path):
    # Wird aufgerufen, wenn der Nutzer auf 'Müll analysieren' klickt.
    if image_path is None:
        return None, "<p>Bitte lade ein Bild hoch.</p>", None, None

    try:
        # 1. Bild vorbereiten
        pil_image, image_tensor, _ = preprocess_image(image_path)
        # 2. Modellvorhersage treffen
        output, pred_idx, confidence = predict(image_tensor)

        pred_class = class_names[pred_idx]
        
        # 3. Erklärbarkeit (Heatmap) generieren
        cam_map = generate_gradcam(output, pred_idx)
        original, heatmap, overlay = overlay_heatmap_on_image(pil_image, cam_map)

        # 4. Ergebnisbild speichern
        save_path = save_visualization(
            original=original, heatmap=heatmap, overlay=overlay,
            image_path=image_path, pred_class=pred_class, confidence=confidence
        )
        
        # Emoji-Mapping für hübschere Darstellung
        icon_map = {
            "Rest": "🗑️", "Plastik": "🟡", "Papier": "📦", "Bio": "🍎", "Glas": "🍾"      
        }
        icon = icon_map.get(pred_class, "♻️")
        
        # HTML für die große Anzeige der Vorhersage
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
        
        # Gibt das Bild, das HTML, das Originalbild (als State) und die Klasse (als State) zurück
        return save_path, result_html, image_path, pred_class
        
    except Exception as e:
        return None, f"<p style='color:red;'>Fehler bei der Bildverarbeitung: {str(e)}</p>", None, None

def verify_admin(password, current_img, current_pred):
    # Überprüft das Passwort und schaltet die Admin-Ansicht frei.
    if password == ADMIN_PASSWORD:
        pred_val = current_pred if current_pred else "Noch kein Bild analysiert"
        dropdown_val = current_pred if current_pred else None
        
        return (
            gr.update(visible=False),              # Verstecke Login-Formular
            gr.update(visible=True),               # Zeige Admin-Dashboard
            gr.update(value=""),                   # Leere Login-Fehlermeldung
            gr.update(value=current_img),          # Aktualisiere Admin-Bildvorschau
            gr.update(value=pred_val),             # Zeige letzte KI-Vorhersage
            gr.update(value=dropdown_val)          # Setze Dropdown auf KI-Vorhersage vorab
        )
    else:
        return (
            gr.update(visible=True), 
            gr.update(visible=False), 
            gr.update(value="<div style='color:#ef4444; text-align:center; font-weight:600; margin-top:10px;'>❌ Falsches Passwort! Bitte versuchen Sie es erneut.</div>"),
            gr.update(value=None),                      
            gr.update(value=""),
            gr.update(value=None)
        )

def auto_reset_admin():
    # Wird aufgerufen, wenn auf den Admin-Tab geklickt wird. Loggt den User quasi aus.
    return (
        gr.update(visible=True),   # Zeige Login
        gr.update(visible=False),  # Verstecke Dashboard
        gr.update(value=""),       # Passwortfeld leeren
        gr.update(value=""),       # Fehler-HTML leeren
        gr.update(value="")        # Status-HTML leeren
    )

def save_feedback(image_path, selected_class, current_prediction):
    # Sichert ein Bild basierend auf der menschlichen Überprüfung im entsprechenden Ordner.
    if not image_path:
        return """
        <div style='padding: 12px; color: #ef4444; font-family: sans-serif; font-size: 14px;'>
            ⚠️ <strong>Kein Bild vorhanden:</strong> Bitte analysieren Sie zuerst ein Bild in der Nutzer-Ansicht.
        </div>
        """
    if not selected_class:
        return """
        <div style='padding: 12px; color: #f59e0b; font-family: sans-serif; font-size: 14px;'>
            💡 <strong>Hinweis:</strong> Bitte wählen Sie eine Kategorie aus dem Dropdown-Menü.
        </div>
        """

    try:
        # Zielordner anlegen (z.B. ./Training/Plastik)
        target_folder = os.path.join(TRAINING_DIR, selected_class)
        os.makedirs(target_folder, exist_ok=True)
        
        # Eindeutigen Dateinamen per Zeitstempel generieren
        timestamp = int(time.time())
        filename = f"feedback_{timestamp}.jpg"
        target_path = os.path.join(target_folder, filename)
        
        # Bild in den Trainingsordner kopieren
        shutil.copy(image_path, target_path)
        
        status_msg = "BESTÄTIGT" if selected_class == current_prediction else "KORRIGIERT"
        
        # Stylings für visuelles Feedback (Bestätigt vs. Korrigiert)
        if status_msg == "BESTÄTIGT":
            badge_text = f"Bestätigt: {selected_class}"
            badge_bg = "rgba(16, 185, 129, 0.15)"
            badge_color = "#10b981"
            badge_border = "1px solid rgba(16, 185, 129, 0.3)"
        else:
            badge_text = f"Korrigiert: {current_prediction} ➔ {selected_class}"
            badge_bg = "rgba(245, 158, 11, 0.15)"
            badge_color = "#f59e0b"
            badge_border = "1px solid rgba(245, 158, 11, 0.3)"
        
        return f"""
        <div style='margin-top: 25px; padding: 20px; border-radius: 10px; border: 1px solid rgba(128, 128, 128, 0.15); background-color: rgba(128, 128, 128, 0.02); font-family: sans-serif;'>
            <div style='display: flex; align-items: center; gap: 12px; flex-wrap: wrap; margin-bottom: 8px;'>
                <span style='color: #10b981; font-size: 18px; font-weight: bold;'>✓</span>
                <span style='font-weight: 600; color: inherit; font-size: 15px;'>Klassifizierung gesichert!</span>
                <span style='background-color: {badge_bg}; color: {badge_color}; border: {badge_border}; padding: 3px 10px; border-radius: 6px; font-size: 12px; font-weight: 700; letter-spacing: 0.5px;'>{badge_text}</span>
            </div>
            <div style='color: inherit; opacity: 0.6; font-size: 14px;'>
                Speicherpfad: <code style='background-color: rgba(128, 128, 128, 0.08); color: inherit; padding: 3px 8px; border-radius: 5px; font-family: monospace; font-size: 13px;'>Training/{selected_class}/</code>
            </div>
        </div>
        """
    except Exception as e:
        return f"<p style='color:#ef4444;'>Fehler beim Speichern: {str(e)}</p>"


# =====================================================================
# Gradio UI Layout (Die Benutzeroberfläche)
# =====================================================================
if __name__ == "__main__":
    
    # Direkt ausführbares JS, um den Light-Mode zu erzwingen
    force_light_js = """
    document.body.classList.remove('dark');
    document.documentElement.classList.remove('dark');
    localStorage.setItem('theme', 'light');
    """
    
    # Custom CSS für kleine Anpassungen der Optik
    custom_css = """
    footer {display: none !important;} /* Blendet den Gradio-Footer aus */
    .fixed-height-img img { max-height: 380px !important; object-fit: contain !important; }
    """
    
    # Erstelle die Gradio-Anwendung
    with gr.Blocks(
        title="Müll-Klassifizierung", 
        theme=gr.themes.Soft(), 
        css=custom_css,
        js=force_light_js
    ) as demo:
        
        # Status-Variablen, die über Nutzer-Sitzungen hinweg temporär Daten halten
        current_image_state = gr.State(None)
        current_pred_state = gr.State(None)
        
        gr.Markdown("<h1 style='text-align: center; margin-bottom: 10px;'>♻️ Müll-Klassifizierung mit KI</h1>")
        
        with gr.Tabs():
            
            # ---------------------------------------------------------
            # TAB 1: NUTZER ANSICHT
            # ---------------------------------------------------------
            with gr.TabItem("👤 Nutzer Ansicht") as user_tab:
                gr.Markdown("<p style='text-align: center; font-size: 16px; color: #666;'>Lade ein Bild von Abfall hoch. Das KI-Modell analysiert den Müll und ordnet ihn einer Kategorie zu.</p>")
                
                with gr.Row():
                    with gr.Column():
                        input_image = gr.Image(type="filepath", label="Bild hochladen", elem_classes=["fixed-height-img"])
                        analyze_btn = gr.Button("🔍 Müll analysieren", variant="primary", size="lg")
                        
                    with gr.Column():
                        output_html = gr.HTML(label="KI-Ergebnis")

                # Aufklappbares Menü für die Heatmap (verhindert Überladung der UI)
                with gr.Accordion("📊 Detailanalyse (Heatmap) einblenden", open=False):
                    gr.Markdown("Hier kannst du sehen, **welche Bildbereiche** für die Entscheidung der KI ausschlaggebend waren (rot = sehr wichtig).")
                    output_image = gr.Image(type="filepath", show_label=False, interactive=False, elem_classes=["fixed-height-img"])

                # Event Listener: Was passiert beim Klick auf "Analysieren"?
                analyze_btn.click(
                    fn=run_inference_ui, # Ruft diese Funktion auf
                    inputs=input_image,  # Reicht das hochgeladene Bild hinein
                    # Definiert, welche UI-Elemente mit den Rückgabewerten der Funktion aktualisiert werden
                    outputs=[output_image, output_html, current_image_state, current_pred_state] 
                )
            
            # ---------------------------------------------------------
            # TAB 2: ADMIN ANSICHT
            # ---------------------------------------------------------
            with gr.TabItem("🛠️ Admin Ansicht (Training)") as admin_tab:
                
                # --- Login Bereich ---
                with gr.Column(visible=True) as login_group:
                    with gr.Row():
                        with gr.Column(scale=1):
                            pass
                        with gr.Column(scale=2):
                            gr.HTML("""
                            <div style="text-align: center; padding: 40px 0 20px 0;">
                                <div style="font-size: 50px; margin-bottom: 15px; filter: drop-shadow(0px 4px 6px rgba(0,0,0,0.08));">🔒</div>
                                <h2 style="margin: 0; font-size: 24px; color: inherit; font-weight: 700; letter-spacing: -0.5px;">Admin-Authentifizierung</h2>
                                <p style="color: inherit; opacity: 0.5; margin-top: 6px; font-size: 14px;">Bitte Passwort eingeben, um Zugriff auf die Daten-Kuration zu erhalten.</p>
                            </div>
                            """)
                            pwd_input = gr.Textbox(label="Passwort", type="password", placeholder="Hier Admin-Passwort eingeben...", container=True)
                            login_btn = gr.Button("Anmelden ➔", variant="primary", size="lg")
                            login_error = gr.HTML()
                        with gr.Column(scale=1):
                            pass
                
                # --- Admin Dashboard --- (Wird erst nach Login sichtbar)
                with gr.Column(visible=False) as admin_group:
                    with gr.Row():
                        gr.HTML("""
                        <div style="border-bottom: 1px solid rgba(128, 128, 128, 0.2); padding-bottom: 16px; margin-bottom: 25px; text-align: left; margin-top: 10px; width: 100%;">
                            <h2 style="margin: 0; font-size: 26px; color: inherit; font-weight: 800; display: flex; align-items: center; gap: 10px; letter-spacing: -0.5px;">🎓 Modell-Feedback & Daten-Kuration</h2>
                            <p style="color: inherit; opacity: 0.5; margin-top: 6px; font-size: 15px;">Validieren oder korrigieren Sie hier die Ergebnisse der Nutzerseite, um den Datensatz für zukünftige Trainingszyklen zu optimieren.</p>
                        </div>
                        """)
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.HTML("<div style='font-weight: 700; color: inherit; opacity: 0.6; font-size: 13px; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 12px;'>🔎 Zuletzt analysierter Kontext</div>")
                            admin_image_preview = gr.Image(label="Vorschau", interactive=False, type="filepath", height=240)
                            admin_pred_display = gr.Textbox(label="Von der KI getroffene Vorhersage", interactive=False)

                        with gr.Column(scale=1):
                            gr.HTML("<div style='font-weight: 700; color: inherit; opacity: 0.6; font-size: 13px; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 12px;'>🛠️ Daten-Klassifizierung</div>")
                            
                            # Erlaubt dem Admin, die KI-Entscheidung zu überschreiben
                            correct_class_dropdown = gr.Dropdown(
                                choices=class_names, 
                                label="Tatsächliche, korrekte Kategorie", 
                                info="Stimmt die Erkennung? Dann einfach bestätigen. Ist sie falsch, wählen Sie die richtige Zielklasse aus."
                            )
                            
                            save_feedback_btn = gr.Button("💾 Feedback einreichen & Bild sichern", variant="primary", size="lg")
                            feedback_status_html = gr.HTML()

        # =========================================================================
        # EVENT LISTENER (Verknüpfung von UI-Aktionen mit Funktionen)
        # =========================================================================
        
        # Reset des Admin-Bereichs (Login erzwingen), sobald der Tab angeklickt wird
        admin_tab.select(
            fn=auto_reset_admin,
            inputs=None,
            outputs=[login_group, admin_group, pwd_input, login_error, feedback_status_html]
        )
        
        # Login über den Klick auf den Anmelde-Button
        login_btn.click(
            fn=verify_admin,
            inputs=[pwd_input, current_image_state, current_pred_state],
            outputs=[login_group, admin_group, login_error, admin_image_preview, admin_pred_display, correct_class_dropdown]
        )
        
        # Login über das Drücken der ENTER-Taste im Passwort-Feld
        pwd_input.submit(
            fn=verify_admin,
            inputs=[pwd_input, current_image_state, current_pred_state],
            outputs=[login_group, admin_group, login_error, admin_image_preview, admin_pred_display, correct_class_dropdown]
        )
        
        # Führt den Speichervorgang im Admin-Dashboard aus
        save_feedback_btn.click(
            fn=save_feedback,
            inputs=[current_image_state, correct_class_dropdown, current_pred_state],
            outputs=feedback_status_html
        )

    # Startet den lokalen Server (mit Freigabe des Pfades, damit Gradio Bilder lesen kann)
    demo.launch(inbrowser=True, allowed_paths=[PROJECT_ROOT])