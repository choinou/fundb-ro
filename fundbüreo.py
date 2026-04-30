import streamlit as st
import numpy as np
import os
import json
import uuid
from datetime import datetime
from PIL import Image
from ultralytics import YOLO

# ----------------------------
# Datei- und Ordner-Setup
# ----------------------------
UPLOAD_DIR = "uploads"
DB_FILE = "fundstuecke.json"

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

if not os.path.exists(DB_FILE):
    with open(DB_FILE, "w") as f:
        json.dump([], f)

def load_db():
    try:
        with open(DB_FILE, "r") as f:
            return json.load(f)
    except:
        return []

def save_db(data):
    with open(DB_FILE, "w") as f:
        json.dump(data, f, indent=4)

# ----------------------------
# Seiteneinstellungen
# ----------------------------
st.set_page_config(page_title="YOLOv8 Fundbüro", layout="centered", page_icon="🔍")
st.title("🔍 KI-Fundbüro (YOLOv8)")

# ----------------------------
# YOLOv8 Modell laden
# ----------------------------
@st.cache_resource
def load_yolo_model():
    # Nutzt das vortrainierte YOLOv8 Nano Modell (leichtgewichtiger)
    # Es erkennt automatisch 80 Standard-Objekte (COCO Dataset)
    model = YOLO("yolov8n.pt") 
    return model

model = load_yolo_model()

# ----------------------------
# UI Layout
# ----------------------------
tab1, tab2 = st.tabs(["📸 Fundstück scannen", "📦 Lagerbestand"])

# ==========================================
# TAB 1: OBJEKTERKENNUNG & SPEICHERN
# ==========================================
with tab1:
    st.subheader("Neues Objekt erfassen")
    
    uploaded_file = st.file_uploader("Bild hochladen...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        
        # YOLOv8 Inference
        results = model(img)
        
        # Das erste Ergebnis visualisieren (Boxen zeichnen)
        res_plotted = results[0].plot() # Gibt ein BGR-Array zurück
        # Konvertierung von BGR zu RGB für Streamlit
        res_image = Image.fromarray(res_plotted[:, :, ::-1])
        
        st.image(res_image, caption="Erkennungsergebnis", use_container_width=True)

        # Gefundene Objekte extrahieren
        detected_objects = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            conf = float(box.conf[0])
            detected_objects.append(f"{label} ({conf:.2%})")

        if detected_objects:
            st.success(f"Erkannt: {', '.join(detected_objects)}")
            
            # Da YOLO mehrere Objekte finden kann, nehmen wir das mit der höchsten Confidence für die DB
            best_guess = model.names[int(results[0].boxes.cls[0])]

            if st.button("💾 In Datenbank speichern"):
                item_id = str(uuid.uuid4())
                img_path = os.path.join(UPLOAD_DIR, f"{item_id}.jpg")

                # Speichere das Originalbild oder das markierte Bild (hier: markiert)
                res_image.save(img_path)

                db = load_db()
                db.append({
                    "id": item_id,
                    "class_name": best_guess,
                    "all_detected": detected_objects,
                    "date": datetime.now().strftime("%d.%m.%Y %H:%M"),
                    "img_path": img_path
                })
                save_db(db)
                st.balloons()
                st.success(f"'{best_guess}' wurde im System registriert!")
        else:
            st.warning("Kein bekanntes Objekt gefunden. Versuche es mit einem anderen Foto.")

# ==========================================
# TAB 2: ÜBERSICHT & ABHOLEN
# ==========================================
with tab2:
    st.subheader("Aktuelle Fundstücke")
    db = load_db()

    if not db:
        st.info("Das Lager ist leer.")
    else:
        # Erstellt ein Grid-Layout
        cols = st.columns(2)
        for idx, item in enumerate(db):
            with cols[idx % 2]:
                if os.path.exists(item["img_path"]):
                    st.image(item["img_path"], use_container_width=True)
                
                st.markdown(f"**Typ:** {item['class_name'].capitalize()}")
                st.caption(f"📅 {item['date']}")
                
                if st.button(f"✅ Abgeholt", key=item["id"]):
                    new_db = [x for x in db if x["id"] != item["id"]]
                    save_db(new_db)
                    if os.path.exists(item["img_path"]):
                        os.remove(item["img_path"])
                    st.rerun()
