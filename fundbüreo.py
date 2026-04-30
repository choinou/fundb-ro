import streamlit as st
import numpy as np
import os
import json
import uuid
from datetime import datetime
from PIL import Image
from ultralytics import YOLO

# ----------------------------
# 1. KONFIGURATION & KATEGORIEN
# ----------------------------
UPLOAD_DIR = "uploads"
DB_FILE = "fundstuecke.json"

# Mapping von YOLO (COCO) Labels zu deutschen Kategorien
CATEGORY_MAP = {
    "cell phone": "Elektronik", "laptop": "Elektronik", "mouse": "Elektronik", "keyboard": "Elektronik",
    "remote": "Elektronik", "tv": "Elektronik", "watch": "Accessoires", "backpack": "Taschen & Gepäck",
    "handbag": "Taschen & Gepäck", "suitcase": "Taschen & Gepäck", "umbrella": "Accessoires",
    "bottle": "Haushalt", "cup": "Haushalt", "wine glass": "Haushalt", "knife": "Haushalt",
    "fork": "Haushalt", "spoon": "Haushalt", "bowl": "Haushalt", "bicycle": "Fahrzeuge/Sport",
    "skateboard": "Fahrzeuge/Sport", "sports ball": "Fahrzeuge/Sport", "tennis racket": "Fahrzeuge/Sport",
    "gloves": "Kleidung", "scarf": "Kleidung", "tie": "Kleidung", "hat": "Kleidung",
    "handbag": "Taschen & Gepäck", "book": "Büro & Medien", "scissors": "Büro & Medien",
    "teddy bear": "Spielzeug", "hair drier": "Elektronik/Hygiene", "toothbrush": "Hygiene"
}

# ----------------------------
# 2. HILFSFUNKTIONEN
# ----------------------------
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

def load_db():
    if not os.path.exists(DB_FILE):
        return []
    try:
        with open(DB_FILE, "r") as f:
            return json.load(f)
    except:
        return []

def save_db(data):
    with open(DB_FILE, "w") as f:
        json.dump(data, f, indent=4)

@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

model = load_yolo()

# ----------------------------
# 3. UI LAYOUT
# ----------------------------
st.set_page_config(page_title="KI Fundbüro Pro", layout="wide", page_icon="🕵️")
st.title("🕵️ Digitales Fundbüro mit YOLOv8")

tab1, tab2 = st.tabs(["📸 Fund melden", "📦 Lager & Suche"])

# ==========================================
# TAB 1: FUND MELDEN
# ==========================================
with tab1:
    st.header("Neues Fundstück registrieren")
    uploaded_file = st.file_uploader("Bild aufnehmen oder hochladen", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        results = model(img)
        
        # Resultat-Bild mit Boxen erzeugen
        res_plotted = results[0].plot()
        res_image = Image.fromarray(res_plotted[:, :, ::-1])
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(res_image, caption="KI-Erkennung", use_container_width=True)
        
        with col2:
            if len(results[0].boxes) > 0:
                # Primäres Objekt erkennen
                label_en = model.names[int(results[0].boxes.cls[0])]
                confidence = float(results[0].boxes.conf[0])
                kategorie = CATEGORY_MAP.get(label_en, "Sonstiges")
                
                st.metric("Erkannt als:", label_en.capitalize())
                st.write(f"**Kategorie:** {kategorie}")
                st.write(f"**Sicherheit:** {confidence:.2%}")

                if st.button("💾 In Datenbank speichern"):
                    item_id = str(uuid.uuid4())
                    img_path = os.path.join(UPLOAD_DIR, f"{item_id}.jpg")
                    res_image.save(img_path)
                    
                    db = load_db()
                    db.append({
                        "id": item_id,
                        "label": label_en,
                        "category": kategorie,
                        "date": datetime.now().strftime("%d.%m.%Y, %H:%M"),
                        "img_path": img_path
                    })
                    save_db(db)
                    st.success("Gegenstand erfolgreich archiviert!")
            else:
                st.warning("Kein Objekt erkannt. Bitte lade ein deutlicheres Bild hoch.")

# ==========================================
# TAB 2: LAGERÜBERSICHT
# ==========================================
with tab2:
    db = load_db()
    
    # Filter-Optionen
    alle_kategorien = sorted(list(set([item["category"] for item in db] + ["Alle"])))
    filter_kat = st.selectbox("Nach Kategorie filtern:", alle_kategorien, index=alle_kategorien.index("Alle"))
    
    filtered_db = [i for i in db if i["category"] == filter_kat] if filter_kat != "Alle" else db

    if not filtered_db:
        st.info("Keine Fundstücke in dieser Kategorie vorhanden.")
    else:
        # Anzeige in 4er Spalten
        cols = st.columns(4)
        for idx, item in enumerate(filtered_db):
            with cols[idx % 4]:
                st.image(item["img_path"], use_container_width=True)
                st.markdown(f"**{item['category']}**")
                st.caption(f"{item['label'].capitalize()} | {item['date']}")
                
                if st.button("✅ Abholen", key=item["id"]):
                    new_db = [x for x in db if x["id"] != item["id"]]
                    save_db(new_db)
                    if os.path.exists(item["img_path"]):
                        os.remove(item["img_path"])
                    st.rerun()
