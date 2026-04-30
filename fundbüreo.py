import streamlit as st
import numpy as np
import os
import json
import uuid
from datetime import datetime
from PIL import Image
from ultralytics import YOLO

# ----------------------------
# 1. KONFIGURATION
# ----------------------------
UPLOAD_DIR = "uploads"
DB_FILE = "fundstuecke.json"

# Optionale "schöne" Namen. Alles was hier NICHT drin steht, 
# wird einfach 1:1 als Kategorie übernommen (z.B. "Vase", "Toaster")
PRETTY_NAMES = {
    "cell phone": "Smartphone/Elektronik",
    "laptop": "Computer",
    "backpack": "Taschen & Rucksäcke",
    "handbag": "Taschen & Rucksäcke",
    "umbrella": "Regenschirme",
    "bottle": "Trinkflaschen"
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
st.set_page_config(page_title="Dynamisches KI-Fundbüro", layout="wide", page_icon="🏺")
st.title("🏺 Dynamisches Fundbüro (Self-Organizing)")

tab1, tab2 = st.tabs(["📸 Fund registrieren", "📦 Lager durchsuchen"])

# ==========================================
# TAB 1: FUND REGISTRIEREN (DYNAMISCHE LOGIK)
# ==========================================
with tab1:
    st.header("Neues Objekt scannen")
    uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        results = model(img)
        
        res_plotted = results[0].plot()
        res_image = Image.fromarray(res_plotted[:, :, ::-1])
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(res_image, use_container_width=True)
        
        with col2:
            if len(results[0].boxes) > 0:
                # Das am besten erkannte Objekt nehmen
                label_en = model.names[int(results[0].boxes.cls[0])]
                
                # LOGIK: Checke PRETTY_NAMES, sonst nimm den Originalnamen (groß geschrieben)
                kategorie = PRETTY_NAMES.get(label_en, label_en.capitalize())
                
                st.subheader(f"Erkannt: {kategorie}")
                st.info(f"Dieses Objekt wird automatisch in die Kategorie '{kategorie}' einsortiert.")

                if st.button(f"💾 Als '{kategorie}' speichern"):
                    item_id = str(uuid.uuid4())
                    img_path = os.path.join(UPLOAD_DIR, f"{item_id}.jpg")
                    res_image.save(img_path)
                    
                    db = load_db()
                    db.append({
                        "id": item_id,
                        "label": label_en,
                        "category": kategorie, # Hier wird der dynamische Name gespeichert
                        "date": datetime.now().strftime("%d.%m.%Y, %H:%M"),
                        "img_path": img_path
                    })
                    save_db(db)
                    st.success(f"Gespeichert! Alle '{kategorie}'-Objekte sind nun gruppiert.")
            else:
                st.error("Nichts erkannt. Bitte näher ranzoomen oder Licht verbessern.")

# ==========================================
# TAB 2: LAGER & AUTOMATISCHE GRUPPIERUNG
# ==========================================
with tab2:
    db = load_db()
    
    if not db:
        st.info("Das Lager ist aktuell leer.")
    else:
        # Extrahiere alle vorhandenen Kategorien aus der Datenbank für den Filter
        vorhandene_kategorien = sorted(list(set([item["category"] for item in db])))
        
        # Filter-UI
        auswahl = st.radio("Kategorie wählen:", ["Alle anzeigen"] + vorhandene_kategorien, horizontal=True)
        
        st.divider()

        # Filtern der Daten
        display_items = [i for i in db if i["category"] == auswahl] if auswahl != "Alle anzeigen" else db

        # Anzeige
        cols = st.columns(4)
        for idx, item in enumerate(display_items):
            with cols[idx % 4]:
                st.image(item["img_path"], use_container_width=True)
                st.markdown(f"**{item['category']}**")
                st.caption(f"Gefunden: {item['date']}")
                
                if st.button("✅ Abgeholt", key=item["id"]):
                    new_db = [x for x in db if x["id"] != item["id"]]
                    save_db(new_db)
                    if os.path.exists(item["img_path"]):
                        os.remove(item["img_path"])
                    st.rerun()
