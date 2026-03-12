import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import os
import json
import uuid
from datetime import datetime

# ----------------------------
# Datei- und Ordner-Setup für Speicherfunktion
# ----------------------------
UPLOAD_DIR = "uploads"
DB_FILE = "fundstuecke.json"

# Ordner für Bilder erstellen, falls nicht vorhanden
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# JSON-Datenbank erstellen, falls nicht vorhanden
if not os.path.exists(DB_FILE):
    with open(DB_FILE, "w") as f:
        json.dump([], f)

# Hilfsfunktionen für die Datenbank
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
st.set_page_config(page_title="Digitales Fundbüro", layout="centered")
st.title("Digitales Fundbüro mit Keras Modell")

# ----------------------------
# Modell & Labels laden
# ----------------------------
@st.cache_resource
def load_keras_model():
    # Stelle sicher, dass "keras_model.h5" im Repository liegt
    model = load_model("keras_model.h5", compile=False)
    return model

model = load_keras_model()

# Labels laden
try:
    with open("labels.txt", "r") as f:
        class_names = f.readlines()
except FileNotFoundError:
    st.error("Datei 'labels.txt' nicht gefunden. Bitte prüfen!")
    st.stop()

# ----------------------------
# UI Layout: Tabs für die Navigation
# ----------------------------
tab1, tab2 = st.tabs(["🔍 Neues Fundstück melden", "📦 Übersicht der Fundstücke"])

# ==========================================
# TAB 1: FUNDSTÜCK MELDEN
# ==========================================
with tab1:
    st.subheader("Fundstück einscannen und abspeichern")
    
    uploaded_file = st.file_uploader(
        "Lade ein Bild hoch...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Bild anzeigen
        image = Image.open(uploaded_file).convert("RGB")
        # FIX: Hier wurde use_column_width verwendet für Kompatibilität mit v1.32.2
        st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

        # Bild Preprocessing
        size = (224, 224)
        image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image_resized)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # Vorhersage
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = float(prediction[0][index])

        # Ergebnis anzeigen
        st.write("---")
        st.write(f"**Erkannte Klasse:** {class_name}")
        st.write(f"**Sicherheit:** {confidence_score:.2%}")

        # Speichern-Button
        if st.button("💾 Dieses Fundstück speichern"):
            # Eindeutige ID für das Bild erstellen
            item_id = str(uuid.uuid4())
            file_ext = uploaded_file.name.split('.')[-1]
            img_path = os.path.join(UPLOAD_DIR, f"{item_id}.{file_ext}")

            # Bild physisch speichern
            with open(img_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Infos in die JSON-Datenbank schreiben
            db = load_db()
            db.append({
                "id": item_id,
                "class_name": class_name,
                "confidence": confidence_score,
                "date": datetime.now().strftime("%d.%m.%Y %H:%M"),
                "img_path": img_path
            })
            save_db(db)
            
            st.success("Erfolgreich gespeichert! Schau in der Übersicht nach.")

# ==========================================
# TAB 2: ÜBERSICHT & ABHOLEN
# ==========================================
with tab2:
    st.subheader("Alle gespeicherten Fundstücke")
    
    db = load_db()

    if not db:
        st.info("Bisher wurden keine Fundstücke gemeldet.")
    else:
        # Fundstücke in Spalten anzeigen
        cols = st.columns(2)
        
        for idx, item in enumerate(db):
            col = cols[idx % 2] 
            with col:
                if os.path.exists(item["img_path"]):
                    # FIX: Auch hier use_column_width verwendet
                    st.image(item["img_path"], use_column_width=True)
                
                st.write(f"**Gegenstand:** {item['class_name']}")
                st.write(f"**Gefunden am:** {item['date']}")
                
                # Button zum Löschen/Abholen
                if st.button(f"🙋‍♂️ Das ist meins!", key=item["id"]):
                    # Aus Datenbank entfernen
                    new_db = [x for x in db if x["id"] != item["id"]]
                    save_db(new_db)
                    
                    # Datei löschen
                    if os.path.exists(item["img_path"]):
                        os.remove(item["img_path"])
                        
                    st.success("Gegenstand abgeholt!")
                    st.rerun()
