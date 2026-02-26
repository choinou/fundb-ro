import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

# ----------------------------
# Seiteneinstellungen
# ----------------------------
st.set_page_config(page_title="Bildklassifikation", layout="centered")
st.title("Bildklassifikation mit Keras Modell")

# ----------------------------
# Modell laden (Caching wichtig!)
# ----------------------------
@st.cache_resource
def load_keras_model():
    model = load_model("keras_Model.h5", compile=False)
    return model

model = load_keras_model()

# Labels laden
with open("labels.txt", "r") as f:
    class_names = f.readlines()

# ----------------------------
# Bild-Upload
# ----------------------------
uploaded_file = st.file_uploader(
    "Lade ein Bild hoch...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Bild anzeigen
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    # ----------------------------
    # Bild Preprocessing
    # ----------------------------
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)

    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # ----------------------------
    # Vorhersage
    # ----------------------------
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = float(prediction[0][index])

    # ----------------------------
    # Ergebnis anzeigen
    # ----------------------------
    st.subheader("Ergebnis")
    st.write(f"**Klasse:** {class_name}")
    st.write(f"**Konfidenz:** {confidence_score:.2%}")
