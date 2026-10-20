import json
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Plant Disease Prediction")


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("plant_model.keras")
    with open("class_names.json") as f:
        class_names = json.load(f)
    return model, class_names


model, class_names = load_model()


def clean(name):
    return name.replace("__", " - ").replace("_", " ")


st.title("Plant Disease Prediction")
st.write("Patte ki saaf photo upload karo.")

file = st.file_uploader("Leaf photo", type=["jpg", "jpeg", "png"])

if file is not None:
    img = Image.open(file).convert("RGB")
    st.image(img, width=300)

    x = np.array(img.resize((224, 224)), dtype="float32")
    x = np.expand_dims(x, 0)
    p = model.predict(x, verbose=0)[0]

    top = np.argsort(p)[::-1][:3]
    best = top[0]
    st.success(clean(class_names[best]) + " (" + str(round(float(p[best]) * 100, 2)) + "%)")

    for i in top:
        st.write(clean(class_names[i]))
        st.progress(float(p[i]))