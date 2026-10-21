import io
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

MODEL_PATH = Path("plant_model.keras")
CLASSES_PATH = Path("class_names.json")

MIN_IMAGE_SIDE = 32
MIN_LEAF_COLOR_RATIO = 0.15

PREPROCESS_MODES = {
    "Raw 0-255 (same as old code)": "raw",
    "Divide by 255 (0 to 1)": "div255",
    "MobileNet style (-1 to 1)": "minus1_1",
}

st.set_page_config(
    page_title="Plant Disease Prediction",
    page_icon="🌿",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1100px;}
    .hero {
        padding: 1.4rem 1.6rem; border-radius: 14px;
        background: linear-gradient(135deg, #1b5e20 0%, #43a047 100%);
        color: #fff; margin-bottom: 1.2rem;
    }
    .hero h1 {margin: 0; font-size: 2rem; color: #fff;}
    .hero p {margin: .3rem 0 0; opacity: .92;}
    .result-card {
        padding: 1.1rem 1.3rem; border-radius: 12px;
        border: 1px solid rgba(128,128,128,.25);
        margin-bottom: .8rem;
    }
    .result-card.ok {background: rgba(67,160,71,.12); border-color: #43a047;}
    .result-card.bad {background: rgba(229,57,53,.10); border-color: #e53935;}
    .result-card.warn {background: rgba(251,192,45,.14); border-color: #fbc02d;}
    .result-card.error {background: rgba(229,57,53,.10); border-color: #e53935;}
    .result-card h3 {margin: 0 0 .2rem; font-size: 1.35rem;}
    .result-card small {opacity: .8;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASSES_PATH, encoding="utf-8") as f:
        class_names = json.load(f)
    size = model.input_shape[1:3]
    if None in size:
        size = (224, 224)
    return model, class_names, tuple(int(s) for s in size)


def clean(name: str) -> str:
    return name.replace("__", " - ").replace("_", " ").strip()


def split_label(name: str):
    crop, _, condition = clean(name).partition(" - ")
    return crop, (condition or "Unknown")


def is_healthy(name: str) -> bool:
    return "healthy" in name.lower()


def read_image(image_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(image_bytes))
    return ImageOps.exif_transpose(img).convert("RGB")


def leaf_color_ratio(img: Image.Image) -> float:
    """Share of pixels with leaf-like colors (green, yellow-green, yellow, brown)."""
    hsv = np.array(img.resize((128, 128)).convert("HSV"), dtype="float32")
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (hue >= 7) & (hue <= 120) & (sat >= 45) & (val >= 35)
    return float(mask.mean())


def preprocess(img: Image.Image, size, mode: str) -> np.ndarray:
    x = np.array(img.resize(size), dtype="float32")
    if mode == "div255":
        x = x / 255.0
    elif mode == "minus1_1":
        x = x / 127.5 - 1.0
    return np.expand_dims(x, 0)


@st.cache_data(show_spinner=False)
def predict(_model, image_bytes: bytes, size, mode: str) -> np.ndarray:
    img = read_image(image_bytes)
    return _model.predict(preprocess(img, size, mode), verbose=0)[0]


if "history" not in st.session_state:
    st.session_state.history = []

if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
    st.error(
        f"`{MODEL_PATH}` or `{CLASSES_PATH}` not found. "
        "Place both files in the same folder as app.py."
    )
    st.stop()

model, class_names, input_size = load_model()

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of predictions to show", 1, min(10, len(class_names)), 3)
    min_conf = st.slider(
        "Minimum confidence to accept (%)",
        0, 90, 40, step=5,
        help="Below this value the photo is rejected with an error.",
    )
    threshold = st.slider("Low-confidence warning (%)", 10, 95, 60, step=5)
    reject_non_leaf = st.checkbox(
        "Reject photos that do not look like a leaf",
        value=True,
        help="Uses a quick color check before running the model.",
    )
    mode_label = st.selectbox(
        "Preprocessing",
        list(PREPROCESS_MODES),
        help="Pick the same option used during training. A wrong choice gives wrong results.",
    )
    st.caption(f"Model input: {input_size[0]}x{input_size[1]} px")
    st.caption(f"Classes: {len(class_names)}")
    st.divider()
    if st.button("Clear history", use_container_width=True):
        st.session_state.history = []
        st.rerun()

st.markdown(
    """
    <div class="hero">
      <h1>🌿 Plant Disease Prediction</h1>
      <p>Upload a clear photo of a leaf or take one with your camera. AI will identify the disease.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

source = st.radio(
    "Photo source",
    ["📁 Upload", "📷 Camera"],
    horizontal=True,
    label_visibility="collapsed",
)

if source == "📁 Upload":
    file = st.file_uploader(
        "Leaf photo", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
    )
else:
    file = st.camera_input(
        "Hold the leaf in front of the camera", label_visibility="collapsed"
    )

if file is None:
    st.info("Waiting for a photo. Choose a clear leaf in good light.")
else:
    file_name = getattr(file, "name", None) or "camera_photo.jpg"
    image_bytes = file.getvalue()

    img = None
    probs = None
    error = None

    try:
        img = read_image(image_bytes)
    except Exception:
        error = "This file could not be read as an image. Upload a valid JPG or PNG."

    if error is None and min(img.size) < MIN_IMAGE_SIDE:
        error = f"Image is too small. Use a photo of at least {MIN_IMAGE_SIDE}x{MIN_IMAGE_SIDE} px."

    if error is None and reject_non_leaf and leaf_color_ratio(img) < MIN_LEAF_COLOR_RATIO:
        error = "This does not look like a leaf photo. Upload a clear photo of a single leaf."

    if error is None:
        try:
            with st.spinner("Analyzing photo..."):
                probs = predict(model, image_bytes, input_size, PREPROCESS_MODES[mode_label])
        except Exception as exc:
            error = f"Prediction failed: {exc}"

    if error is None:
        order = np.argsort(probs)[::-1][:top_k]
        best = int(order[0])
        best_conf = float(probs[best]) * 100
        if best_conf < min_conf:
            error = (
                f"The model could not match this photo to a known leaf "
                f"(best match {best_conf:.1f}%, minimum is {min_conf}%). "
                "Upload a clear photo of a single leaf."
            )

    col_img, col_res = st.columns([1, 1.2], gap="large")

    with col_img:
        if img is not None:
            st.image(img, caption=file_name, use_container_width=True)
        else:
            st.warning("Preview not available for this file.")

    with col_res:
        if error is not None:
            st.markdown(
                f"""
                <div class="result-card error">
                  <small>Disease detected</small>
                  <h3>Error</h3>
                  <small>Crop: -</small>
                </div>
                """,
                unsafe_allow_html=True,
            )
            m1, m2 = st.columns(2)
            m1.metric("Confidence", "0%")
            m2.metric("Gap to runner-up", "0%")
            st.error(error)
        else:
            crop, condition = split_label(class_names[best])

            if best_conf < threshold:
                kind, title = "warn", "Result is uncertain"
            elif is_healthy(class_names[best]):
                kind, title = "ok", "Leaf is healthy"
            else:
                kind, title = "bad", "Disease detected"

            st.markdown(
                f"""
                <div class="result-card {kind}">
                  <small>{title}</small>
                  <h3>{condition}</h3>
                  <small>Crop: {crop}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )
            m1, m2 = st.columns(2)
            m1.metric("Confidence", f"{best_conf:.2f}%")
            runner_up = float(probs[order[1]]) * 100 if len(order) > 1 else 0.0
            m2.metric("Gap to runner-up", f"{best_conf - runner_up:.2f}%")

            if kind == "warn":
                st.warning(
                    "Model confidence is low. Retake a clear photo of a single leaf "
                    "in bright light. Also check the preprocessing setting."
                )

            df = pd.DataFrame(
                {
                    "Crop": [split_label(class_names[i])[0] for i in order],
                    "Condition": [split_label(class_names[i])[1] for i in order],
                    "Confidence": [float(probs[i]) * 100 for i in order],
                }
            )
            st.dataframe(
                df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Confidence": st.column_config.ProgressColumn(
                        "Confidence", format="%.2f%%", min_value=0, max_value=100
                    )
                },
            )

            report = {
                "time": datetime.now().isoformat(timespec="seconds"),
                "file": file_name,
                "prediction": clean(class_names[best]),
                "confidence_percent": round(best_conf, 2),
                "top": df.round(2).to_dict(orient="records"),
            }
            st.download_button(
                "Download report (JSON)",
                data=json.dumps(report, indent=2, ensure_ascii=False),
                file_name="plant_report.json",
                mime="application/json",
                use_container_width=True,
            )

            entry = (file_name, report["prediction"], report["confidence_percent"])
            if not st.session_state.history or st.session_state.history[-1][:2] != entry[:2]:
                st.session_state.history.append(entry)

    st.caption(
        "This is an AI estimate, not a final diagnosis. Consult a local "
        "agriculture expert for confirmation."
    )

if st.session_state.history:
    with st.expander(f"Past predictions ({len(st.session_state.history)})"):
        st.dataframe(
            pd.DataFrame(
                st.session_state.history[::-1],
                columns=["File", "Prediction", "Confidence %"],
            ),
            hide_index=True,
            use_container_width=True,
        )
