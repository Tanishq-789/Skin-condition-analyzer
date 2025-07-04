# skin_rec_UI.py

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import tempfile
import os
import google.generativeai as genai
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input  # ✅ Corrected

# -------------------- CONFIG --------------------
# Set Gemini API key securely via .streamlit/secrets.toml
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Load the trained model (should be saved in .keras format)
model = tf.keras.models.load_model("skin_model.keras")
class_names = ["Acne", "Carcinoma", "Eczema", "Keratosis", "Milia", "Rosacea"]

# -------------------- FUNCTIONS --------------------

def get_prediction(image):
    """Resize and preprocess image, then predict class."""
    img = image.resize((224, 224))
    arr = np.array(img).astype("float32")
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)  # ✅ Use official EfficientNetV2 preprocessing
    preds = model.predict(arr)[0]
    idx = np.argmax(preds)
    return class_names[idx], preds[idx], preds


def query_gemini(prompt):
    """Query Gemini 1.5 Flash model."""
    response = gemini_model.generate_content(prompt)
    return response.text


def generate_pdf_report(file_path, prediction, confidence, skincare_plan, explanation):
    """Create a basic skin condition PDF report."""
    c = canvas.Canvas(file_path, pagesize=letter)
    c.drawString(50, 750, "🧠 Skin Condition Report")
    c.drawString(50, 720, f"Condition: {prediction} ({confidence * 100:.2f}% confidence)")
    c.drawString(50, 690, "🧠 Severity Analysis:")
    c.drawString(70, 670, explanation[:250])
    c.drawString(50, 630, "🧴 Skincare Plan:")
    c.drawString(70, 610, skincare_plan[:250])
    c.save()
    return file_path

# -------------------- STREAMLIT UI --------------------

st.set_page_config(page_title="Skin Condition Classifier", layout="wide")
st.title("🧠 AI-Powered Skin Condition Classifier")

uploaded_file = st.file_uploader("📷 Upload a clear skin image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing image..."):
        prediction, confidence, _ = get_prediction(image)
        st.success(f"🧬 Prediction: **{prediction}** ({confidence * 100:.2f}% confidence)")

        # ------------------ Severity Analysis ------------------
        severity_prompt = f"A user has {prediction}. The model has {confidence*100:.2f}% confidence. Rate the severity as Mild, Moderate, or Severe with brief explanation."
        severity = query_gemini(severity_prompt)
        st.info(f"🩺 **Severity**: {severity}")

        # ------------------ Ingredient Analysis ------------------
        ingredients = st.text_area("🔍 Enter product ingredients (comma-separated):")
        if ingredients:
            ingredient_prompt = f"Analyze these ingredients for someone with {prediction}: {ingredients}. Identify helpful vs harmful ones."
            analysis = query_gemini(ingredient_prompt)
            st.markdown("📊 **Ingredient Analysis:**")
            st.write(analysis)

        # ------------------ Personalized Skincare Plan ------------------
        skin_type = st.selectbox("Your Skin Type", ["Oily", "Dry", "Combination", "Sensitive"])
        age = st.slider("Your Age", 10, 70, 25)
        plan_prompt = f"Generate a 7-day skincare routine for a {age}-year-old with {skin_type} skin suffering from {prediction}."
        skincare_plan = query_gemini(plan_prompt)
        st.markdown("📅 **Skincare Plan:**")
        st.text(skincare_plan)

        # ------------------ Gemini Q&A Chatbox ------------------
        st.markdown("💬 **Ask a Dermatology-related Question:**")
        user_q = st.text_input("Your question")
        if user_q:
            full_prompt = f"User has {prediction}. Question: {user_q}. Respond with medically relevant, educational insight (non-diagnostic)."
            answer = query_gemini(full_prompt)
            st.write(answer)

        # ------------------ PDF Download ------------------
        if st.button("📄 Generate PDF Report"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                path = generate_pdf_report(tmp.name, prediction, confidence, skincare_plan, severity)
                with open(path, "rb") as f:
                    st.download_button("⬇️ Download PDF Report", f, file_name="skin_report.pdf", mime="application/pdf")
