import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import google.generativeai as genai
import time
from bs4 import BeautifulSoup
import os
import requests

# -------------------- CONFIG --------------------

st.set_page_config(page_title="Skincare AI Assistant", layout="centered")

# --- AI & MODEL CONFIGURATION ---
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
except Exception as e:
    st.error(f"Error configuring Gemini API: {e}")
    st.stop()


# --- TFLite MODEL DOWNLOADING AND LOADING ---
@st.cache_resource
def load_and_cache_tflite_model():
    MODEL_PATH = "skin_model.tflite"
    MODEL_URL = "https://huggingface.co/Tanishq77/skin-condition-classifier/resolve/main/skin_model.tflite"

    if not os.path.exists(MODEL_PATH):
        st.info("Downloading AI model... This may take a moment on first run.")
        try:
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to download model: {e}")
            return None

    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Error loading TFLite model: {e}")
        return None

interpreter = load_and_cache_tflite_model()
if interpreter is None:
    st.stop()

class_names = ["Acne", "Carcinoma", "Eczema", "Keratosis", "Milia", "Rosacea"]


# -------------------- BACKEND FUNCTIONS --------------------

def get_prediction(image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    img = image.resize((input_details[0]['shape'][1], input_details[0]['shape'][2]))
    arr = np.array(img).astype("float32")
    arr = np.expand_dims(arr, axis=0)
    arr = tf.keras.applications.efficientnet_v2.preprocess_input(arr)

    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()
    
    preds = interpreter.get_tensor(output_details[0]['index'])[0]
    idx = np.argmax(preds)
    confidence = float(preds[idx])
    return class_names[idx], confidence


def get_gemini_response(prompt):
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error communicating with Gemini: {e}"


def scrape_with_api(city: str):
    """Scrapes Justdial via its internal API (fast, for deployment)."""
    clinic_list = []
    api_url = "https://www.justdial.com/api/search"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': f'https://www.justdial.com/{city.lower().strip()}/Dermatologists'
    }
    payload = {"search": "Dermatologists", "location": city.lower().strip()}

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        data = response.json()
        if data.get("results"):
            for clinic in data["results"]:
                clinic_list.append({
                    "name": clinic.get("name", "N/A"),
                    "location": clinic.get("address", "N/A"),
                    "contact": clinic.get("contact_number", "N/A")
                })
    except Exception as e:
        st.error(f"Failed to fetch data using API method: {e}")
        return None
    return clinic_list


def scrape_with_selenium(city: str):
    """Scrapes Justdial using Selenium (resource-heavy, for local testing only)."""
    # --- IMPORTS MOVED INSIDE THE FUNCTION ---
    # This prevents the deployed app from crashing.
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium.webdriver.chrome.options import Options

    clinic_list = []
    url = f"https://www.justdial.com/{city.lower().strip()}/Dermatologists"
    try:
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        service = Service(ChromeDriverManager().install())
        with webdriver.Chrome(service=service, options=options) as driver:
            driver.get(url)
            time.sleep(7)
            page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')
        clinic_cards = soup.select("div.resultbox")
        for card in clinic_cards:
            name_element = card.select_one('h3.resultbox_title_anchor')
            if name_element:
                location_element = card.select_one('div.locatcity')
                contact_element = card.select_one('span.callcontent')
                clinic_list.append({
                    "name": name_element.text.strip(),
                    "location": location_element.text.strip() if location_element else "N/A",
                    "contact": contact_element.text.strip() if contact_element else "N/A"
                })
    except Exception as e:
        st.error(f"An error occurred during Selenium scraping: {e}")
        return None
    return clinic_list


# -------------------- STREAMLIT UI --------------------
st.title("🌿 AI Skincare Assistant")
st.markdown("Upload a clear image of your skin concern for an AI-powered analysis, recommendations, and help finding local professionals.")
st.warning("⚠️ **Disclaimer:** This tool is for educational purposes and is not a substitute for professional medical advice.")

uploaded_file = st.file_uploader("📷 Upload a clear skin image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    with st.spinner("Analyzing your skin and preparing recommendations..."):
        prediction, confidence = get_prediction(image)
        remedies_prompt = f"Generate 5-6 actionable home remedies for {prediction}. Format as a bulleted list."
        remedies = get_gemini_response(remedies_prompt)

    st.success("Analysis Complete!")
    with st.container(border=True):
        st.markdown(f"### 🎯 Condition: **{prediction}**")
        st.progress(confidence, text=f"Confidence: {confidence * 100:.2f}%")
        st.markdown("---")
        st.markdown("#### 💊 Gemini-Generated Remedies")
        st.markdown(remedies)

    st.markdown("---")
    st.subheader("👩‍⚕️ Consult Professionals")
    city = st.text_input("Enter your city to find local clinics:", placeholder="e.g., Pune, Mumbai")

    if city:
        with st.spinner(f"Finding dermatologists in {city}..."):
            scraper_mode = os.getenv("SCRAPER_MODE", "api")

            if scraper_mode == "selenium":
                st.info("Running in local Selenium mode...")
                clinic_data = scrape_with_selenium(city)
            else:
                clinic_data = scrape_with_api(city)
            
            if clinic_data:
                st.success(f"Found {len(clinic_data)} clinics in {city}:")
                for clinic in clinic_data:
                    with st.container(border=True):
                        st.markdown(f"**🏨 {clinic['name']}**")
                        st.text(f"📍 Location: {clinic['location']}")
                        st.text(f"📞 Contact:  {clinic['contact']}")
                st.caption("Data sourced from Justdial.com.")
            else:
                st.warning(f"Could not find any clinics for '{city}'. This may be due to anti-scraping measures by the site.")
