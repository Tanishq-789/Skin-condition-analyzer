import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import google.generativeai as genai
import time
from bs4 import BeautifulSoup
import os
import requests

# --- New imports for the working scraper ---
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


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
    """
    Downloads the TFLite model from Hugging Face if not already present,
    then loads it into a TFLite Interpreter. This function is cached to run only once.
    """
    MODEL_PATH = "skin_model.tflite"
    # CORRECT URL: This must be the direct download link (using /resolve/main/)
    MODEL_URL = "https://huggingface.co/Tanishq77/skin-condition-classifier/resolve/main/skin_model.tflite"

    if not os.path.exists(MODEL_PATH):
        st.info("Downloading AI model... This may take a moment on first run.")
        try:
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            progress_bar = st.progress(0, text="Downloading Model...")
            
            with open(MODEL_PATH, "wb") as f:
                downloaded_size = 0
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    progress = min(int(100 * downloaded_size / total_size), 100) if total_size > 0 else 50
                    progress_bar.progress(progress, text=f"Downloading Model... {progress}%")
            progress_bar.empty()
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to download model: {e}")
            return None

    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        print("TFLite model loaded successfully.")
        return interpreter
    except Exception as e:
        st.error(f"Error loading TFLite model: {e}")
        return None

# Load the interpreter using the cached function
interpreter = load_and_cache_tflite_model()
if interpreter is None:
    st.error("Model could not be loaded. The application cannot proceed.")
    st.stop()


class_names = ["Acne", "Carcinoma", "Eczema", "Keratosis", "Milia", "Rosacea"]


# -------------------- BACKEND FUNCTIONS --------------------

def get_prediction(image):
    """Resize, preprocess, and predict using the TFLite interpreter."""
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
    """Query the Gemini model and return the response text."""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error communicating with Gemini: {e}"


# In Skincare_rec.py

def scrape_justdial(city: str):
    """
    Scrapes Justdial using a system-installed Chromium driver on Streamlit Cloud.
    """
    clinic_list = []
    sanitized_city = city.lower().strip().replace(" ", "-")
    url = f"https://www.justdial.com/{sanitized_city}/Dermatologists"

    try:
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument(
            'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')

        # UPDATED: Point directly to the system-installed chromedriver
        service = Service(executable_path="/usr/bin/chromedriver")
        
        with webdriver.Chrome(service=service, options=options) as driver:
            driver.get(url)
            time.sleep(7)
            page_source = driver.page_source

        soup = BeautifulSoup(page_source, 'html.parser')
        clinic_cards = soup.select("div.resultbox")

        if not clinic_cards:
            return []

        for card in clinic_cards:
            name_element = card.select_one('h3.resultbox_title_anchor')
            location_element = card.select_one('div.locatcity')
            contact_element = card.select_one('span.callcontent')

            if name_element:
                name = name_element.text.strip()
                location = location_element.text.strip() if location_element else "Location not available"
                contact = contact_element.text.strip() if contact_element else "Contact not available"
                clinic_list.append({"name": name, "location": location, "contact": contact})

    except Exception as e:
        # Provide a more specific error message for this common issue
        if "Permission denied" in str(e) or "cannot find" in str(e):
             st.error("Error with browser driver. Ensure 'chromium' and 'chromium-driver' are in packages.txt.")
        else:
            st.error(f"An error occurred during scraping: {e}")
        return None
        
    return clinic_list


# -------------------- STREAMLIT UI --------------------

st.title("🌿 AI Skincare Assistant")
st.markdown(
    "Upload a clear image of your skin concern. Our AI will analyze it, suggest remedies, and help you find local professionals.")

st.warning(
    "⚠️ **Disclaimer:** This tool provides AI-generated suggestions and is not a substitute for professional medical advice. Please consult a certified dermatologist for an accurate diagnosis.")

uploaded_file = st.file_uploader("📷 Upload a clear skin image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Analyzing your skin and preparing recommendations..."):
        prediction, confidence = get_prediction(image)

        progress_bar = st.progress(0, text="Analyzing...")
        for percent_complete in range(100):
            time.sleep(0.01)
            progress_bar.progress(percent_complete + 1, text=f"Confidence: {confidence * 100:.2f}%")

        remedies_prompt = f"Generate 5-6 actionable home remedies for someone with {prediction}. Format them as a bulleted list."
        skincare_plan_prompt = f"Create a simple 7-day skincare plan (Morning and Night routine) for managing {prediction}. Be concise."
        tips_prompt = f"Provide 3 essential tips for managing skin that is prone to {prediction}."

        remedies = get_gemini_response(remedies_prompt)
        skincare_plan = get_gemini_response(skincare_plan_prompt)
        skin_type_tips = get_gemini_response(tips_prompt)

    st.success("Analysis Complete!")

    with st.container(border=True):
        st.markdown(f"### 🎯 Condition: **{prediction}**")
        st.progress(confidence, text=f"Confidence: {confidence * 100:.2f}%")
        st.markdown("---")

        st.markdown("#### 💊 Gemini-Generated Remedies")
        st.markdown(remedies)

        with st.expander("🗓️ View 7-Day Skincare Plan"):
            st.markdown(skincare_plan)

        st.info(f"**🌿 Tips for {prediction}-Prone Skin:**\n{skin_type_tips}", icon="💡")

    st.markdown("---")
    st.subheader("👩‍⚕️ Consult Professionals")
    st.markdown("Find top-rated dermatologists in your city for an expert consultation.")

    city = st.text_input("Enter your city:", placeholder="e.g., Pune, Mumbai, Delhi")

    if city:
        with st.spinner(f"Finding dermatologists in {city}... (this may take a moment)"):
            clinic_data = scrape_justdial(city)

            if clinic_data is None:
                st.error("Failed to fetch data. The website may be blocking requests or is temporarily down.")
            elif not clinic_data:
                st.warning(
                    f"Could not find any dermatologists for '{city}' on Justdial. Please check the spelling or try a different major city.")
            else:
                st.success(f"Found {len(clinic_data)} top clinics in {city}:")
                for clinic in clinic_data:
                    with st.container(border=True):
                        st.markdown(f"**🏨 {clinic['name']}**")
                        st.text(f"📍 Location: {clinic['location']}")
                        st.text(f"📞 Contact:  {clinic['contact']}")
                st.caption("Data sourced from Justdial.com and may not be exhaustive.")
