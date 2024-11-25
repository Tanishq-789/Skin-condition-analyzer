import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Ensure correct model path and input image size
MODEL_PATH = 'skin_condition_model.keras'
IMAGE_SIZE = (224, 224)


# Load your model with caching
@st.cache_resource
def load_model_cached():
    return load_model(MODEL_PATH)


model = load_model_cached()

# Define class names and remedies
class_names = ["Acne", "Carcinoma", "Eczema", "Keratosis", "Milia", "Rosacea"]

remedies_info = {
    "Acne": ["Use Salicylic Acid", "Avoid Dairy Products", "Use Non-comedogenic Moisturizers", "Exfoliate Regularly",
             "Hydrate the Skin"],
    "Carcinoma": ["Use Sunscreen", "Avoid Tanning Beds", "Get Regular Skin Checkups", "Use Retinoids",
                  "Maintain Healthy Diet"],
    "Eczema": ["Use Moisturizers", "Avoid Triggers like Fragrances", "Use Topical Steroids", "Take Lukewarm Baths",
               "Wear Soft Fabrics"],
    "Keratosis": ["Exfoliate Gently", "Use Retinoids", "Moisturize Daily", "Use Lactic Acid",
                  "Consult a Dermatologist"],
    "Milia": ["Exfoliate Gently", "Avoid Heavy Creams", "Use Retinoids", "Keep Face Clean", "Do Not Pick at Milia"],
    "Rosacea": ["Use Gentle Skincare", "Avoid Spicy Foods", "Wear Sunscreen", "Avoid Alcohol",
                "Use Green-Tinted Moisturizers"]
}


def preprocess_image(uploaded_file):
    """Preprocesses the uploaded image to ensure correct format."""
    try:
        # Load image with target size
        img = image.load_img(uploaded_file, target_size=IMAGE_SIZE)
        img_array = image.img_to_array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1, 224, 224, 3)
        return img_array
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None


def predict_skin_condition(img_array):
    """Predicts the skin condition from preprocessed image array."""
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction, axis=1)[0]
    predicted_class_name = class_names[predicted_label]
    return predicted_class_name


# Streamlit app
st.title("Skin Condition Detector")
st.write("Upload an image of your skin condition to get a prediction and remedies.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img_array = preprocess_image(uploaded_file)

    if img_array is not None:
        # Ensure the correct input shape
        st.write(f"Image array shape: {img_array.shape}")

        # Predict the skin condition
        predicted_condition = predict_skin_condition(img_array)

        # Retrieve remedies for the predicted condition
        remedies = remedies_info.get(predicted_condition, ["No remedies found"])

        # Display the results
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        st.write(f"**Predicted Skin Condition:** {predicted_condition}")
        st.write("**Top Remedies:**")
        for i, remedy in enumerate(remedies[:5], 1):
            st.write(f"{i}. {remedy}")
    else:
        st.write("Image preprocessing failed. Please try again.")
