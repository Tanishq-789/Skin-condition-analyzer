# 🧠 AI Skincare Assistant

An end-to-end AI-powered web application built with Streamlit that leverages a deep learning model (EfficientNetV2B0) to classify six common skin conditions from facial images, generates personalized treatment plans using Gemini LLM, analyzes skincare ingredients, and assists users in locating local dermatologists through location-based web scraping.

---

## 🏗️ Project Structure

This project is a monolithic Streamlit application. The model file is hosted on Hugging Face and downloaded on the first run.
```
skin-condition-app/
├── 📄 Skincare_rec.py               # The main Streamlit application script
├── 📄 requirements.txt     # Python dependencies for deployment
├── 📁 .streamlit/
│   └── secrets.toml        # For storing API keys securely
└── README.md               # Project overview (this file)
```

---

## 🧩 Features

-   **🔍 Skin Condition Prediction**
    -   Upload a clear image of a skin concern.
    -   The app uses a fine-tuned **EfficientNetV2B0** model to predict one of six conditions:
        -   Acne
        -   Carcinoma
        -   Eczema
        -   Keratosis
        -   Milia
        -   Rosacea

-   **🤖 AI-Powered Recommendations**
    -   Utilizes the **Gemini AI API** to provide instant, personalized advice based on the prediction.
    -   Generates actionable home remedies and a tailored 7-day skincare plan.

-   **👩‍⚕️ Consult Dermatologists**
    -   An integrated feature to find local dermatologists for professional consultation.
    -   Users can enter their city, and the app scrapes **Justdial.com** to provide a list of top-rated specialists with their name, location, and contact number.

---

## 🚀 How to Run Locally

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Tanishq-789/Skin-condition-analyzer.git
    cd Skin-condition-analyzer
    ```

2.  **Create and Activate a Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up API Key**
    -   Create a folder named `.streamlit` in your project directory.
    -   Inside it, create a file named `secrets.toml`.
    -   Add your Gemini API key to the file:
    ```ini
    GEMINI_API_KEY="your_google_gemini_api_key"
    ```

5.  **Run the Streamlit App**
    ```bash
    streamlit run app.py
    ```
    The app will automatically download the required Keras model from Hugging Face on the first run.

---

## 📊 Model Information

-   **Architecture**: **EfficientNetV2B0** (pretrained on ImageNet)
-   **Accuracy**: Achieved **~95.6%** accuracy on the test dataset.
-   **Loss Function**: Sparse Categorical Crossentropy
-   **Training**: Fine-tuned on the augmented skin condition dataset with class weights to handle imbalances.

---

## 📦 Tech Stack

-   **Framework**: Streamlit
-   **ML Framework**: TensorFlow / Keras
-   **AI Service**: Gemini API
-   **Web Scraping**: Selenium, BeautifulSoup4
-   **Deployed Model**: [skin-condition-classifier on Hugging Face](https://huggingface.co/Tanishq77/skin-condition-classifier/tree/main)
-   **Dataset**: [Augmented Skin Conditions Dataset on Kaggle](https://www.kaggle.com/datasets/syedalinaqvi/augmented-skin-conditions-image-dataset)

---

## ✍️ Author

**Tanishq Shinde**
🎓 B.E. Computer Engineering, PICT
🌐 [GitHub](https://github.com/) | [LinkedIn](https://linkedin.com/) | [Hugging Face](https://huggingface.co/)
