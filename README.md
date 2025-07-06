# 🧠 Skin Diagnosis App — AI-Powered Dermatology Assistant

An end-to-end full-stack web application that uses **AI and deep learning** to diagnose **6 common skin conditions** from facial images, suggest personalized skincare routines, analyze product ingredients, and generate professional reports — all with the power of 🧠 **Gemini AI** and 📸 **EfficientNetV2B0**.

---

## 🏗️ Project Structure

```
skin-diagnosis-app/
├── 📁 backend/          # FastAPI backend for ML prediction, Gemini API, and PDF generation
├── 📁 frontend/         # ReactJS frontend with user-friendly UI
├── README.md            # Project overview (this file)
└── .gitignore
```

---

## 🧩 Features

### 🔍 Skin Condition Prediction
- Upload a face image
- Detects one of the following:
  - **Acne**
  - **Carcinoma** 
  - **Eczema**
  - **Keratosis**
  - **Milia**
  - **Rosacea**

### 🤖 AI-Powered Analysis
- Uses **Gemini AI** to:
  - Rate condition severity
  - Analyze skincare product ingredients

### 🧴 Personalized Skincare Plan
- Tailors a **7-day skincare routine** based on:
  - Age
  - Skin type 
  - Condition

### 📄 PDF Report Generation
- Creates downloadable reports with:
  - Prediction results
  - AI explanation
  - Suggested treatment plan

---

## 🔙 Backend (`/backend`)

| File | Description |
|------|-------------|
| `main.py` | FastAPI application entry point |
| `model/skin_model.keras` | Trained Keras model (EfficientNetV2B0) |
| `model/predictor.py` | Image preprocessing and prediction |
| `services/gemini_client.py` | Gemini API integration |
| `utils/pdf_generator.py` | PDF report generation |
| `schemas/request_response.py` | Pydantic models for API schemas |
| `.env` | Environment variables |
| `requirements.txt` | Python dependencies |

---

## 🖼️ Frontend (`/frontend`)

| File | Description |
|------|-------------|
| `public/index.html` | Application root HTML |
| `src/components/` | React components (Upload, Results, etc.) |
| `src/pages/Home.jsx` | Main application page |
| `src/services/api.js` | API service configuration |
| `src/styles/app.css` | Custom styles |
| `.env` | Frontend environment variables |
| `App.jsx` | Root React component |
| `package.json` | Frontend dependencies |

---

## 🚀 How to Run Locally

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` file:
```ini
GEMINI_API_KEY=your_google_gemini_api_key
```

Start backend:
```bash
uvicorn main:app --reload
```

### Frontend Setup

```bash
cd frontend
npm install
```

Create `.env` file:
```ini
VITE_BACKEND_URL=http://localhost:8000
```

Start frontend:
```bash
npm run dev
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict/` | Image upload and prediction |
| POST | `/analyze/ingredients` | Ingredient analysis |
| POST | `/generate-plan` | Skincare plan generation | 
| POST | `/generate-pdf` | PDF report generation |

---

## 📊 Model Information

- **Architecture**: EfficientNetV2B0 (ImageNet pretrained)
- **Accuracy**: ~95.6%
- **Loss Function**: Sparse Categorical Crossentropy
- **Training**: Fine-tuned with class weights

---

## 📦 Tech Stack

- **Frontend**: React + Vite
- **Backend**: FastAPI
- **ML Framework**: TensorFlow/Keras
- **AI Service**: Gemini API
- **PDF Generation**: ReportLab

---

## ✍️ Author

**Tanishq Shinde**  
🎓 B.E. Computer Engineering, PICT  
🌐 [GitHub](https://github.com/) | [LinkedIn](https://linkedin.com/) | [Hugging Face](https://huggingface.co/)
