# SehatAI

SehatAI is an AI-powered healthcare platform designed to support doctors throughout the clinical workflow by combining **AI-based disease diagnosis**, an **AI Doctor Assistant**, and **patient management** into a single intelligent system.


## Project Overview

SehatAI helps medical professionals streamline diagnosis and clinical documentation through intelligent automation. The platform integrates multiple deep learning-based diagnostic models for diseases such as Pneumonia, Skin Disease, Stroke Risk, Diabetes, and Kidney Disease (currently in progress). These models provide predictions and confidence scores to assist in medical decision-making.


## ✨ Key Features

- 🧠 **AI Disease Diagnosis**
  - Pneumonia Detection
  - Skin Disease Classification
  - Diabetes Prediction
  - Stroke Risk Prediction
  - Kidney Disease Prediction *(In Progress)*

- 👨‍⚕️ **AI Doctor Assistant**
  - Preliminary diagnosis
  - Disease severity assessment
  - Differential diagnosis
  - Treatment & medication recommendations

- 📋 **AI-Generated Clinical Reports**
  - Editable reports before final approval by doctors

- 👥 **Patient Management System**
  - Patient registration
  - Medical history
  - Prescriptions
  - Follow-up records

- 📁 **Centralized Medical Records**
  - Laboratory reports
  - Previous diagnoses
  - Test history

- ⚡ **Clinical Decision Support**
  - AI-powered insights for faster and more informed decision-making

- 🔒 **Doctor-in-the-Loop Workflow**
  - Doctors review, edit, and approve all AI-generated recommendations


## Technology Stack

| Category | Technologies |
|----------|--------------|
| **Backend** | Python, Flask |
| **Frontend** | HTML, CSS, JavaScript, Bootstrap |
| **AI/ML** | TensorFlow, scikit-learn, Pandas, NumPy, Joblib |
| **LLM** | Google Gemini API |
| **Database** | SQLite |


## 📂 Project Structure

```text
SehatAI/
│
├── backend/
│   ├── app.py
│   ├── chatbot/
│   ├── model/
│   ├── database
│   └── notebook/
    └── ...
│
├── frontend/
│   ├── templates/
│   └── ...
│
├── requirements.txt
└── README.md
```


## Setup and Run Instructions

### 1. Install Dependencies

install the required Python packages:

```bash
pip install -r requirements.txt
```

### 2. Configure Gemini API Key

Before running the application, create or update the environment file located at:

- backend/chatbot/.env

Add your Gemini API key in that file:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```


### 3. Run the Application

From the project root, go to the backend directory and start the app:

```bash
cd backend
python app.py
```

Once the server starts, open your browser and visit:

```text
http://127.0.0.1:5000/
```


## 📌 Notes

- Flask is used as the backend framework.
- Google Gemini powers the AI Doctor Assistant.
- Configure the `.env` file before running the application.
- Kidney Disease Prediction is currently under development.
- AI recommendations are intended to assist healthcare professionals and should always be reviewed by qualified doctors.

## 🔮 Future Enhancements

- Multi-language support (English, Urdu & Sindhi)
- OCR for medical reports and prescriptions
- Telemedicine & video consultations
- Integration with Hospital Information Systems (HIS)
- Drug interaction and allergy checking
- Appointment scheduling & reminders
- Cloud deployment with secure authentication


## 👨‍💻 Developed By

**Tooba Rani**  
BS Artificial Intelligence Student | AI & Healthcare Enthusiast