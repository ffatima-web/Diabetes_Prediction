# 🩺 Diabetes Prediction System

Predict diabetes risk in seconds using Machine Learning — clean UI, instant results, zero guesswork.

### 🔗 [Try it live here!](https://diabetesprediction-4xhsdhwh6lyzhmnpybxy7v.streamlit.app/)

---

## ✨ What It Does

Enter a few basic health metrics — glucose, BMI, blood pressure, age, and more — and get an instant, ML-powered prediction on diabetes risk, complete with a probability score and personalized health tips. 💡

## 🚀 Features

- 🔮 **Instant Predictions** — powered by a trained Scikit-learn model
- 📊 **Probability Score** — not just yes/no, but how confident the model is
- 🥗 **Personalized Tips** — diet and lifestyle suggestions based on your result
- 🌗 **Light & Dark Mode** — toggle to match your vibe
- 📱 **Responsive Design** — clean, card-based UI that works everywhere

## 🛠️ Tech Stack

| Layer | Tech |
|---|---|
| 🖥️ App/Frontend | Streamlit |
| 🧠 Machine Learning | Scikit-learn |
| 🐼 Data Handling | Pandas |
| 💾 Model Storage | Joblib |

## 📂 Project Structure

```
Diabetes_ML_project/
├── 🧠 app_professional.py       → Main Streamlit app
├── 📊 diabetes.csv              → Training dataset
├── 📓 Diabetes_Prediction.ipynb → Model training notebook
├── 🎯 diabetes_model.pkl        → Trained ML model
├── 🧩 diabetes_features.pkl     → Feature order used by the model
├── ⚖️ diabetes_scaler.pkl       → Feature scaler
├── 📦 requirements.txt          → Python dependencies
└── 📘 README.md
```

## 🏃 Run It Locally

```bash
# 1️⃣ Clone the repo
git clone https://github.com/ffatima-web/Diabetes_Prediction.git
cd Diabetes_Prediction

# 2️⃣ Create a virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Launch the app
streamlit run app_professional.py
```

## ⚠️ Disclaimer

This project is built for **educational and demonstration purposes only**. It is **not** a medical diagnostic tool. Always consult a qualified healthcare professional for medical advice. 💛

## 👩‍💻 Author

**Faiza Fatima**
🎓 B.Tech Information Technology, Shadan Women's College Of Engineering And Technology

---

⭐ If you found this project interesting, consider giving it a star!
