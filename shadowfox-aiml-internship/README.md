# ShadowFox AIML Internship — Task Submissions

**Program:** Artificial Intelligence & Machine Learning (Machine Learning Engineer) — Virtual Internship
**Group:** G2

This repository contains my completed tasks for the ShadowFox AIML internship: all three levels
(Beginner, Intermediate, Advanced), each with an executed Jupyter notebook, results, and a plain-English explainer.

## 📁 Repository Structure

| Folder | Level | Task | Headline result | Status |
|--------|-------|------|-----------------|--------|
| `Task-1-Beginner/` | Beginner | **Boston House Price Prediction** | Gradient Boosting — **R² = 0.875**, RMSE = 2.24 ($1000s) on the test set | ✅ Complete |
| `Task-2-Intermediate/` | Intermediate | **Car Selling Price Prediction + Flask deployment** | Random Forest — **R² = 0.96**, RMSE = 0.96 lakh; live web app + JSON API | ✅ Complete |
| `Task-3-Advanced-NLP/` | Advanced | **Language Model deployment & analysis** | DistilBERT (sentiment) + DistilGPT-2 (generation) — 3 research questions measured + ethics | ✅ Complete |

Every task folder contains:
- an **executed notebook** with charts, tables and conclusions
- an **`EXPLAIN.md`** — how the project works, concept cheat-sheet, and Q&A (how I presented this work)
- a `README.md` with problem statement, approach and reproduction steps

## 🛠️ Tech Stack
Python · pandas · NumPy · scikit-learn · Matplotlib/Seaborn · Joblib · Flask · Hugging Face Transformers · PyTorch (CPU) · Jupyter

## ⚙️ Setup & Run

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Task 1
cd Task-1-Beginner && jupyter notebook Boston_House_Price_Prediction.ipynb

# Task 2 (model training notebook, then the deployed web app)
cd Task-2-Intermediate && jupyter notebook Car_Price_Prediction.ipynb
cd app && python app.py             # open http://localhost:5000

# Task 3 (models auto-download from Hugging Face on first run)
cd Task-3-Advanced-NLP && jupyter notebook Language_Model_NLP_Project.ipynb
```


