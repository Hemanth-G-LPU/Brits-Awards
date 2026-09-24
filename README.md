# 🏆 BRIT Awards Trends — Analysis & Forecasting (1982–2025)

![Stars](https://img.shields.io/github/stars/<your-username>/<your-repo-name>?style=social) ![last commit](https://img.shields.io/badge/last%20commit-september%202026-blue)

An end-to-end **data analysis and machine learning** project exploring 43 years of BRIT Awards history. Covers data cleaning, exploratory analysis, trend visualization, and classification modeling using **Python**, **Pandas**, and **Scikit-learn**.

---

## 🖼️ Preview

```
Dataset loaded successfully.
Shape: 649 rows × 9 columns
Year Range: 1982 to 2025

Random Forest Accuracy: 91.54%
Logistic Regression Accuracy: 86.15%
```

---

## 🚀 Features

- ✅ Full data cleaning pipeline (missing values, duplicates, type fixes)
- ✅ Category normalization into broad, readable groups
- ✅ 10+ visualizations: trends, top artists, venues, hosts, category mix
- ✅ Decade-by-decade trend and correlation analysis
- ✅ Feature engineering (`decade`, `category_group`, `winner_total_wins`)
- ✅ Machine learning: Logistic Regression vs Random Forest classification
- ✅ Confusion matrices & feature importance plots
- ✅ Available as both a Jupyter Notebook and a standalone Python script

---

## 🛠️ Technologies Used

- Python 3.10+
- Pandas & NumPy
- Matplotlib & Seaborn
- Scikit-learn (Logistic Regression, Random Forest, Label Encoding)
- Jupyter Notebook

---

## 🌍 Compatibility

🧭 Runs on any platform with Python 3.8+ installed: **Windows**, **macOS**, **Linux**

---

## 📁 Project Structure

```
BRIT-Awards-Analysis
│
├── brit_awards_analysis.ipynb    # Full notebook — EDA, cleaning, viz, ML
├── brit_awards_analysis.py       # Standalone script version
├── brit_awards.csv               # Dataset (649 rows × 9 columns)
└── README.md                     # Project documentation
```

---

## 📦 How to Use

1. **Clone the repository:**

   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. **Install dependencies:**

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

3. **Run the notebook:**

   ```bash
   jupyter notebook brit_awards_analysis.ipynb
   ```

   Or run the script version directly:

   ```bash
   python brit_awards_analysis.py
   ```

---

## 📊 Key Findings

| Insight | Finding |
|---|---|
| 🏆 Most Decorated Artist | U2 / Robbie Williams (varies by grouping) |
| 📅 Year Range | 1982 – 2025 (43 years) |
| 📍 Most Common Venue | The O2, London |
| 🎤 Most Frequent Host | Chris Evans |
| 📉 Smallest Ceremony | 2020 (COVID-19) |
| 🤖 Best ML Model | Random Forest |

---

## ✨ Future Enhancements (Ideas)

- 🔮 Predict future BRIT Award winners using external music chart data
- 💬 Sentiment analysis of award category descriptions over time
- 🕸️ Network analysis of artists winning multiple categories in the same year
- 📈 Interactive dashboard (Streamlit / Plotly Dash)

---

## 🙌 Credits

- 💡 Developed by https://github.com/Hemanth-G-LPU
- 📊 Dataset: Official BRIT Awards Historical Records

---

## 📄 License

This project is open-source and free to use for personal or educational purposes.
