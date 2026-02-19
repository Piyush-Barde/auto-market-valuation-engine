# Auto Market Valuation Engine 🚗📊

An AI-powered system that analyzes real-world used car listings to estimate fair market value, model depreciation effects, and classify listings as **GOOD**, **FAIR**, or **BAD** deals using machine learning.

---

## Problem Statement

Used car prices are often inconsistent, inflated, or misleading.  
This project aims to bring **data-driven transparency** to used car pricing by:

- Cleaning and analyzing real-world car listing data
- Modeling depreciation trends
- Predicting fair market value using regression models
- Comparing listed price vs predicted price to evaluate deal quality

---

## Core Features

- **Data preprocessing & feature engineering**
- **Depreciation-aware price modeling**
- **Regression-based price prediction**
- **Deal classification**:
  - GOOD Deal → Listed price ≪ Predicted value
  - FAIR Deal → Listed price ≈ Predicted value
  - BAD Deal → Listed price ≫ Predicted value
- **Exploratory Data Analysis (EDA)** with visual insights

---

## Tech Stack

- **Language**: Python
- **Libraries**:
  - Pandas, NumPy
  - Scikit-learn
  - Matplotlib, Seaborn
- **Environment**: Jupyter Notebook + Python scripts

---

## Dataset

The dataset is **not included** in the repository.

Place the used car dataset (e.g. `cardekho.csv`) inside the `data/` directory before running the project.
