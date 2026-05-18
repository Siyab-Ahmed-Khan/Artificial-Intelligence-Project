# AI Nutritionist — Setup & Run Guide

## Step 1 — Install Libraries
Open Terminal and run:
```
pip install streamlit pandas numpy scikit-learn matplotlib fuzzywuzzy python-Levenshtein
```

## Step 2 — Download Dataset (Optional but recommended)
Go to: https://www.kaggle.com/datasets/utsavdey1410/food-nutrition-dataset
Download the CSV, rename it to: food_data.csv
Place it in this same folder.

The app works WITHOUT the Kaggle dataset too (uses built-in Pakistani foods).

## Step 3 — Run the App
Open Terminal, navigate to this folder:
```
cd path/to/ai_nutritionist
streamlit run app.py
```

Browser opens automatically at: http://localhost:8501

## Files in This Project
- app.py              → Main application (only file you need to run)
- pakistani_foods.csv → 40 desi + common foods with nutrition data
- food_data.csv       → (Optional) Kaggle dataset for more foods
- requirements.txt    → All required libraries

## How to Use
1. Type a food name in the search box
2. Select the correct food from dropdown
3. Set the grams using the slider
4. Click Add to Meal Log
5. See your full nutrition analysis on the right

## Project Info
Student: Siyab Ahmed Khan
SAP ID: 55051
University: Riphah International University, Islamabad
Course: Artificial Intelligence Lab
