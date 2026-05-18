# NutriAI: An AI-Powered Personal Nutrition System

NutriAI is a locally deployed, machine-learning-powered personal nutrition analysis system built entirely in Python using Streamlit. It allows users to log meals, receive comprehensive nutritional breakdowns, and get an AI-generated diet quality score based on K-Nearest Neighbours (KNN) classification.

## Features
- **Extensive Food Database:** Dynamically loads 3,000 unique food items from a CSV dataset, featuring a massive variety of global and authentic Pakistani/Desi meals.
- **Diet Quality Classification:** Uses a KNN model (k=3) with `StandardScaler` to classify daily intake into Excellent, Good, Poor, or Critical based on Daily Recommended Values.
- **BMR-Based Diet Plans:** Automatically generates three personalized meal plans based on your weight, height, age, gender, and fitness goals.
- **Multi-User System:** Local JSON-based authentication and persistent meal history tracking.
- **Data Visualisation:** Real-time Matplotlib rendering for macro pie charts and nutrient bar charts.

## Project Structure
```text
├── app.py                    # Main application: UI, ML logic, and data management
├── dataset_description.ipynb # Explanatory notebook of data and feature engineering
├── pakistani_foods.csv       # Primary dataset containing 3,000 food items
├── nutriai_users.json        # Auto-generated runtime storage for user accounts
├── requirements.txt          # Python dependency list
└── README.md                 # Setup and execution instructions
