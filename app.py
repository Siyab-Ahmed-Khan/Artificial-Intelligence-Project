import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import hashlib, json, os, datetime, warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="NutriAI", page_icon="🥗", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');
*, *::before, *::after { box-sizing: border-box; }
html, body, .stApp { font-family: 'DM Sans', sans-serif; background-color: #0f1117; color: #e8e8e8; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 2rem 2rem !important; max-width: 1400px; }
[data-testid="stSidebar"] { background: #161b27 !important; border-right: 1px solid #1e2535; }
[data-testid="stSidebar"] * { color: #c8cfe0 !important; }
.stTextInput input, .stNumberInput input { background: #1a2035 !important; border: 1px solid #2a3350 !important; border-radius: 8px !important; color: #e8e8e8 !important; font-family: 'DM Sans', sans-serif !important; }
.stSelectbox > div > div { background: #1a2035 !important; border: 1px solid #2a3350 !important; border-radius: 8px !important; color: #e8e8e8 !important; }
.stSlider > div > div > div { background: #2a9d6e !important; }
.stButton > button { background: #2a9d6e !important; color: white !important; border: none !important; border-radius: 8px !important; font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important; width: 100% !important; transition: all 0.2s ease !important; }
.stButton > button:hover { background: #238a5e !important; transform: translateY(-1px); }
[data-testid="stMetric"] { background: #161b27; border: 1px solid #1e2535; border-radius: 10px; padding: 0.8rem !important; }
[data-testid="stMetricLabel"] { color: #7a8499 !important; font-size: 0.68rem !important; text-transform: uppercase; letter-spacing: 0.08em; }
[data-testid="stMetricValue"] { color: #e8e8e8 !important; font-size: 1.2rem !important; font-weight: 600 !important; }
[data-testid="stMetricDelta"] { font-size: 0.72rem !important; }
.stTabs [data-baseweb="tab-list"] { background: #161b27; border-radius: 10px; padding: 4px; gap: 4px; border: 1px solid #1e2535; }
.stTabs [data-baseweb="tab"] { background: transparent; color: #7a8499; border-radius: 7px; font-family: 'DM Sans', sans-serif; font-size: 0.85rem; font-weight: 500; border: none; }
.stTabs [aria-selected="true"] { background: #2a9d6e !important; color: white !important; }
.stAlert { border-radius: 10px !important; border: none !important; }
hr { border-color: #1e2535 !important; }

.page-header { padding: 2rem 0 1.2rem 0; border-bottom: 1px solid #1e2535; margin-bottom: 1.5rem; }
.page-header h1 { font-family: 'DM Serif Display', serif; font-size: 2rem; color: #e8e8e8; letter-spacing: -0.02em; margin-bottom: 0.2rem; }
.page-header p { color: #7a8499; font-size: 0.85rem; margin: 0; }
.section-label { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.12em; color: #7a8499; font-weight: 500; margin-bottom: 0.6rem; margin-top: 1.2rem; display: block; }
.brand { font-family: 'DM Serif Display', serif; font-size: 1.5rem; color: #e8e8e8; padding: 1.2rem 0 0.3rem 0; display: block; }
.brand span { color: #2a9d6e; }

.meal-row { display: flex; justify-content: space-between; align-items: center; padding: 0.7rem 0.9rem; background: #1a2035; border-radius: 8px; margin-bottom: 0.4rem; border: 1px solid #1e2535; gap: 0.5rem; }
.meal-row .food-name { font-weight: 500; font-size: 0.88rem; color: #e8e8e8; }
.meal-row .food-meta { font-size: 0.75rem; color: #7a8499; margin-top: 0.15rem; }
.meal-row .food-cal { font-size: 0.85rem; color: #2a9d6e; font-weight: 600; white-space: nowrap; flex-shrink: 0; }

.score-pill { display: inline-block; padding: 0.35rem 1.1rem; border-radius: 100px; font-size: 0.82rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; }
.rec-card { padding: 0.9rem 1.1rem; border-radius: 10px; margin-bottom: 0.6rem; font-size: 0.86rem; line-height: 1.6; border-left: 3px solid; }
.rec-good { background: #0d2018; border-color: #2a9d6e; color: #a8d5be; }
.rec-warn { background: #1e1a0d; border-color: #e0a020; color: #d4c07a; }
.rec-bad  { background: #1e0d0d; border-color: #c94040; color: #d4a0a0; }

.diet-card { background: #161b27; border: 1px solid #1e2535; border-radius: 12px; padding: 1.3rem 1.5rem; margin-bottom: 0.8rem; }
.diet-card.active { border-color: #2a9d6e; }
.diet-card h4 { color: #2a9d6e; font-size: 0.92rem; font-weight: 600; margin-bottom: 0.5rem; margin-top: 0; }
.diet-card .plan-desc { color: #7a8499; font-size: 0.78rem; line-height: 1.5; margin-bottom: 0.8rem; }
.macro-row { display: flex; justify-content: space-between; align-items: center; padding: 0.35rem 0; border-bottom: 1px solid #1e2535; font-size: 0.83rem; }
.macro-row:last-child { border-bottom: none; }
.macro-label { color: #7a8499; }
.macro-val { color: #e8e8e8; font-weight: 500; }

.meal-schedule { background: #0f1117; border-radius: 8px; padding: 0.8rem 1rem; margin-top: 0.6rem; }
.ms-row { display: flex; align-items: center; padding: 0.32rem 0; border-bottom: 1px solid #1e253533; font-size: 0.8rem; gap: 0.5rem; }
.ms-row:last-child { border-bottom: none; }
.ms-time { color: #7a8499; min-width: 78px; }
.ms-food { color: #c8cfe0; flex: 1; }
.ms-cal  { color: #2a9d6e; font-weight: 600; min-width: 58px; text-align: right; }

.qty-info { background: #1a2035; border: 1px solid #2a3350; border-radius: 8px; padding: 0.5rem 0.9rem; font-size: 0.82rem; color: #c8cfe0; margin-bottom: 0.5rem; }
.qty-info span { color: #2a9d6e; font-weight: 600; }

.profile-row { display: flex; align-items: center; gap: 0.3rem; font-size: 0.85rem; margin-bottom: 0.3rem; flex-wrap: wrap; }
.profile-row b { color: #e8e8e8; }
.profile-row .lbl { color: #7a8499; }
</style>
""", unsafe_allow_html=True)

# ─── FOOD DATABASE ────────────────────────────────────────────────────────────
# unit_weight = grams per 1 natural unit (None = no natural counting unit)
FOOD_DB = {
    "Desi": {
        "Roti":           {"Calories":264,"Protein":8.0, "Fat":3.7, "Carbs":53.0,"Fiber":2.7, "unit_weight":40,  "unit_name":"roti"},
        "Naan":           {"Calories":317,"Protein":9.0, "Fat":6.0, "Carbs":57.0,"Fiber":2.0, "unit_weight":90,  "unit_name":"naan"},
        "Paratha":        {"Calories":326,"Protein":7.5, "Fat":14.0,"Carbs":43.0,"Fiber":2.1, "unit_weight":80,  "unit_name":"paratha"},
        "Biryani":        {"Calories":200,"Protein":9.0, "Fat":7.0, "Carbs":28.0,"Fiber":0.8, "unit_weight":None,"unit_name":None},
        "Daal":           {"Calories":116,"Protein":8.0, "Fat":0.4, "Carbs":20.0,"Fiber":7.9, "unit_weight":None,"unit_name":None},
        "Nihari":         {"Calories":180,"Protein":15.0,"Fat":11.0,"Carbs":4.0, "Fiber":0.5, "unit_weight":None,"unit_name":None},
        "Halwa Puri":     {"Calories":380,"Protein":6.0, "Fat":18.0,"Carbs":48.0,"Fiber":1.5, "unit_weight":None,"unit_name":None},
        "Samosa":         {"Calories":262,"Protein":5.0, "Fat":14.0,"Carbs":31.0,"Fiber":2.2, "unit_weight":55,  "unit_name":"samosa"},
        "Lassi":          {"Calories":70, "Protein":3.5, "Fat":3.5, "Carbs":6.5, "Fiber":0.0, "unit_weight":None,"unit_name":None},
        "Chai":           {"Calories":40, "Protein":1.5, "Fat":2.0, "Carbs":4.5, "Fiber":0.0, "unit_weight":None,"unit_name":None},
        "Chicken Karahi": {"Calories":210,"Protein":20.0,"Fat":13.0,"Carbs":3.0, "Fiber":0.5, "unit_weight":None,"unit_name":None},
        "Aloo Gosht":     {"Calories":175,"Protein":13.0,"Fat":10.0,"Carbs":9.0, "Fiber":1.2, "unit_weight":None,"unit_name":None},
        "Saag":           {"Calories":60, "Protein":3.5, "Fat":3.0, "Carbs":6.0, "Fiber":2.5, "unit_weight":None,"unit_name":None},
        "Chana Masala":   {"Calories":164,"Protein":8.9, "Fat":2.6, "Carbs":27.0,"Fiber":7.6, "unit_weight":None,"unit_name":None},
        "Seekh Kebab":    {"Calories":220,"Protein":19.0,"Fat":15.0,"Carbs":2.0, "Fiber":0.5, "unit_weight":70,  "unit_name":"kebab"},
        "Chapli Kebab":   {"Calories":280,"Protein":18.0,"Fat":20.0,"Carbs":5.0, "Fiber":0.5, "unit_weight":80,  "unit_name":"kebab"},
        "Pulao":          {"Calories":180,"Protein":6.0, "Fat":5.0, "Carbs":28.0,"Fiber":0.8, "unit_weight":None,"unit_name":None},
        "Qorma":          {"Calories":250,"Protein":18.0,"Fat":16.0,"Carbs":8.0, "Fiber":0.5, "unit_weight":None,"unit_name":None},
        "Dahi":           {"Calories":60, "Protein":3.5, "Fat":3.2, "Carbs":4.7, "Fiber":0.0, "unit_weight":None,"unit_name":None},
        "Raita":          {"Calories":60, "Protein":3.0, "Fat":3.5, "Carbs":4.5, "Fiber":0.0, "unit_weight":None,"unit_name":None},
        "Kheer":          {"Calories":150,"Protein":4.0, "Fat":5.0, "Carbs":22.0,"Fiber":0.2, "unit_weight":None,"unit_name":None},
        "Paye":           {"Calories":185,"Protein":17.0,"Fat":12.0,"Carbs":2.0, "Fiber":0.0, "unit_weight":None,"unit_name":None},
        "White Rice":     {"Calories":130,"Protein":2.7, "Fat":0.3, "Carbs":28.0,"Fiber":0.4, "unit_weight":None,"unit_name":None},
        "Mutton":         {"Calories":294,"Protein":25.0,"Fat":21.0,"Carbs":0.0, "Fiber":0.0, "unit_weight":None,"unit_name":None},
    },
    "Western": {
        "Grilled Chicken":{"Calories":165,"Protein":31.0,"Fat":3.6, "Carbs":0.0, "Fiber":0.0, "unit_weight":None,"unit_name":None},
        "Burger":         {"Calories":295,"Protein":17.0,"Fat":14.0,"Carbs":24.0,"Fiber":1.5, "unit_weight":200,"unit_name":"burger"},
        "Pizza (slice)":  {"Calories":266,"Protein":11.0,"Fat":10.0,"Carbs":33.0,"Fiber":2.3, "unit_weight":107,"unit_name":"slice"},
        "Caesar Salad":   {"Calories":190,"Protein":7.0, "Fat":14.0,"Carbs":11.0,"Fiber":2.0, "unit_weight":None,"unit_name":None},
        "Pasta":          {"Calories":220,"Protein":8.0, "Fat":5.0, "Carbs":38.0,"Fiber":2.5, "unit_weight":None,"unit_name":None},
        "Steak":          {"Calories":271,"Protein":26.0,"Fat":18.0,"Carbs":0.0, "Fiber":0.0, "unit_weight":None,"unit_name":None},
        "Sandwich":       {"Calories":250,"Protein":12.0,"Fat":9.0, "Carbs":33.0,"Fiber":2.0, "unit_weight":150,"unit_name":"sandwich"},
        "French Fries":   {"Calories":312,"Protein":3.4, "Fat":15.0,"Carbs":41.0,"Fiber":3.8, "unit_weight":None,"unit_name":None},
        "Grilled Salmon": {"Calories":208,"Protein":20.0,"Fat":13.0,"Carbs":0.0, "Fiber":0.0, "unit_weight":None,"unit_name":None},
        "Mac & Cheese":   {"Calories":320,"Protein":12.0,"Fat":13.0,"Carbs":40.0,"Fiber":1.5, "unit_weight":None,"unit_name":None},
        "Omelette":       {"Calories":154,"Protein":11.0,"Fat":11.0,"Carbs":1.6, "Fiber":0.0, "unit_weight":None,"unit_name":None},
        "Chicken Wings":  {"Calories":290,"Protein":27.0,"Fat":19.0,"Carbs":0.0, "Fiber":0.0, "unit_weight":32,  "unit_name":"wing"},
        "Beef Tacos":     {"Calories":226,"Protein":12.0,"Fat":11.0,"Carbs":20.0,"Fiber":3.0, "unit_weight":90,  "unit_name":"taco"},
        "Soup":           {"Calories":75, "Protein":4.0, "Fat":2.0, "Carbs":10.0,"Fiber":1.5, "unit_weight":None,"unit_name":None},
    },
    "Breakfast": {
        "Boiled Eggs":         {"Calories":155,"Protein":13.0,"Fat":11.0,"Carbs":1.1, "Fiber":0.0, "unit_weight":50,  "unit_name":"egg"},
        "Oatmeal":             {"Calories":158,"Protein":6.0, "Fat":3.2, "Carbs":27.0,"Fiber":4.0, "unit_weight":None,"unit_name":None},
        "Pancakes":            {"Calories":227,"Protein":6.0, "Fat":7.0, "Carbs":35.0,"Fiber":1.5, "unit_weight":45,  "unit_name":"pancake"},
        "Milk":                {"Calories":61, "Protein":3.2, "Fat":3.3, "Carbs":4.8, "Fiber":0.0, "unit_weight":None,"unit_name":None},
        "Greek Yogurt":        {"Calories":100,"Protein":17.0,"Fat":0.7, "Carbs":6.0, "Fiber":0.0, "unit_weight":None,"unit_name":None},
        "Banana":              {"Calories":89, "Protein":1.1, "Fat":0.3, "Carbs":23.0,"Fiber":2.6, "unit_weight":118,"unit_name":"banana"},
        "Peanut Butter Toast": {"Calories":270,"Protein":10.0,"Fat":14.0,"Carbs":26.0,"Fiber":3.0, "unit_weight":None,"unit_name":None},
        "Cereal":              {"Calories":150,"Protein":3.0, "Fat":2.0, "Carbs":30.0,"Fiber":2.5, "unit_weight":None,"unit_name":None},
        "Avocado Toast":       {"Calories":240,"Protein":6.0, "Fat":15.0,"Carbs":22.0,"Fiber":6.0, "unit_weight":None,"unit_name":None},
        "Scrambled Eggs":      {"Calories":148,"Protein":10.0,"Fat":11.0,"Carbs":1.5, "Fiber":0.0, "unit_weight":50,  "unit_name":"egg"},
        "Protein Shake":       {"Calories":160,"Protein":30.0,"Fat":3.0, "Carbs":8.0, "Fiber":1.0, "unit_weight":None,"unit_name":None},
        "Whole Wheat Bread":   {"Calories":247,"Protein":13.0,"Fat":4.2, "Carbs":41.0,"Fiber":7.0, "unit_weight":28,  "unit_name":"slice"},
    },
    "Turkish": {
        "Doner Kebab":  {"Calories":250,"Protein":20.0,"Fat":12.0,"Carbs":18.0,"Fiber":1.5, "unit_weight":None,"unit_name":None},
        "Shawarma":     {"Calories":230,"Protein":18.0,"Fat":11.0,"Carbs":16.0,"Fiber":1.2, "unit_weight":None,"unit_name":None},
        "Baklava":      {"Calories":334,"Protein":5.0, "Fat":18.0,"Carbs":40.0,"Fiber":1.5, "unit_weight":30,  "unit_name":"piece"},
        "Turkish Bread":{"Calories":270,"Protein":9.0, "Fat":3.5, "Carbs":52.0,"Fiber":2.0, "unit_weight":None,"unit_name":None},
        "Lentil Soup":  {"Calories":130,"Protein":8.0, "Fat":2.5, "Carbs":20.0,"Fiber":5.0, "unit_weight":None,"unit_name":None},
        "Hummus":       {"Calories":177,"Protein":8.0, "Fat":10.0,"Carbs":16.0,"Fiber":6.0, "unit_weight":None,"unit_name":None},
        "Falafel":      {"Calories":333,"Protein":13.0,"Fat":18.0,"Carbs":32.0,"Fiber":6.0, "unit_weight":17,  "unit_name":"piece"},
        "Ayran":        {"Calories":55, "Protein":3.0, "Fat":2.5, "Carbs":5.0, "Fiber":0.0, "unit_weight":None,"unit_name":None},
        "Pide":         {"Calories":290,"Protein":11.0,"Fat":9.0, "Carbs":42.0,"Fiber":2.0, "unit_weight":None,"unit_name":None},
        "Kofte":        {"Calories":245,"Protein":19.0,"Fat":16.0,"Carbs":5.0, "Fiber":0.5, "unit_weight":45,  "unit_name":"kofte"},
    },
    "Fruits & Vegetables": {
        "Apple":      {"Calories":52, "Protein":0.3,"Fat":0.2,"Carbs":14.0,"Fiber":2.4, "unit_weight":182,"unit_name":"apple"},
        "Orange":     {"Calories":47, "Protein":0.9,"Fat":0.1,"Carbs":12.0,"Fiber":2.4, "unit_weight":131,"unit_name":"orange"},
        "Watermelon": {"Calories":30, "Protein":0.6,"Fat":0.2,"Carbs":7.6, "Fiber":0.4, "unit_weight":None,"unit_name":None},
        "Mango":      {"Calories":60, "Protein":0.8,"Fat":0.4,"Carbs":15.0,"Fiber":1.6, "unit_weight":200,"unit_name":"mango"},
        "Banana":     {"Calories":89, "Protein":1.1,"Fat":0.3,"Carbs":23.0,"Fiber":2.6, "unit_weight":118,"unit_name":"banana"},
        "Spinach":    {"Calories":23, "Protein":2.9,"Fat":0.4,"Carbs":3.6, "Fiber":2.2, "unit_weight":None,"unit_name":None},
        "Broccoli":   {"Calories":34, "Protein":2.8,"Fat":0.4,"Carbs":6.6, "Fiber":2.6, "unit_weight":None,"unit_name":None},
        "Carrot":     {"Calories":41, "Protein":0.9,"Fat":0.2,"Carbs":10.0,"Fiber":2.8, "unit_weight":61,  "unit_name":"carrot"},
        "Potato":     {"Calories":77, "Protein":2.0,"Fat":0.1,"Carbs":17.0,"Fiber":2.2, "unit_weight":150,"unit_name":"potato"},
        "Tomato":     {"Calories":18, "Protein":0.9,"Fat":0.2,"Carbs":3.9, "Fiber":1.2, "unit_weight":123,"unit_name":"tomato"},
    }
}

DAILY  = {"Calories":2000,"Protein":50,"Fat":65,"Carbs":300,"Fiber":25}
UNITS  = {"Calories":"kcal","Protein":"g","Fat":"g","Carbs":"g","Fiber":"g"}
SCORE_COLOR = {'Excellent':'#2a9d6e','Good':'#3b82f6','Poor':'#e0a020','Critical':'#c94040'}
DATA_FILE = "nutriai_users.json"

def load_users():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE) as f: return json.load(f)
    return {}

def save_users(u):
    with open(DATA_FILE,"w") as f: json.dump(u,f,indent=2)

def hash_pw(pw): return hashlib.sha256(pw.encode()).hexdigest()
def get_today(): return datetime.date.today().isoformat()

@st.cache_resource
def build_knn():
    X=[[20,15,20,20,10],[35,25,35,30,20],[50,40,50,45,30],[65,55,65,60,45],
       [75,70,75,72,60],[85,80,85,82,70],[92,88,90,90,80],[100,95,100,98,90],
       [105,100,102,100,95],[110,98,108,105,85],[120,85,125,118,70],
       [135,75,140,130,55],[155,65,160,150,40],[180,55,190,175,25]]
    y=[0,0,1,1,1,2,2,3,3,2,2,1,1,0]
    sc=StandardScaler(); X_sc=sc.fit_transform(X)
    knn=KNeighborsClassifier(n_neighbors=3); knn.fit(X_sc,y)
    return knn,sc

knn_model,scaler=build_knn()

def get_pct(totals): return {k:round((totals[k]/DAILY[k])*100,1) for k in DAILY}

def classify_diet(totals):
    pct=get_pct(totals)
    f=scaler.transform([[pct['Calories'],pct['Protein'],pct['Fat'],pct['Carbs'],pct['Fiber']]])
    return {0:'Critical',1:'Poor',2:'Good',3:'Excellent'}[knn_model.predict(f)[0]]

def calculate_totals(meals):
    t={k:0.0 for k in DAILY}
    for m in meals:
        f=m['grams']/100
        for k in DAILY: t[k]+=m['nutrition'][k]*f
    return {k:round(v,1) for k,v in t.items()}

def bmr_calc(w,h,a,g):
    return 10*w+6.25*h-5*a+(5 if g=="Male" else -161)

def generate_plans(w,h,a,g,goal,act):
    am={"Sedentary":1.2,"Lightly Active":1.375,"Moderately Active":1.55,"Very Active":1.725}
    base=bmr_calc(w,h,a,g)*am[act]
    if goal=="Lose Weight":
        cals=[base-500,base-300,base-700]
        labels=["Moderate Cut","Mild Cut","Aggressive Cut"]
        durs=["8–10 weeks","12–14 weeks","5–6 weeks"]
        descs=["Steady 0.5 kg/week loss. Best for most people.",
               "Gentle 0.3 kg/week. Easier to sustain long-term.",
               "Fast 0.7+ kg/week. Short-term bursts only."]
        schedules=[
            [("7:00 AM","Oatmeal + boiled eggs (2)","~350"),("10:30 AM","Greek yogurt + fruit","~150"),
             ("1:00 PM","Grilled chicken + salad","~350"),("4:30 PM","Apple + handful of nuts","~150"),("7:30 PM","Daal + 1 roti","~350")],
            [("7:30 AM","Scrambled eggs + toast","~300"),("12:30 PM","Soup + sandwich","~320"),
             ("4:00 PM","Lassi","~70"),("7:30 PM","Saag + 1 roti","~250")],
            [("7:00 AM","Protein shake","~160"),("12:00 PM","Salad + grilled salmon","~280"),
             ("6:00 PM","Daal + brown rice","~310")],
        ]
    elif goal=="Gain Muscle":
        cals=[base+300,base+500,base+150]
        labels=["Lean Bulk","Standard Bulk","Micro Bulk"]
        durs=["12 weeks","8–10 weeks","16 weeks"]
        descs=["Minimal fat gain. Ideal for experienced lifters.",
               "Faster muscle gain. Slight fat increase expected.",
               "Very slow, clean gains. Best after a cut phase."]
        schedules=[
            [("7:00 AM","Eggs (3) + whole wheat bread + milk","~450"),("10:30 AM","Protein shake + banana","~280"),
             ("1:00 PM","Chicken Karahi + rice + raita","~550"),("4:30 PM","Greek yogurt + oats","~260"),("7:30 PM","Mutton + roti (2)","~580")],
            [("7:00 AM","Pancakes (3) + boiled eggs (3)","~530"),("10:30 AM","Protein shake + PB toast","~430"),
             ("1:00 PM","Biryani + raita","~500"),("4:30 PM","Fruits + nuts","~200"),("7:30 PM","Steak + pasta","~600")],
            [("7:30 AM","Oatmeal + eggs (2)","~400"),("12:30 PM","Grilled chicken + salad","~360"),
             ("4:00 PM","Protein shake","~160"),("7:30 PM","Daal + rice + dahi","~430")],
        ]
    else:
        cals=[base,base-100,base+100]
        labels=["Maintain","Slight Cut","Slight Surplus"]
        durs=["Ongoing","Gradual","Gradual"]
        descs=["Keep weight stable. Focus on food quality.",
               "Slowly shed small amounts. Barely noticeable change.",
               "Slowly add lean muscle. Minimal fat gain."]
        schedules=[
            [("7:30 AM","Eggs (2) + toast + chai","~320"),("12:30 PM","Chicken + white rice","~450"),
             ("4:00 PM","Fruit + yogurt","~160"),("7:30 PM","Daal + roti (2)","~380")],
            [("7:30 AM","Oatmeal + boiled eggs (2)","~380"),("12:30 PM","Caesar salad + soup","~300"),
             ("4:00 PM","Apple","~50"),("7:30 PM","Saag + 1 roti","~240")],
            [("7:30 AM","Protein shake + banana","~250"),("12:30 PM","Biryani","~350"),
             ("4:00 PM","Lassi + samosa (1)","~320"),("7:30 PM","Qorma + rice","~450")],
        ]
    plans=[]
    for i in range(3):
        c=round(cals[i]); pr=round(w*2); fa=round(c*0.25/9); ca=round((c-pr*4-fa*9)/4)
        plans.append({
            "label":labels[i],"duration":durs[i],"calories":c,
            "protein":pr,"fat":fa,"carbs":ca,
            "description":descs[i],"schedule":schedules[i]
        })
    return plans

def get_recs(totals):
    pct=get_pct(totals); recs=[]
    if pct['Calories']<50: recs.append(('bad','Caloric intake is critically low. You are significantly under-eating today.'))
    elif pct['Calories']<75: recs.append(('warn','Below calorie target. Consider adding another meal or snack.'))
    elif pct['Calories']>140: recs.append(('bad','Caloric surplus. Reduce portion sizes or limit fried and processed foods.'))
    elif pct['Calories']>115: recs.append(('warn','Slightly above calorie goal. Watch your next meal portion.'))
    if pct['Protein']<60: recs.append(('bad','Protein is low. Include eggs, chicken, fish, daal, or dairy.'))
    elif pct['Protein']>160: recs.append(('warn','Protein intake is high. Slightly reduce meat portions.'))
    if pct['Fiber']<50: recs.append(('bad','Low fiber. Add vegetables, fruits, or whole grain roti.'))
    if pct['Fat']>140: recs.append(('warn','High fat. Cut down on ghee, fried foods, and fatty cuts.'))
    if pct['Carbs']>140: recs.append(('warn','High carbohydrates. Reduce rice, bread, or roti portions.'))
    if not recs: recs.append(('good','Your nutritional intake looks well balanced today. Keep it up!'))
    return recs

# ─── SESSION STATE ────────────────────────────────────────────────────────────
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ''
if 'selected_plan' not in st.session_state:
    st.session_state.selected_plan = 0
if 'qty_mode' not in st.session_state:
    st.session_state.qty_mode = 'grams'

users_db=load_users()

# ═══════════════════════════════════════════════════════════════════════════════
# AUTH
# ═══════════════════════════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    _,col_b,_ = st.columns([1,1.4,1])
    with col_b:
        st.markdown('<br><br>', unsafe_allow_html=True)
        st.markdown(
            '<div style="text-align:center;margin-bottom:2rem;">'
            '<span style="font-family:\'DM Serif Display\',serif;font-size:2.6rem;color:#e8e8e8;">'
            'Nutri<span style="color:#2a9d6e;">AI</span></span>'
            '<p style="color:#7a8499;font-size:0.85rem;margin-top:0.3rem;">Personal Nutrition Intelligence</p>'
            '</div>',
            unsafe_allow_html=True
        )
        a1,a2=st.tabs(["  Sign In  ","  Create Account  "])
        with a1:
            lu=st.text_input("Username",key="li_u",placeholder="username")
            lp=st.text_input("Password",type="password",key="li_p",placeholder="password")
            if st.button("Sign In",key="login_btn"):
                if lu in users_db and users_db[lu]['password']==hash_pw(lp):
                    st.session_state.logged_in=True; st.session_state.username=lu; st.rerun()
                else: st.error("Invalid credentials.")
        with a2:
            ru=st.text_input("Username",key="ru",placeholder="choose username")
            rp=st.text_input("Password",type="password",key="rp",placeholder="password")
            c1,c2=st.columns(2)
            with c1:
                ra=st.number_input("Age",10,80,21,key="ra")
                rg=st.selectbox("Gender",["Male","Female"],key="rg")
            with c2:
                rw=st.number_input("Weight (kg)",30,200,70,key="rw")
                rh=st.number_input("Height (cm)",130,220,170,key="rh")
            rgoal=st.selectbox("Goal",["Lose Weight","Gain Muscle","Maintain Weight"],key="rgoal")
            ract=st.selectbox("Activity",["Sedentary","Lightly Active","Moderately Active","Very Active"],key="ract")
            if st.button("Create Account",key="reg_btn"):
                if not ru.strip() or not rp.strip(): st.error("Fill all fields.")
                elif ru in users_db: st.error("Username taken.")
                else:
                    users_db[ru]={
                        "password":hash_pw(rp),"age":ra,"gender":rg,
                        "weight":rw,"height":rh,"goal":rgoal,"activity":ract,"history":{}
                    }
                    save_users(users_db); st.success("Account created. Sign in now.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════
ud=users_db[st.session_state.username]
today=get_today()
if today not in ud['history']: ud['history'][today]={"meals":[],"score":"—"}

# Sidebar
with st.sidebar:
    st.markdown('<span class="brand">Nutri<span>AI</span></span>', unsafe_allow_html=True)
    st.markdown(
        f'<p style="color:#7a8499;font-size:0.82rem;margin-bottom:1rem;">Hello, {st.session_state.username}</p>',
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.markdown('<span class="section-label">Profile</span>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="profile-row"><span class="lbl">Age:</span> <b>{ud["age"]}</b>'
        f' &nbsp;|&nbsp; <b>{ud["gender"]}</b></div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="profile-row"><span class="lbl">Weight:</span> <b>{ud["weight"]} kg</b>'
        f' &nbsp;|&nbsp; <span class="lbl">Height:</span> <b>{ud["height"]} cm</b></div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="profile-row" style="margin-top:0.2rem;"><span class="lbl">Goal:</span>'
        f' <b style="color:#2a9d6e;">{ud["goal"]}</b></div>',
        unsafe_allow_html=True
    )
    st.markdown("---")
    bmi=round(ud['weight']/((ud['height']/100)**2),1)
    bmi_label="Underweight" if bmi<18.5 else "Normal" if bmi<25 else "Overweight" if bmi<30 else "Obese"
    st.markdown('<span class="section-label">Body Stats</span>', unsafe_allow_html=True)
    st.markdown(
        f'<p style="font-size:1.5rem;font-weight:600;font-family:DM Serif Display,serif;margin:0;">'
        f'{bmi} <span style="font-size:0.78rem;color:#7a8499;">BMI — {bmi_label}</span></p>',
        unsafe_allow_html=True
    )
    base_cal=round(bmr_calc(ud['weight'],ud['height'],ud['age'],ud['gender']))
    st.markdown(
        f'<p style="font-size:0.85rem;margin-top:0.4rem;">Base Metabolism: <b>{base_cal} kcal</b></p>',
        unsafe_allow_html=True
    )
    st.markdown("---")
    if st.button("Sign Out"):
        st.session_state.logged_in=False; st.session_state.username=''; st.rerun()

# Header
st.markdown(
    f'<div class="page-header">'
    f'<h1>Nutrition Dashboard</h1>'
    f'<p>{datetime.date.today().strftime("%A, %d %B %Y")}</p>'
    f'</div>',
    unsafe_allow_html=True
)

tab_log,tab_plans,tab_hist=st.tabs(["  Meal Log & Analysis  ","  Diet Plans  ","  History  "])

# ══ TAB 1 ══════════════════════════════════════════════════════════════════════
with tab_log:
    L,R=st.columns([1,1.2],gap="large")
    with L:
        st.markdown('<span class="section-label">Add Food</span>', unsafe_allow_html=True)
        cat=st.selectbox("Category",list(FOOD_DB.keys()),label_visibility="collapsed")
        food=st.selectbox("Food",list(FOOD_DB[cat].keys()),label_visibility="collapsed")
        nutr=FOOD_DB[cat][food]
        has_unit = nutr.get("unit_weight") is not None
        unit_w   = nutr.get("unit_weight") or 100
        unit_name= nutr.get("unit_name") or "piece"

        # ── Input mode toggle ─────────────────────────────────────────────────
        if has_unit:
            tm1, tm2 = st.columns(2)
            with tm1:
                if st.button("📏  By Grams", key="btn_grams"):
                    st.session_state.qty_mode = "grams"; st.rerun()
            with tm2:
                if st.button(f"🔢  By Quantity", key="btn_qty"):
                    st.session_state.qty_mode = "qty"; st.rerun()
            # highlight which mode is active
            active_label = "grams" if st.session_state.qty_mode == "grams" else f"{unit_name}s"
            st.markdown(
                f'<p style="font-size:0.72rem;color:#7a8499;margin-top:-0.3rem;margin-bottom:0.2rem;">'
                f'Mode: <b style="color:#2a9d6e;">{active_label}</b></p>',
                unsafe_allow_html=True
            )
        else:
            st.session_state.qty_mode = "grams"

        # ── Slider ────────────────────────────────────────────────────────────
        if st.session_state.qty_mode == "qty" and has_unit:
            max_qty = max(1, min(20, int(800 / unit_w)))
            qty = st.slider(f"Quantity ({unit_name}s)", 1, max_qty, 1, 1)
            grams = qty * unit_w
            st.markdown(
                f'<div class="qty-info">'
                f'<span>{qty} {unit_name}{"s" if qty > 1 else ""}</span>'
                f' &nbsp;=&nbsp; <span>{grams} g</span>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            grams = st.slider("Grams", 25, 800, 100, 25)

        factor=grams/100
        pc=st.columns(5)
        for i,(lbl,key) in enumerate(zip(['Cal','Pro','Fat','Carb','Fiber'],['Calories','Protein','Fat','Carbs','Fiber'])):
            pc[i].metric(lbl, f"{round(nutr[key]*factor,1)}")

        if st.button("Add to Log"):
            ud['history'][today]['meals'].append({
                "food": food, "category": cat, "grams": grams,
                "nutrition": {k:nutr[k] for k in DAILY},
                "time": datetime.datetime.now().strftime("%H:%M")
            })
            save_users(users_db); st.rerun()

        st.markdown('<span class="section-label" style="margin-top:1.5rem;">Today\'s Log</span>', unsafe_allow_html=True)
        meals=ud['history'][today]['meals']
        if meals:
            for i,m in enumerate(meals):
                cal_m=round(m['nutrition']['Calories']*m['grams']/100,1)
                c1,c2=st.columns([6,1])
                with c1:
                    st.markdown(
                        f'<div class="meal-row">'
                        f'  <div>'
                        f'    <div class="food-name">{m["food"]}</div>'
                        f'    <div class="food-meta">{m["grams"]} g &nbsp;·&nbsp; {m.get("time","")} &nbsp;·&nbsp; {m["category"]}</div>'
                        f'  </div>'
                        f'  <div class="food-cal">{cal_m} kcal</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                with c2:
                    if st.button("✕", key=f"rm{i}"):
                        ud['history'][today]['meals'].pop(i); save_users(users_db); st.rerun()
            if st.button("Clear All"):
                ud['history'][today]['meals']=[]; save_users(users_db); st.rerun()
        else:
            st.markdown('<p style="color:#7a8499;font-size:0.85rem;">No food logged yet.</p>', unsafe_allow_html=True)

    with R:
        meals=ud['history'][today]['meals']
        if meals:
            totals=calculate_totals(meals); score=classify_diet(totals); pct=get_pct(totals); recs=get_recs(totals)
            col_sc=SCORE_COLOR[score]
            ud['history'][today]['score']=score; save_users(users_db)

            st.markdown(
                f'<div style="margin-bottom:1.2rem;">'
                f'  <p style="font-size:0.68rem;text-transform:uppercase;letter-spacing:0.1em;color:#7a8499;margin-bottom:0.3rem;">Diet Score</p>'
                f'  <span class="score-pill" style="background:{col_sc}25;color:{col_sc};border:1px solid {col_sc}50;">{score}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

            mc=st.columns(5)
            for i,key in enumerate(['Calories','Protein','Fat','Carbs','Fiber']):
                mc[i].metric(key,f"{totals[key]}{UNITS[key]}",f"{pct[key]}%")

            st.markdown('<span class="section-label" style="margin-top:1.2rem;">Nutrient Coverage</span>', unsafe_allow_html=True)
            fig,ax=plt.subplots(figsize=(6,2.6))
            fig.patch.set_facecolor('#161b27'); ax.set_facecolor('#161b27')
            ks=['Calories','Protein','Fat','Carbs','Fiber']
            vs=[pct[k] for k in ks]
            bc=['#2a9d6e' if 60<=v<=120 else '#e0a020' if 40<=v<60 or 120<v<=150 else '#c94040' for v in vs]
            bars=ax.barh(ks,vs,color=bc,height=0.45,edgecolor='none')
            ax.axvline(100,color='#3b82f6',linestyle='--',linewidth=1.2,alpha=0.6)
            ax.set_xlim(0,max(max(vs)+25,130))
            ax.tick_params(colors='#7a8499',labelsize=8.5)
            ax.spines[['top','right','bottom','left']].set_visible(False)
            ax.xaxis.set_tick_params(length=0); ax.yaxis.set_tick_params(length=0)
            for bar,val in zip(bars,vs):
                ax.text(bar.get_width()+1.5,bar.get_y()+bar.get_height()/2,f'{val}%',va='center',fontsize=8,color='#7a8499')
            plt.tight_layout(pad=0.3); st.pyplot(fig,use_container_width=True); plt.close()

            st.markdown('<span class="section-label">Calorie Sources</span>', unsafe_allow_html=True)
            fig2,ax2=plt.subplots(figsize=(4,2.5))
            fig2.patch.set_facecolor('#161b27'); ax2.set_facecolor('#161b27')
            pv=[totals['Protein']*4,totals['Fat']*9,totals['Carbs']*4]
            if sum(pv)>0:
                _w,_t,_a=ax2.pie(
                    pv,labels=['Protein','Fat','Carbs'],
                    colors=['#3b82f6','#c94040','#e0a020'],
                    autopct='%1.0f%%',startangle=90,
                    textprops={'color':'#c8cfe0','fontsize':8.5},
                    wedgeprops={'linewidth':0}
                )
                for at in _a: at.set_color('#e8e8e8'); at.set_fontsize(8)
            plt.tight_layout(pad=0.2); st.pyplot(fig2,use_container_width=True); plt.close()

            st.markdown('<span class="section-label">Recommendations</span>', unsafe_allow_html=True)
            for kind,rec in recs:
                css='rec-good' if kind=='good' else 'rec-warn' if kind=='warn' else 'rec-bad'
                st.markdown(f'<div class="rec-card {css}">{rec}</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div style="display:flex;justify-content:center;align-items:center;height:320px;'
                'background:#161b27;border-radius:12px;border:1px solid #1e2535;">'
                '<p style="color:#7a8499;font-size:0.88rem;text-align:center;">'
                'Add food from the left panel<br>to view your analysis.</p></div>',
                unsafe_allow_html=True
            )

# ══ TAB 2 ══════════════════════════════════════════════════════════════════════
with tab_plans:
    st.markdown('<span class="section-label" style="margin-top:0.5rem;">Personalised Diet Plans</span>', unsafe_allow_html=True)
    st.markdown(
        f'<p style="color:#7a8499;font-size:0.85rem;margin-bottom:1rem;">'
        f'Goal: <b style="color:#2a9d6e;">{ud["goal"]}</b> &nbsp;·&nbsp; '
        f'Activity: <b style="color:#e8e8e8;">{ud["activity"]}</b></p>',
        unsafe_allow_html=True
    )

    plans=generate_plans(ud['weight'],ud['height'],ud['age'],ud['gender'],ud['goal'],ud['activity'])

    # ── Plan selector row ──────────────────────────────────────────────────────
    sel_c=st.columns(3)
    for i,(col,plan) in enumerate(zip(sel_c,plans)):
        with col:
            btn_label=f"{'✓  ' if st.session_state.selected_plan==i else ''}{plan['label']}"
            if st.button(btn_label, key=f"plan_btn_{i}"):
                st.session_state.selected_plan=i; st.rerun()

    st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)

    # ── Plan overview cards ────────────────────────────────────────────────────
    pc=st.columns(3)
    for i,(col,plan) in enumerate(zip(pc,plans)):
        with col:
            is_sel=(i==st.session_state.selected_plan)
            border="border:1px solid #2a9d6e;" if is_sel else "border:1px solid #1e2535;"
            badge='&nbsp;<span style="font-size:0.62rem;background:#2a9d6e22;color:#2a9d6e;border-radius:4px;padding:0.1rem 0.4rem;letter-spacing:0.03em;">Selected</span>' if is_sel else ''
            st.markdown(
                f'<div class="diet-card {"active" if is_sel else ""}" style="{border}">'
                f'  <h4>{plan["label"]}{badge}</h4>'
                f'  <p class="plan-desc">{plan["description"]}</p>'
                f'  <div class="macro-row"><span class="macro-label">Duration</span><span class="macro-val">{plan["duration"]}</span></div>'
                f'  <div class="macro-row"><span class="macro-label">Calories</span><span class="macro-val">{plan["calories"]} kcal</span></div>'
                f'  <div class="macro-row"><span class="macro-label">Protein</span><span class="macro-val">{plan["protein"]} g</span></div>'
                f'  <div class="macro-row"><span class="macro-label">Fat</span><span class="macro-val">{plan["fat"]} g</span></div>'
                f'  <div class="macro-row"><span class="macro-label">Carbs</span><span class="macro-val">{plan["carbs"]} g</span></div>'
                f'</div>',
                unsafe_allow_html=True
            )

    # ── Selected plan detail ───────────────────────────────────────────────────
    sel=plans[st.session_state.selected_plan]
    st.markdown("---")
    st.markdown(
        f'<p style="font-size:0.68rem;text-transform:uppercase;letter-spacing:0.1em;color:#7a8499;margin-bottom:0.5rem;">'
        f'Sample Meal Schedule — {sel["label"]}</p>',
        unsafe_allow_html=True
    )
    sched_html='<div class="meal-schedule">'
    for time_s,food_s,cal_s in sel["schedule"]:
        sched_html+=(
            f'<div class="ms-row">'
            f'  <span class="ms-time">{time_s}</span>'
            f'  <span class="ms-food">{food_s}</span>'
            f'  <span class="ms-cal">{cal_s} kcal</span>'
            f'</div>'
        )
    sched_html+='</div>'
    st.markdown(sched_html, unsafe_allow_html=True)

    # ── Macro bar for selected plan ────────────────────────────────────────────
    st.markdown('<span class="section-label" style="margin-top:1rem;">Macro Targets</span>', unsafe_allow_html=True)
    fig3,ax3=plt.subplots(figsize=(6,1.6))
    fig3.patch.set_facecolor('#161b27'); ax3.set_facecolor('#161b27')
    bars3=ax3.barh(['Protein','Fat','Carbs'],[sel['protein'],sel['fat'],sel['carbs']],
                   color=['#3b82f6','#c94040','#e0a020'],height=0.4,edgecolor='none')
    ax3.tick_params(colors='#7a8499',labelsize=8.5)
    ax3.spines[['top','right','bottom','left']].set_visible(False)
    ax3.xaxis.set_tick_params(length=0); ax3.yaxis.set_tick_params(length=0)
    for bar,val in zip(bars3,[sel['protein'],sel['fat'],sel['carbs']]):
        ax3.text(bar.get_width()+1,bar.get_y()+bar.get_height()/2,f'{val} g',va='center',fontsize=8,color='#7a8499')
    plt.tight_layout(pad=0.3); st.pyplot(fig3,use_container_width=True); plt.close()

    st.markdown("---")
    st.markdown('<span class="section-label">Update Profile</span>', unsafe_allow_html=True)
    uc1,uc2,uc3,uc4=st.columns(4)
    with uc1: nw=st.number_input("Weight (kg)",30,200,ud['weight'])
    with uc2: nh=st.number_input("Height (cm)",130,220,ud['height'])
    with uc3: ng=st.selectbox("Goal",["Lose Weight","Gain Muscle","Maintain Weight"],
                               index=["Lose Weight","Gain Muscle","Maintain Weight"].index(ud['goal']))
    with uc4: na=st.selectbox("Activity",["Sedentary","Lightly Active","Moderately Active","Very Active"],
                               index=["Sedentary","Lightly Active","Moderately Active","Very Active"].index(ud['activity']))
    if st.button("Update Profile"):
        ud['weight']=nw; ud['height']=nh; ud['goal']=ng; ud['activity']=na
        save_users(users_db); st.success("Profile updated."); st.rerun()

# ══ TAB 3 ══════════════════════════════════════════════════════════════════════
with tab_hist:
    st.markdown('<span class="section-label" style="margin-top:0.5rem;">Tracking History</span>', unsafe_allow_html=True)
    hist=ud.get('history',{})
    days=sorted(hist.keys(),reverse=True)
    if days:
        for day in days:
            dm=hist[day]['meals']; ds=hist[day].get('score','—')
            sc=SCORE_COLOR.get(ds,'#7a8499')
            dc=round(sum(m['nutrition']['Calories']*m['grams']/100 for m in dm),1)
            dlabel="Today" if day==today else day
            with st.expander(f"{dlabel}   —   {len(dm)} meals   ·   {dc} kcal   ·   Score: {ds}"):
                if dm:
                    h1,h2,h3,h4=st.columns([3,1,1,1])
                    for h,t in zip([h1,h2,h3,h4],['FOOD','GRAMS','CALORIES','PROTEIN']):
                        h.markdown(f'<span style="color:#7a8499;font-size:0.72rem;font-weight:600;">{t}</span>', unsafe_allow_html=True)
                    for m in dm:
                        cm=round(m['nutrition']['Calories']*m['grams']/100,1)
                        pm=round(m['nutrition']['Protein']*m['grams']/100,1)
                        r1,r2,r3,r4=st.columns([3,1,1,1])
                        r1.markdown(f'<span style="font-size:0.85rem;">{m["food"]}</span>', unsafe_allow_html=True)
                        r2.markdown(f'<span style="font-size:0.85rem;color:#7a8499;">{m["grams"]} g</span>', unsafe_allow_html=True)
                        r3.markdown(f'<span style="font-size:0.85rem;color:#2a9d6e;">{cm}</span>', unsafe_allow_html=True)
                        r4.markdown(f'<span style="font-size:0.85rem;color:#3b82f6;">{pm} g</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span style="color:#7a8499;font-size:0.85rem;">No meals logged.</span>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:#7a8499;font-size:0.85rem;">No history yet.</p>', unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#3a4155;font-size:0.76rem;letter-spacing:0.04em;padding-bottom:1rem;">'
    'NutriAI &nbsp;·&nbsp; Siyab Ahmed Khan (SAP: 55051) &nbsp;·&nbsp; Shafay Khan (SAP: 44632) &nbsp;·&nbsp; Riphah International University'
    '</div>',
    unsafe_allow_html=True
)
