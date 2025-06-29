"""
🕒 Trendyol Akıllı Saat: Web-Tabanlı Karar Destek Sistemi (Streamlit)
===================================================================
Bu dosya, veri oluşturma, model eğitimi, tahmin ve **modern Streamlit arayüzünü** tek bir
Python betiğinde toplar.

Kurulum
-------
```bash
pip install streamlit pandas numpy scikit-learn joblib openpyxl altair
# Gerçek Trendyol scraping isterseniz ekleyin:
pip install selenium webdriver-manager
```

Çalıştırma
----------
```bash
# CLI modları:
python trendyol_watch.py mock      # sahte veri üret
python trendyol_watch.py train     # modeli eğit
python trendyol_watch.py predict Samsung 4.4 350  # fiyat tahmini (CLI)

# Web arayüzü:
streamlit run trendyol_watch.py
```

Arayüz
------
* **Veri** – CSV yükle veya tek tıkla sentetik veri oluştur, grafiklerle veri dağılımı incele.
* **Model Eğit** – Met­rik kutusunda MAE, RMSE, MAPE ve R² sonuçları.
* **Tahmin** – Marka, puan, yorum sayısı → anlık fiyat tahmini.
* Koyu tema uyumlu; kutular #1e1e1e, metin #fafafa.
"""

from __future__ import annotations
import sys, random, os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import joblib
import altair as alt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -------------------------------
# 0) Streamlit Mod Tespiti
# -------------------------------
WEB_MODE = False
try:
    import streamlit as st
    if hasattr(st, 'runtime') and callable(getattr(st.runtime, 'exists', None)):
        WEB_MODE = st.runtime.exists()
    else:
        WEB_MODE = bool(getattr(st, '_is_running_with_streamlit', False)) or bool(os.getenv('STREAMLIT_SERVER_PORT'))
except ImportError:
    pass

# -------------------------------
# Sabitler
# -------------------------------
DEF_CSV = 'trendyol_akilli_saat.csv'
MODEL_FILE = 'model.joblib'

# -------------------------------
# 1) Sahte Veri Üretici
# -------------------------------
def generate_mock_data(n_samples: int = 300) -> pd.DataFrame:
    brands = [
        'Apple', 'Samsung', 'Xiaomi', 'Huawei', 'Amazfit', 'Garmin',
        'Fossil', 'Fitbit', 'Realme', 'Haylou', 'Lenovo', 'TicWatch'
    ]
    base_price = {
        'Apple':9000, 'Samsung':6000, 'Garmin':7000, 'Huawei':5000,
        'Xiaomi':3000, 'Amazfit':2500, 'Fossil':4000, 'Fitbit':3500,
        'Realme':2200, 'Haylou':1800, 'Lenovo':2000, 'TicWatch':3200
    }
    rows: List[dict] = []
    for _ in range(n_samples):
        b = random.choice(brands)
        rows.append({
            'Marka': b,
            'Ortalama Puanı': round(np.clip(np.random.normal(4.2, 0.4), 1, 5), 1),
            'Yorum Sayısı': int(max(0, np.random.normal(500, 600))),
            'Satış Fiyatı (TL)': round(max(600, np.random.normal(base_price[b], base_price[b]*0.15)), 2)
        })
    df = pd.DataFrame(rows)
    df.to_csv(DEF_CSV, index=False, encoding='utf-8-sig')
    return df

# -------------------------------
# 2) ML Pipeline
# -------------------------------
def build_pipeline() -> Pipeline:
    num_feats = ['Ortalama Puanı', 'Yorum Sayısı']
    cat_feats = ['Marka']
    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), num_feats),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_feats)
    ])
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    return Pipeline([('pre', preprocessor), ('model', model)])


def train_model(df: pd.DataFrame):
    X = df[['Marka', 'Ortalama Puanı', 'Yorum Sayısı']]
    y = df['Satış Fiyatı (TL)']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
    pipe = build_pipeline()
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    metrics = {
        'MAE': mean_absolute_error(y_te, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_te, y_pred)),
        'MAPE': mean_absolute_percentage_error(y_te, y_pred)*100,
        'R2': r2_score(y_te, y_pred)
    }
    joblib.dump(pipe, MODEL_FILE)
    return metrics, pipe


def load_model():
    return joblib.load(MODEL_FILE) if Path(MODEL_FILE).exists() else None

# -------------------------------
# 3) Streamlit UI
# -------------------------------
if WEB_MODE:
    import streamlit as st
    st.set_page_config(page_title='Akıllı Saat Fiyat Tahmini', page_icon='⌚', layout='centered')
    st.markdown('''<style>
        body {background:#0e1117; color:#fafafa;}
        .stButton>button {border-radius:8px; padding:0.55rem 1.1rem;}
        .metrics-box{background:#1e1e1e; color:#fafafa; padding:1rem; border-radius:10px;
                     border:1px solid #333; box-shadow:0 2px 6px rgba(0,0,0,0.4);}
    </style>''', unsafe_allow_html=True)
    st.title('⌚ Akıllı Saat Fiyat Tahmin')
    tab1, tab2, tab3 = st.tabs(['Veri', 'Model Eğit', 'Tahmin'])

    # ----- Veri Sekmesi -----
    with tab1:
        st.header('📦 Veri Hazırlığı')
        if st.button('➕ Sahte Veri Üret'):
            df = generate_mock_data()
            st.success('Sahte veri oluşturuldu.')
            st.dataframe(df.head())
            # Grafikler
            st.subheader('Marka Dağılımı')
            counts = df['Marka'].value_counts().reset_index()
            counts.columns = ['Marka', 'Count']
            bar = alt.Chart(counts).mark_bar().encode(
                x=alt.X('Marka:N', sort='-y'),
                y=alt.Y('Count:Q'),
                tooltip=['Marka','Count']
            )
            st.altair_chart(bar, use_container_width=True)
            st.subheader('Fiyat Dağılımı')
            hist = alt.Chart(df).mark_bar().encode(
                x=alt.X('Satış Fiyatı (TL):Q', bin=alt.Bin(maxbins=30)),
                y='count()',
                tooltip=['count()']
            )
            st.altair_chart(hist, use_container_width=True)
        up = st.file_uploader('CSV Yükle', type=['csv'])
        if up:
            df = pd.read_csv(up)
            df.to_csv(DEF_CSV, index=False)
            st.success('CSV kaydedildi.')
            st.dataframe(df.head())
            # Grafikler
            st.subheader('Marka Dağılımı')
            counts = df['Marka'].value_counts().reset_index()
            counts.columns = ['Marka', 'Count']
            bar = alt.Chart(counts).mark_bar().encode(
                x=alt.X('Marka:N', sort='-y'),
                y=alt.Y('Count:Q'),
                tooltip=['Marka','Count']
            )
            st.altair_chart(bar, use_container_width=True)
            st.subheader('Fiyat Dağılımı')
            hist = alt.Chart(df).mark_bar().encode(
                x=alt.X('Satış Fiyatı (TL):Q', bin=alt.Bin(maxbins=30)),
                y='count()',
                tooltip=['count()']
            )
            st.altair_chart(hist, use_container_width=True)

    # ----- Model Eğit Sekmesi -----
    with tab2:
        st.header('🤖 Model Eğit')
        if not Path(DEF_CSV).exists():
            st.info('Önce veri hazırlayın.')
        elif st.button('🔄 Eğit'):
            data = pd.read_csv(DEF_CSV)
            m, _ = train_model(data)
            st.markdown(f"""
                <div class='metrics-box'>
                <strong>MAE:</strong> {m['MAE']:.0f} TL &nbsp;|&nbsp;
                <strong>RMSE:</strong> {m['RMSE']:.0f} TL &nbsp;|&nbsp;
                <strong>MAPE:</strong> {m['MAPE']:.1f}% &nbsp;|&nbsp;
                <strong>R²:</strong> {m['R2']:.3f}
                </div>
            """, unsafe_allow_html=True)
            st.success('Model kaydedildi.')

    # ----- Tahmin Sekmesi -----
    with tab3:
        st.header('🔮 Tahmin')
        model = load_model()
        if not model:
            st.info('Önce modeli eğitin.')
        else:
            b = st.selectbox('Marka', ['Apple','Samsung','Xiaomi','Huawei','Amazfit','Garmin','Fossil','Fitbit','Realme','Haylou','Lenovo','TicWatch'])
            r = st.slider('Puan',1.0,5.0,4.5,0.1)
            c = st.number_input('Yorum Sayısı',0,10000,350)
            if st.button('🎯 Tahmin Et'):
                dfp = pd.DataFrame([[b,r,c]],columns=['Marka','Ortalama Puanı','Yorum Sayısı'])
                p = load_model().predict(dfp)[0]
                st.subheader(f'💰 Tahmini Fiyat: **{p:,.0f} TL**')

# -------------------------------
# 4) CLI Giriş
# -------------------------------
if not WEB_MODE and __name__=='__main__':
    if len(sys.argv)<2:
        print('Usage: python trendyol_watch.py [mock|train|predict <Marka> <Puan> <Yorum>]')
        sys.exit(0)
    cmd=sys.argv[1].lower()
    if cmd=='mock':
        print(generate_mock_data().head())
    elif cmd=='train':
        if not Path(DEF_CSV).exists(): sys.exit('CSV missing')
        print('Metrics:', train_model(pd.read_csv(DEF_CSV))[0])
    elif cmd=='predict' and len(sys.argv)==5:
        mdl=load_model()
        dfp=pd.DataFrame([[sys.argv[2],float(sys.argv[3]),int(sys.argv[4])]],columns=['Marka','Ortalama Puanı','Yorum Sayısı'])
        print(f'Tahmin: {mdl.predict(dfp)[0]:,.2f} TL')
    else:
        print('Invalid command')
