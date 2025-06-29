# 🕒 Trendyol Akıllı Saat: Web-Tabanlı Karar Destek Sistemi

Akıllı saatlerin fiyat tahminini yapan, **veri üretimi**, **makine öğrenimi modeli** ve **modern Streamlit arayüzünü** bir araya getiren bir sistem.

> 🚀 Proje, yapay verilerle fiyat tahmin modelleri kurar ve kullanıcıya hem CLI hem de Web UI üzerinden hizmet sunar.

---

## 🚧 Özellikler

- 📊 **Veri Sekmesi**  
  - Tek tıkla sahte veri üretimi  
  - CSV yükleme desteği  
  - Otomatik grafikler (bar & histogram)

- 🤖 **Model Eğit**  
  - RandomForestRegressor ile fiyat tahmini  
  - MAE, RMSE, MAPE ve R² metrik kutuları  
  - Model `joblib` ile kayıtlı

- 🔮 **Tahmin Paneli**  
  - Marka, puan, yorum sayısı gir → anlık fiyat tahmini  
  - 🎯 Arayüz tamamen koyu temaya uygun tasarlandı

---

## 💻 Kurulum

```bash
pip install streamlit pandas numpy scikit-learn joblib openpyxl altair
# Eğer scraping istersen:
pip install selenium webdriver-manager


python trendyol_watch.py mock            # Sahte veri üret
python trendyol_watch.py train           # Model eğit
python trendyol_watch.py predict Samsung 4.4 350  # Tahmin örneği


streamlit run trendyol_watch.py

📁 Proje Yapısı

📦 trendyol_watch.py        # Ana Python betiği (CLI + Web UI)
📄 trendyol_akilli_saat.csv # Verisetiniz (otomatik oluşur)
📄 model.joblib             # Eğitilmiş model

