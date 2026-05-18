# Banka Müşteri Davranışı Analizi

Bu proje, bir bankanın müşterilerinin vadeli mevduat hesabı açma olasılıklarını tahmin etmeyi amaçlayan bir Makine Öğrenmesi (Machine Learning) ve Web Uygulaması projesidir. 

Doğru müşteri hedefleme ile kampanya maliyetlerini azaltmak ve dönüşüm oranlarını (business value) artırmak hedeflenmiştir.

## 🚀 Özellikler

- **Makine Öğrenmesi Modeli:** Rastgele Orman (Random Forest) algoritması ile yüksek doğruluk oranına sahip sınıflandırma modeli.
- **Kullanıcı Dostu Web Arayüzü:** Müşteri demografik, mali ve kampanya bilgilerini girerek anlık tahmin alabileceğiniz interaktif arayüz.
- **Senaryo Simülatörü (What-If Analysis):** "Görüşme süresi biraz daha uzun olsaydı veya bakiye farklı olsaydı sonuç ne olurdu?" gibi sorulara yanıt arayan simülasyon aracı.
- **Toplu Tahmin (Batch Processing):** CSV dosyası yükleyerek aynı anda birden fazla müşteri için toplu tahmin yapabilme.
- **Açıklanabilir Yapay Zeka (XAI) & Raporlama:** Model sonuçlarını yorumlayabilme ve tahmin raporlarını PDF olarak indirebilme imkanı.

## 📁 Proje Dosya Yapısı

- `app.py`: Streamlit web uygulamasının ana dosyası.
- `train_model.py`: Veriyi temizleyip Random Forest modelini eğiten ve kaydeden Python betiği.
- `model.joblib`: Eğitilmiş makine öğrenmesi modeli.
- `bank (1).csv`: Modelin eğitilmesi ve analizi için kullanılan veri seti.
- `bank2 (1).ipynb`: Keşifçi veri analizi (EDA) ve modelleme adımlarını içeren Jupyter Notebook.
- `requirements.txt`: Projenin çalışması için gerekli kütüphanelerin listesi.

## 🛠️ Kullanılan Teknolojiler

- **Programlama Dili:** Python
- **Veri Manipülasyonu:** Pandas, Numpy
- **Makine Öğrenmesi:** Scikit-Learn, Joblib
- **Web Çerçevesi:** Streamlit
- **Görselleştirme & Diğer:** Plotly, SHAP, FPDF, Streamlit-Lottie

## 💻 Kurulum ve Çalıştırma

Projeyi yerel bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz:

1. **Gerekli kütüphaneleri yükleyin:**
   Terminal veya komut istemcisinde proje dizinine gidin ve aşağıdaki komutu çalıştırın:
   ```bash
   pip install -r requirements.txt
   ```

2. **(Opsiyonel) Modeli yeniden eğitin:**
   Eğer modeli baştan eğitmek isterseniz:
   ```bash
   python train_model.py
   ```

3. **Streamlit uygulamasını başlatın:**
   ```bash
   streamlit run app.py
   ```
   Tarayıcınızda otomatik olarak açılan sekmede veya `http://localhost:8501` adresinde uygulamayı kullanmaya başlayabilirsiniz.

## 👥 Proje Ekibi

- Osman Gündüz SİLMEOĞLU
- Hamza BERRA
- Nihat Kütükoğlu