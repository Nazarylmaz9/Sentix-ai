# 🌌 Sentix AI — E-Ticaret Yorumları için NLP Tabanlı Duygu Analizi

Türkçe e-ticaret platformlarındaki müşteri yorumlarını yapay zeka ile analiz eden, 
duygu sınıflandırması ve kök neden tespiti yapabilen uçtan uca bir web uygulaması.

## 🚀 Özellikler

- **Kurumsal Dashboard** — CSV/Excel yükle, toplu duygu analizi yap, kök neden grafiklerini gör
- **YZ Simülatörü** — Tek yorum gir, anlık Pozitif/Negatif/Nötr tahmini al
- **Çift Motor** — SVM (hızlı) veya BERT (derin analiz) seçeneği
- **Otomatik PDF Raporu** — F1 metrikleri, aksiyon önerileri ve örnek vakalar dahil
- **Akademik Doğrulama** — Etiketli veri yüklenince model metrikleri otomatik hesaplanır

## 🛠️ Kullanılan Teknolojiler

| Alan | Teknoloji |
|------|-----------|
| Makine Öğrenmesi | scikit-learn, SVM, TF-IDF |
| Derin Öğrenme | HuggingFace Transformers, BERT (savasy/bert-base-turkish-sentiment-cased) |
| Veri Dengeleme | imbalanced-learn, SMOTE |
| Web Arayüzü | Streamlit |
| Görselleştirme | Plotly |
| Raporlama | fpdf2 |

## 📊 Model Performansı

Bağımsız test seti (UCI Machine Learning Repository — 27.000+ yorum) üzerinde:

| Metrik | Değer |
|--------|-------|
| Genel Doğruluk | %87,7 |
| Pozitif F1 | %93,6 |
| Negatif F1 | %42,9 |
| Macro F1 | %45,5 |

> Negatif F1'in düşüklüğü, eğitim ve test veri setleri arasındaki domain farklılığını 
> yansıtır. Model ezberleme yapmamış, bağımsız bir veri setinde gerçekçi performans sergilemiştir.

## ⚙️ Kurulum

```bash
pip install streamlit scikit-learn pandas numpy plotly imbalanced-learn transformers fpdf2 joblib
```

## ▶️ Çalıştırma

```bash
streamlit run app.py
```

Eğitilmiş model dosyaları (`svm_model_son.pkl`, `tfidf_vectorizer.pkl`) 
repoda mevcut olduğundan sıfırdan eğitime gerek yoktur.

## 📁 Dosya Yapısı
