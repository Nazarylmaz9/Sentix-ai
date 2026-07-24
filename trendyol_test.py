import pandas as pd
import joblib
import re
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings('ignore')

print("🧠 1. Adım: Hepsiburada'da eğitilmiş zeki modelimiz (SVM) uykudan uyanıyor...")
svm_model = joblib.load('svm_model_son.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

print("📂 2. Adım: Trendyol verisi okunuyor... (Bozuk harfler UTF-8 ile düzeltiliyor)")
# Senin bilgisayarındaki dosya yolu:
dosya_yolu = r"C:\Users\yilma\Downloads\archive (2)\trendyol.csv"
df_trendyol = pd.read_csv(dosya_yolu, encoding='utf-8')

print("⚙️ 3. Adım: Veri etiketleniyor (3 yıldızlar atılıyor, 4-5 Pozitif, 1-2 Negatif)...")
# Nötr yorumları atalım (Hepsiburada'da da böyle yapmıştık ki kafası karışmasın)
df_trendyol = df_trendyol[df_trendyol['review_rating'] != 3]
df_trendyol = df_trendyol.dropna(subset=['review_body'])

# 3'ten büyük puanlar 1 (Pozitif), küçükler 0 (Negatif)
df_trendyol['Etiket'] = df_trendyol['review_rating'].apply(lambda x: 1 if x > 3 else 0)

print("🧽 4. Adım: Trendyol yorumları temizleniyor...")
def metin_temizle(metin):
    metin = str(metin).lower()
    metin = re.sub(r'[^\w\s]', '', metin)
    metin = re.sub(r'\d+', '', metin)
    return metin.strip()

df_trendyol['Temiz_Metin'] = df_trendyol['review_body'].apply(metin_temizle)

# ==========================================
# ⚠️ İŞTE EN KRİTİK NOKTA: SADECE TRANSFORM!
# ==========================================
print("⏳ 5. Adım: Yorumlar vektörlere çevriliyor ve Şampiyon Model tahmin yapıyor...")
# fit_transform DEĞİL, sadece transform! Modelin hafızasını bozmuyoruz.
X_test_trendyol = vectorizer.transform(df_trendyol['Temiz_Metin'])
y_gercek = df_trendyol['Etiket']

# Tahminleri alalım
y_tahmin = svm_model.predict(X_test_trendyol)

# --- SONUÇ RAPORU ---
acc = accuracy_score(y_gercek, y_tahmin)
print("\n" + "🌟"*25)
print(f"🏆 TRENDYOL GENELLEME BAŞARISI: %{acc*100:.2f}")
print("🌟"*25)

print("\n📝 Detaylı Sınıflandırma Raporu:\n")
print(classification_report(y_gercek, y_tahmin, target_names=['Negatif (0)', 'Pozitif (1)']))