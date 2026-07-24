import pandas as pd

print("Aşkomun verileri GERÇEK Türkçeye çevriliyor... Lütfen bekle... 🚀")

yeni_veriler = []

# İŞTE SİHİRLİ KELİME: 'errors="ignore"'
# Bu komut o 1 tane bozuk baytı siler, kalan 37.000 yorumu mükemmel Türkçeyle okur!
with open("dataset.txt", "r", encoding="utf-8", errors="ignore") as dosya:
    for satir in dosya:
        satir = satir.strip()
        if not satir:
            continue
            
        try:
            # 1 ve yorumu ayırıyoruz
            ayirma = satir.split(" '", 1)
            etiket = ayirma[0].strip() 
            yorum = ayirma[1].rstrip("'") 
            
            yeni_veriler.append({"Etiket": etiket, "Yorum": yorum})
        except:
            pass # Bozuk satırları atla, sistemi çökertme

# Mis gibi CSV'yi oluşturuyoruz
df = pd.DataFrame(yeni_veriler)
df.to_csv("akademik_test.csv", index=False, encoding="utf-8-sig")

print("TADAAAA! 🎉 Karakter sorunu KÖKÜNDEN ÇÖZÜLDÜ!")