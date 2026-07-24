import streamlit as st
import joblib
import re
import pandas as pd
import plotly.express as px
from collections import Counter
from streamlit_option_menu import option_menu 
import warnings
from fpdf import FPDF
import base64
from datetime import datetime

warnings.filterwarnings('ignore')

# --- 1. SAYFA AYARLARI VE GİZLİ CSS MAKYAJI ---
st.set_page_config(page_title="Sentix AI | İçgörü Paneli", page_icon="🌌", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=DM+Mono:wght@400;500&display=swap');

    :root {
        --bg-base:     #07091a;
        --bg-sidebar:  #0a0d24;
        --bg-surface:  #0e1228;
        --bg-elevated: #141830;
        --bg-card:     #181d38;
        --bg-hover:    #1e2440;
        --border-subtle:  #1d2444;
        --border-default: #253060;
        --border-active:  #4f7cff;
        --text-primary:   #dde4f5;
        --text-secondary: #7e8db0;
        --text-muted:     #3d4d70;
        --accent-blue:  #4f7cff;
        --accent-cyan:  #22d3ee;
        --accent-green: #34d399;
        --accent-red:   #f43f5e;
        --accent-amber: #fbbf24;
        --glow-blue:  rgba(79, 124, 255, 0.18);
        --glow-cyan:  rgba(34, 211, 238, 0.12);
        --radius-sm: 6px;
        --radius-md: 10px;
        --radius-lg: 14px;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif !important;
        background-color: var(--bg-base) !important;
    }

    /* ── Ana sayfa arka planı ── */
    .stApp {
        background: radial-gradient(ellipse 80% 50% at 50% -20%, rgba(30,50,120,0.35) 0%, transparent 70%),
                    linear-gradient(180deg, #07091a 0%, #090b1f 100%) !important;
    }

    /* ── Ana Sayfa Düzeni ── */
    .block-container {
        padding-top: 3rem !important;
        padding-left: 4.5rem !important;
        padding-right: 3.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 100% !important;
    }

    /* ── Sidebar arkaplanı ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0d24 0%, #080b1e 100%) !important;
        border-right: 1px solid var(--border-subtle) !important;
    }

    /* ── Metrik Kartları ── */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-elevated) 100%);
        border: 1px solid var(--border-default);
        padding: 22px 20px;
        border-radius: var(--radius-lg);
        transition: all 0.25s ease;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0,0,8,0.35);
    }
    div[data-testid="metric-container"]::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan));
        opacity: 1;
    }
    div[data-testid="metric-container"]::after {
        content: '';
        position: absolute;
        inset: 0;
        background: radial-gradient(circle at top left, rgba(79,124,255,0.06) 0%, transparent 60%);
        pointer-events: none;
    }
    div[data-testid="metric-container"]:hover {
        border-color: var(--border-active);
        background: linear-gradient(135deg, var(--bg-hover) 0%, var(--bg-card) 100%);
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(0,0,16,0.5), 0 0 0 1px var(--border-active);
    }
    [data-testid="stMetricLabel"] p {
        font-size: 11px !important;
        font-weight: 600 !important;
        letter-spacing: 0.09em !important;
        text-transform: uppercase !important;
        color: var(--text-muted) !important;
    }
    [data-testid="stMetricValue"] {
        font-family: 'DM Mono', monospace !important;
        font-size: 28px !important;
        font-weight: 500 !important;
        color: var(--text-primary) !important;
    }

    /* ── İndirme Butonları ── */
    .stDownloadButton button {
        background: linear-gradient(135deg, #141830 0%, #1a1f3d 100%) !important;
        color: var(--accent-cyan) !important;
        font-weight: 600 !important;
        border-radius: var(--radius-md) !important;
        border: 1px solid var(--border-default) !important;
        letter-spacing: 0.03em !important;
        transition: all 0.22s ease !important;
    }
    .stDownloadButton button:hover {
        background: linear-gradient(135deg, #1e2444 0%, #22285a 100%) !important;
        border-color: var(--accent-cyan) !important;
        box-shadow: 0 0 20px var(--glow-cyan), 0 4px 12px rgba(0,0,0,0.4) !important;
        color: var(--accent-cyan) !important;
    }

    /* ── Divider rengi ── */
    hr {
        border-color: var(--border-subtle) !important;
    }

    /* ── Sidebar Resizer Gizle ── */
    [data-testid="stSidebarResizer"] {
        display: none !important;
        width: 0px !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. MODEL YÜKLEME ---
@st.cache_resource 
def model_yukle():
    model = joblib.load('svm_model_son.pkl')
    vektorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vektorizer

try:
    svm_model, vectorizer = model_yukle()
except FileNotFoundError:
    st.error("HATA: .pkl dosyaları bulunamadı!")
    st.stop()

def metin_temizle(metin):
    metin = str(metin).lower()
    metin = re.sub(r'[^\w\s]', '', metin)
    metin = re.sub(r'\d+', '', metin)
    return metin.strip()

def kategori_bul(metin):
    metin = str(metin).lower()
    kategoriler = []
    if re.search(r'kargo|teslimat|kurye|hız|gün|ulaştı|getirdi|geç|bekle', metin):
        kategoriler.append('Kargo')
    if re.search(r'paket|kutu|jelatin|ambalaj|sarma|patpat|ezik|kırık|açık|özensiz', metin):
        kategoriler.append('Paketleme')
    if re.search(r'fiyat|indirim|pahalı|ucuz|para|kuruş|kampanya|f/p|performans|değer', metin):
        kategoriler.append('Fiyat/Performans')
    if re.search(r'ürün|kalite|bozuk|çalışmıyor|orijinal|sahte|kötü|berbat|iade', metin):
         kategoriler.append('Ürün Kalitesi')
    return ", ".join(kategoriler) if kategoriler else 'Genel'

stop_words_tr = {'ve', 'ile', 'ama', 'fakat', 'lakin', 'ancak', 'çünkü', 'bu', 'şu', 'o', 'bir', 'daha', 'en', 'çok', 'az', 'kadar', 'gibi', 'için', 'ben', 'sen', 'biz', 'siz', 'onlar', 'mı', 'mi', 'mu', 'mü', 'de', 'da', 'ki', 'ne', 'nasıl', 'neden', 'diye', 'her', 'şey', 'biraz', 'böyle'}

# --- 3. YAN MENÜ (PREMIUM SAAS SIDEBAR) ---
with st.sidebar:
    # 3.1 MARKALAMA (Siber Logo Efekti)
    st.sidebar.markdown("""
        <div style='padding: 10px 4px 25px 4px; margin-top: -20px;'>
            <div style='display:flex; align-items:center; gap:12px; margin-bottom:4px;'>
                <div style='flex-shrink:0; width:56px; height:56px; background: linear-gradient(135deg, #4f7cff 0%, #22d3ee 100%); border-radius:14px; display:flex; align-items:center; justify-content:center; font-size:30px; box-shadow: 0 6px 20px rgba(79,124,255,0.4);'>🌌</div>
                <div>
                    <div style='color: #dde4f5; font-size: 28px; font-weight: 800; letter-spacing: -0.5px; line-height:1.1;'>Sentix AI</div>
                    <div style='color: #3d4d70; font-size: 12px; font-weight: 700; letter-spacing: 0.2em; text-transform: uppercase; margin-top:4px;'>İçgörü Platformu</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # 3.2 NEON NAVİGASYON MENÜSÜ
    secim = option_menu(
        menu_title=None, 
        options=["Kurumsal Dashboard", "Yapay Zeka Simülatörü"],
        icons=["bar-chart-fill", "cpu"],
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#4d5566", "font-size": "15px"}, 
            "nav-link": {
                "font-size": "14px", "text-align": "left", "margin": "3px 0",
                "--hover-color": "#1c2230", "color": "#8b949e", "font-weight": "500",
                "border-radius": "8px", "padding": "10px 14px",
            },
            "nav-link-selected": {
                "background-color": "rgba(61, 127, 255, 0.08)",
                "color": "#e6edf3",
                "border-left": "3px solid #3d7fff",
                "font-weight": "600",
            },
        }
    )
    
    st.divider()
    
   # 3.3 SİBER BİLGİ VE DESTEK KARTI (FOOTER)
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0f1330 0%, #131838 100%); padding: 14px 16px; border-radius: 12px; border: 1px solid #1d2444; margin-top: 4px;">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 14px;">
            <div style="width: 8px; height: 8px; border-radius: 50%; background: #34d399; box-shadow: 0 0 8px rgba(52,211,153,0.6); flex-shrink:0;"></div>
            <span style="color: #7e8db0; font-size: 11px; font-weight: 600; letter-spacing: 0.10em; text-transform: uppercase;">Sistem Aktif</span>
        </div>
        <div style="border-top: 1px solid #1d2444; padding-top: 12px;">
            <p style="color: #3d4d70; font-size: 10px; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; margin: 0 0 5px 0;">Eğitim Veri Seti</p>
            <p style="color: #dde4f5; font-size: 22px; font-weight: 600; margin: 0; font-family: 'DM Mono', monospace; letter-spacing: -0.5px;">377.144 <span style="font-size: 12px; font-weight: 400; color: #3d4d70;">satır</span></p>
        </div>
    </div>

    <div style="margin-top: 16px; padding: 0 2px;">
        <p style="color: #3d4d70; font-size: 9px; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase; margin: 0 0 10px 0;">Geliştirici</p>
        <div style="display: flex; flex-direction: column; gap: 8px;">
            <a href="https://github.com/Nazarylmaz9" target="_blank"
               style="display:flex; align-items:center; gap:10px; background:#0f1330; border:1px solid #1d2444; border-radius:8px; padding:9px 12px; text-decoration:none; transition:all 0.2s;">
                <span style="font-size:14px;">💻</span>
                <span style="color:#7e8db0; font-size:12px; font-weight:500;">GitHub Reposu</span>
            </a>
            <a href="mailto:yilmaznazar92@gmail.com"
               style="display:flex; align-items:center; gap:10px; background:#0f1330; border:1px solid #1d2444; border-radius:8px; padding:9px 12px; text-decoration:none; transition:all 0.2s;">
                <span style="font-size:14px;">✉️</span>
                <span style="color:#7e8db0; font-size:12px; font-weight:500;">yilmaznazar92@gmail.com</span>
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)


# --- 4. SAYFA İÇERİKLERİ ---
if secim == "Yapay Zeka Simülatörü":
    # 0. SADECE BU SAYFAYA ÖZEL BUTON VE FONT MAKYAJI
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
        html, body, [class*="css"] {
            font-family: 'DM Sans', sans-serif !important;
        }
        [data-testid="stForm"] {
            border: none !important;
            padding: 0 !important;
            background-color: transparent !important;
        }
        div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
            background: linear-gradient(135deg, #3a5fd4 0%, #4f7cff 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            font-size: 14px !important;
            height: 46px !important;
            letter-spacing: 0.03em !important;
            transition: all 0.22s ease !important;
            box-shadow: 0 2px 16px rgba(79, 124, 255, 0.3) !important;
        }
        div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
            background: linear-gradient(135deg, #4f7cff 0%, #6d96ff 100%) !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 24px rgba(79, 124, 255, 0.45) !important;
            color: #ffffff !important;
        }
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        .stAppDeployButton {display:none;}
        div.row-widget.stRadio > div {
            background: linear-gradient(135deg, #0f1330 0%, #131838 100%);
            padding: 14px 18px;
            border-radius: 10px;
            border: 1px solid #1d2444;
        }
    </style>
    """, unsafe_allow_html=True)

    # BERT MODELİNİ SADECE BİR KERE YÜKLEMEK İÇİN (Hocaya şov yapacağın Cache mimarisi)
    @st.cache_resource
    def load_bert_model():
        from transformers import pipeline
        # Türkçe e-ticaret yorumları için özel eğitilmiş BERT modeli
        return pipeline("sentiment-analysis", model="savasy/bert-base-turkish-sentiment-cased")

    # 1. ASİL VE MODERN BAŞLIK 
    st.markdown("""
        <div style='margin-bottom: 28px; padding-bottom: 20px; border-bottom: 1px solid #1d2444;'>
            <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 6px;'>
                <div style='width: 4px; height: 28px; background: linear-gradient(180deg, #4f7cff, #22d3ee); border-radius: 2px;'></div>
                <h2 style='color: #dde4f5; font-size: 26px; font-weight: 700; margin: 0; letter-spacing: -0.4px;'>Canlı Simülatör</h2>
            </div>
            <p style='color: #3d4d70; font-size: 13px; margin: 0; padding-left: 14px; letter-spacing: 0.01em;'>
                NLP motorunun analiz yeteneğini gerçek zamanlı olarak test edin.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # 2. HAYALET FORM YAPISI VE MOTOR SEÇİMİ
    with st.form(key='simulator_form'):
        kullanici_yorumu = st.text_area(
            "Müşteri Geri Bildirimini Girin:", 
            height=80, 
            placeholder="Örn: Kargo inanılmaz hızlı geldi ama ürünün paketlemesi rezaletti..."
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # --- İŞTE O SİHİRLİ MOTOR SEÇİMİ ---
        st.markdown("<h4 style='color: #4d5566; font-size: 12px; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 8px;'>Analiz Motoru</h4>", unsafe_allow_html=True)
        secilen_motor = st.radio(
            label="Motor Seçimi",
            options=["Standart Motor (SVM - Yüksek Hız)", "Premium Motor (BERT - Derin Anlamsal Analiz)"],
            label_visibility="collapsed"
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        submit_button = st.form_submit_button(label="✦ Algoritmayı Çalıştır", use_container_width=True)
    
    # 3. ANALİZ VE SONUÇ EKRANI
    if submit_button:
        if kullanici_yorumu.strip():
            # Hangi motor seçildiyse spinner'da onu yazıyoruz!
            motor_ismi = "Sentix AI (BERT)" if "BERT" in secilen_motor else "Sentix AI (SVM)"
            with st.spinner(f"{motor_ismi} analiz ediyor..."):
                
                temiz_metin = metin_temizle(kullanici_yorumu)
                kullanici_yorumu_kucuk = kullanici_yorumu.lower()
                kategori = kategori_bul(kullanici_yorumu) # Kategori her iki motorda da aynı mantıkla çalışır
                
                # --- YENİ YAPI: MOTORLARI AYIRIYORUZ ---
                if "Standart" in secilen_motor:
                    # MOTOR 1: SENİN EFSANEVİ SVM MODELİN
                    notr_kaliplar = ["ne iyi ne kötü", "idare eder", "eh işte", "fena sayılmaz", "orta şekerli", "şöyle böyle" ,"fena değil"]
                    if any(kalip in kullanici_yorumu_kucuk for kalip in notr_kaliplar):
                        duygu_sonucu = "NÖTR"
                    else:
                        vektor = vectorizer.transform([temiz_metin])
                        karar_skoru = svm_model.decision_function(vektor)[0]
                        if karar_skoru > 0.4:
                            duygu_sonucu = "POZİTİF"
                        elif karar_skoru < -0.4:
                            duygu_sonucu = "NEGATİF"
                        else:
                            duygu_sonucu = "NÖTR"
                            
                else:
                    # MOTOR 2: GOOGLE BERT MİMARİSİ (HuggingFace)
                    try:
                        bert_pipeline = load_bert_model()
                        bert_sonuc = bert_pipeline(kullanici_yorumu)[0]
                        # BERT modeli 'positive' veya 'negative' döndürür
                        if bert_sonuc['label'] == 'positive':
                            duygu_sonucu = "POZİTİF"
                        else:
                            duygu_sonucu = "NEGATİF"
                    except Exception as e:
                        st.error("Model yüklenemedi. Lütfen terminalden 'pip install transformers torch' komutunu çalıştırdığınıza emin olun.")
                        duygu_sonucu = "HATA"

                # Geri kalan kart tasarımın ve renklerin BİREBİR AYNI KALDI
                if duygu_sonucu != "HATA":
                    st.divider()
                    st.markdown(f"<p style='color: #4d5566; font-size: 11px; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 14px;'>Analiz Sonuçları — {'BERT Motoru' if 'BERT' in secilen_motor else 'SVM Motoru'}</p>", unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #0f1330 0%, #141838 100%); border: 1px solid #1d2444; border-radius: 12px; padding: 28px 24px; text-align: center; min-height: 150px; display: flex; flex-direction: column; justify-content: center; gap: 8px; box-shadow: 0 4px 20px rgba(0,0,12,0.4);">
                            <p style="color: #3d4d70; font-size: 10px; font-weight: 600; margin: 0; letter-spacing: 0.12em; text-transform: uppercase;">Tespit Edilen Kategori</p>
                            <h2 style="color: #dde4f5; margin: 0; font-size: 22px; font-weight: 600; letter-spacing: -0.2px;">{kategori}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with col2:
                        if duygu_sonucu == "POZİTİF":
                            sonuc_renk = "#00b4d8" 
                            alt_metin = "Müşteri Memnuniyeti Sağlandı"
                            gorsel = "✦"
                        elif duygu_sonucu == "NEGATİF":
                            sonuc_renk = "#f72585" 
                            alt_metin = "Riskli Müşteri / Şikayet"
                            gorsel = "⊗"
                        else:
                            sonuc_renk = "#a8b2c1" 
                            alt_metin = "Duygu Tespiti Belirsiz / Nötr"
                            gorsel = "⊜" 
                            
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #0f1330 0%, #141838 100%); border: 1px solid #1d2444; border-radius: 12px; padding: 28px 24px; text-align: center; min-height: 150px; display: flex; flex-direction: column; justify-content: center; gap: 6px; border-top: 3px solid {sonuc_renk}; box-shadow: 0 4px 24px rgba(0,0,12,0.4);">
                            <p style="color: #3d4d70; font-size: 10px; font-weight: 600; margin: 0; letter-spacing: 0.12em; text-transform: uppercase;">Duygu Analizi</p>
                            <h2 style="color: {sonuc_renk}; margin: 0; font-size: 28px; font-weight: 700; letter-spacing: 0.5px;">{duygu_sonucu}</h2>
                            <p style="color: {sonuc_renk}; font-size: 12px; margin: 0; opacity: 0.7; font-weight: 500;">{alt_metin}</p>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.warning("⚠ Lütfen analiz edilecek bir müşteri yorumu girin.")
            
elif secim == "Kurumsal Dashboard":
    # 0. SİBER TASARIM CSS ENTEGRASYONU (Sayfa Geneli)
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
        html, body, [class*="css"] {
            font-family: 'DM Sans', sans-serif !important;
        }
        div.stButton > button {
            background: linear-gradient(135deg, #3a5fd4 0%, #4f7cff 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            font-size: 14px !important;
            height: 46px !important;
            transition: all 0.22s ease !important;
            letter-spacing: 0.03em !important;
            box-shadow: 0 2px 16px rgba(79, 124, 255, 0.3) !important;
        }
        div.stButton > button:hover {
            background: linear-gradient(135deg, #4f7cff 0%, #6d96ff 100%) !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 24px rgba(79, 124, 255, 0.45) !important;
            color: white !important;
        }
        [data-testid="stFileUploadDropzone"] div div span { display: none !important; }
        [data-testid="stFileUploadDropzone"] div div small { display: none !important; }
        [data-testid="stFileUploadDropzone"] div div::before {
            content: "Dosyanızı Buraya Sürükleyip Bırakın" !important;
            color: #7e8db0 !important;
            font-weight: 500 !important;
            font-size: 15px !important;
            display: block !important;
            margin-bottom: 4px !important;
            visibility: visible !important;
        }
        [data-testid="stFileUploadDropzone"] div div::after {
            content: "Maks. 200MB · CSV veya XLSX" !important;
            color: #3d4d70 !important;
            font-size: 12px !important;
            display: block !important;
            visibility: visible !important;
            margin-top: 4px !important;
        }
        [data-testid="stFileUploadDropzone"] button { color: transparent !important; }
        [data-testid="stFileUploadDropzone"] button::before {
            content: "Dosya Seç" !important;
            color: #ffffff !important;
            position: absolute !important;
            top: 50% !important;
            left: 50% !important;
            transform: translate(-50%, -50%) !important;
            font-weight: 600 !important;
            visibility: visible !important;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style='margin-bottom: 28px; padding-bottom: 20px; border-bottom: 1px solid #1d2444;'>
            <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 6px;'>
                <div style='width: 4px; height: 28px; background: linear-gradient(180deg, #4f7cff, #22d3ee); border-radius: 2px;'></div>
                <h2 style='color: #dde4f5; font-size: 26px; font-weight: 700; margin: 0; letter-spacing: -0.4px;'>Kurumsal Dashboard</h2>
            </div>
            <p style='color: #3d4d70; font-size: 13px; margin: 0; padding-left: 14px; letter-spacing: 0.01em;'>
                Müşteri veri setinizi yükleyin — NLP motoru duygu ve kök neden analizini saniyeler içinde çıkarsın.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # 2. DOSYA YÜKLEYİCİ
    yuklenen_dosya = st.file_uploader("Analiz Edilecek Veri Setini Yükleyin (CSV veya Excel)", type=['csv', 'xlsx'])
    
    if yuklenen_dosya:
        try:
            df_yeni = pd.read_csv(yuklenen_dosya) if yuklenen_dosya.name.endswith('.csv') else pd.read_excel(yuklenen_dosya)
            
            metin_sutunlari = [col for col in df_yeni.select_dtypes(include=['object']).columns if not any(x in col.lower() for x in ['url', 'link', 'id', 'date', 'tarih'])]
            
            varsayilan_index = 0
            for i, col in enumerate(metin_sutunlari):
                if 'review' in col.lower() or 'yorum' in col.lower() or 'body' in col.lower():
                    varsayilan_index = i
                    break
            
            st.info("Sistem, yorumların bulunduğu sütunu otomatik tespit etmeye çalıştı. Lütfen doğruluğunu kontrol edin.")
            yorum_sutunu = st.selectbox("Analiz Edilecek Sütun:", metin_sutunlari, index=varsayilan_index)
            
            if 'analiz_tamamlandi' not in st.session_state:
                st.session_state.analiz_tamamlandi = False

            if st.button("✦ Kapsamlı Raporu Oluştur", use_container_width=True):
                with st.spinner('Yapay zeka verileri analiz ediyor... Lütfen bekleyin.'):
                    df_islem = df_yeni.dropna(subset=[yorum_sutunu]).copy()
                    
                    # 1. Metin Temizleme ve Vektörizasyon (Birebir Senin Kodun)
                    df_islem['Temiz_Metin'] = df_islem[yorum_sutunu].astype(str).apply(metin_temizle)
                    vektorler = vectorizer.transform(df_islem['Temiz_Metin'])
                    skorlar = svm_model.decision_function(vektorler)
                    
                    # 2. Matematiksel Sınıflandırma (Hassas Ayarlarını Koruduk!)
                    df_islem['Karar_Skoru'] = skorlar
                    df_islem['Duygu_Etiketi'] = 'Nötr' 
                    
                    # Senin 0.1 ve -0.4 olan özel sınırların:
                    df_islem.loc[df_islem['Karar_Skoru'] > 0.1, 'Duygu_Etiketi'] = 'Pozitif'
                    df_islem.loc[df_islem['Karar_Skoru'] < -0.4, 'Duygu_Etiketi'] = 'Negatif'
                    df_islem.loc[df_islem['Temiz_Metin'].str.strip() == '', 'Duygu_Etiketi'] = 'Nötr'
                    
                    # 3. Kategori Bulma
                    df_islem['Kategori'] = df_islem[yorum_sutunu].astype(str).apply(kategori_bul)
                    
                    st.session_state.df_sonuc = df_islem
                    st.session_state.yorum_sutunu = yorum_sutunu
                    st.session_state.analiz_tamamlandi = True

            if st.session_state.analiz_tamamlandi:
                df_sonuc = st.session_state.df_sonuc
                aktif_yorum_sutunu = st.session_state.yorum_sutunu
                
                st.divider()
                
                # --- 🪄 AKILLI HAYALET METRİK (SADECE ETİKET VARSA GÖRÜNÜR) ---
                etiket_sutun_adi = next((col for col in df_yeni.columns if col.lower() in ['etiket', 'label', 'target', 'sentiment', 'label_binary']), None)
                
                if etiket_sutun_adi:
                    basarili_tahmin = 0
                    toplam_test = len(df_sonuc)

                    if toplam_test > 0:
                        for index, row in df_sonuc.iterrows():
                            gercek_etiket = str(row[etiket_sutun_adi]).strip().lower()
                            tahmin = str(row['Duygu_Etiketi']).strip().lower()  # upper() yerine lower() kullan!

                            is_real_positive = any(word in gercek_etiket for word in ['1', 'pos', 'poz', 'olumlu', 'positive'])
                            is_real_negative = any(word in gercek_etiket for word in ['0', 'neg', 'olumsuz', 'negative'])
                            is_real_neutral = any(word in gercek_etiket for word in ['2', 'not', 'nöt', 'neutral'])

                            # upper() yerine lower() ile karşılaştır - Türkçe İ/i sorunu ortadan kalkar!
                            if (tahmin == "pozitif" and is_real_positive) or \
                               (tahmin == "negatif" and is_real_negative) or \
                               (tahmin == "nötr" and is_real_neutral):
                                basarili_tahmin += 1
                    dogruluk_orani = (basarili_tahmin / toplam_test) * 100

                    # --- METRİK HESAPLAMA ---
                    from sklearn.metrics import classification_report

                    gercekler = []
                    tahminler = []

                    for index, row in df_sonuc.iterrows():
                        gercek = str(row[etiket_sutun_adi]).strip().lower()
                        tahmin = str(row['Duygu_Etiketi']).strip().lower()
    
                        if any(w in gercek for w in ['1', 'pos', 'positive', 'poz', 'olumlu']):
                            gercekler.append('pozitif')
                        elif any(w in gercek for w in ['0', 'neg', 'negative', 'olumsuz']):
                            gercekler.append('negatif')
                        else:
                            gercekler.append('nötr')
    
                        tahminler.append(tahmin)

                    rapor = classification_report(gercekler, tahminler, output_dict=True, zero_division=0)

                    # Sınıf bazlı F1 skorları
                    f1_pozitif = rapor.get('pozitif', {}).get('f1-score', 0) * 100
                    f1_negatif = rapor.get('negatif', {}).get('f1-score', 0) * 100
                    f1_macro   = rapor.get('macro avg', {}).get('f1-score', 0) * 100

                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #0f1330 0%, #14193a 100%); padding: 20px 24px; border-radius: 12px; border: 1px solid #1d2444; border-left: 3px solid #4f7cff; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 4px 20px rgba(0,0,16,0.4);">
                        <div>
                            <p style="color: #4f7cff; margin: 0 0 4px 0; font-size: 10px; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase;">Akademik Doğrulama Modu</p>
                            <p style="color: #7e8db0; margin: 0; font-size: 13px;">Etiketli test verisi algılandı. Modelin genelleme yeteneği hesaplanıyor.</p>
                        </div>
                        <div style="text-align: right; flex-shrink: 0; margin-left: 24px;">
                            <p style="color: #dde4f5; margin: 0; font-size: 32px; font-weight: 700; font-family: 'DM Mono', monospace; letter-spacing: -1px;">%{dogruluk_orani:.1f}</p>
                            <p style="color: #3d4d70; margin: 0; font-size: 10px; font-weight: 600; letter-spacing: 0.10em; text-transform: uppercase;">Doğruluk Oranı</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # --- F1 METRİK KARTLARI ---
                    m1, m2, m3 = st.columns(3)

                    with m1:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #0f1330 0%, #141838 100%); border:1px solid #1d2444; border-top: 2px solid #4f7cff; border-radius:10px; padding:18px 20px; text-align:center; box-shadow: 0 4px 16px rgba(0,0,12,0.3);">
                            <p style="color:#3d4d70; font-size:10px; font-weight:600; letter-spacing:0.12em; text-transform:uppercase; margin:0 0 8px 0;">Pozitif F1-Score</p>
                            <h2 style="color:#4f7cff; font-size:28px; font-weight:600; margin:0; font-family:'DM Mono',monospace;">%{f1_pozitif:.1f}</h2>
                        </div>
                        """, unsafe_allow_html=True)

                    with m2:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #0f1330 0%, #141838 100%); border:1px solid #1d2444; border-top: 2px solid #f43f5e; border-radius:10px; padding:18px 20px; text-align:center; box-shadow: 0 4px 16px rgba(0,0,12,0.3);">
                            <p style="color:#3d4d70; font-size:10px; font-weight:600; letter-spacing:0.12em; text-transform:uppercase; margin:0 0 8px 0;">Negatif F1-Score</p>
                            <h2 style="color:#f43f5e; font-size:28px; font-weight:600; margin:0; font-family:'DM Mono',monospace;">%{f1_negatif:.1f}</h2>
                        </div>
                        """, unsafe_allow_html=True)

                    with m3:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #0f1330 0%, #141838 100%); border:1px solid #1d2444; border-top: 2px solid #22d3ee; border-radius:10px; padding:18px 20px; text-align:center; box-shadow: 0 4px 16px rgba(0,0,12,0.3);">
                            <p style="color:#3d4d70; font-size:10px; font-weight:600; letter-spacing:0.12em; text-transform:uppercase; margin:0 0 8px 0;">Macro F1 (Genel)</p>
                            <h2 style="color:#22d3ee; font-size:28px; font-weight:600; margin:0; font-family:'DM Mono',monospace;">%{f1_macro:.1f}</h2>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                # 4. STANDART METRİKLER (Senin 4 kolonlu tasarımın, hiç değişmedi!)
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Toplam Analiz", f"{len(df_sonuc):,}")
                col2.metric("Pozitif Geri Dönüş ✦", f"{len(df_sonuc[df_sonuc['Duygu_Etiketi'] == 'Pozitif']):,}")
                col3.metric("Nötr Yorumlar ⊜", f"{len(df_sonuc[df_sonuc['Duygu_Etiketi'] == 'Nötr']):,}")
                col4.metric("Riskli Müşteri ⊗", f"{len(df_sonuc[df_sonuc['Duygu_Etiketi'] == 'Negatif']):,}")
                
                # 5. GRAFİKLER (Tamamen Yenilenmiş Premium Renk Paleti!)
                st.divider()
                st.subheader("Genel Müşteri Memnuniyet Dağılımı")
                
                renk_paleti = {
                    'Pozitif': '#3d7fff',
                    'Nötr': '#6366f1',
                    'Negatif': '#f43f5e'
                } 
                
                fig_duygu = px.pie(df_sonuc, names='Duygu_Etiketi', 
                                   color='Duygu_Etiketi', color_discrete_map=renk_paleti, 
                                   hole=0.45, 
                                   template='plotly_dark', height=500)
                
                # Dilim çizgilerini arkaplanla aynı yapıyoruz ki kusursuz bütünleşsin
                fig_duygu.update_traces(textposition='inside', textinfo='percent+label', 
                                        textfont_size=15, marker=dict(line=dict(color='#0e1117', width=3)))
                st.plotly_chart(fig_duygu, use_container_width=True)
                    
                st.divider()
                st.subheader("Kök Neden Analizi — Şikayetler Nereden Geliyor?")
                
                # SADECE NET NEGATİFLERİ ALIYORUZ!
                df_negatif = df_sonuc[df_sonuc['Duygu_Etiketi'] == 'Negatif']
                
                kategori_sayilari = df_negatif['Kategori'].value_counts().reset_index().head(6)
                kategori_sayilari.columns = ['Kategori', 'Şikayet Sayısı']
                
                siber_mavi_gecisi = [[0.0, '#1e3a5f'], [0.5, '#3d7fff'], [1.0, '#22d3ee']]
                
                fig_kategori = px.bar(kategori_sayilari, y='Kategori', x='Şikayet Sayısı', 
                                      text_auto=True, color='Şikayet Sayısı', 
                                      color_continuous_scale=siber_mavi_gecisi, 
                                      orientation='h', template='plotly_dark', height=400)
                
                fig_kategori.update_traces(marker_line_width=0, textfont_size=14, textangle=0, textposition="outside", cliponaxis=False)
                fig_kategori.update_yaxes(categoryorder='total ascending', title="", tickfont=dict(size=14))
                fig_kategori.update_xaxes(title="Toplam Şikayet Sayısı", showgrid=True, gridcolor='#30363d')
                
                # Renk cetveli (Legend/Colorbar) geri geldi ve siber temaya uyarlandı!
                fig_kategori.update_layout(coloraxis_colorbar=dict(
                    title="Şikayet Sayısı",
                    thicknessmode="pixels", thickness=15,
                    lenmode="pixels", len=250,
                    yanchor="middle", y=0.5,
                    ticks="outside", tickcolor="#8b949e",
                    title_font=dict(size=13, color="#8b949e"),
                    tickfont=dict(size=12, color="#8b949e")
                ))
                
                st.plotly_chart(fig_kategori, use_container_width=True)

                # 6. KRİTİK ŞİKAYET ALARM AKIŞI (Artık Nötrler ve Karışıklar Buraya Düşmeyecek!)
                st.divider()
                st.subheader("Kritik Şikayet Panosu")
                st.caption("Yapay zekanın riskli olarak işaretlediği müşteri geri bildirimlerinden örnekler.")
                
                if not df_negatif.empty:
                    # 1. KUSURSUZ HAVUZ: Bütün negatifleri en kötüden iyiye (en eksiden artıya) doğru sırala.
                    # Sadece en dipteki, makinenin %1.000 emin olduğu "En Öfkeli 50 Yorumu" bir havuza al!
                    # (Böylece "sağlamdı", "stoklu ürün" gibi 0'a yakın sınırda kalan saçmalıklar bu 50'ye giremez bile)
                    en_kotuler_havuzu = df_negatif.sort_values(by='Karar_Skoru', ascending=True).head(50)
                    
                    # 2. CANLI AKIŞ: Artık bu "garantili öfkeliler" havuzunun içinden rastgele 5 tane seç!
                    ornek_sikayetler = en_kotuler_havuzu.sample(min(5, len(en_kotuler_havuzu)))
                    
                    # 3. Ekrana Basma Döngüsü
                    for index, row in ornek_sikayetler.iterrows():
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #0f1330 0%, #141838 100%); border: 1px solid #1d2444; border-left: 3px solid #f43f5e; padding: 16px 20px; border-radius: 10px; margin-bottom: 12px; box-shadow: 0 4px 16px rgba(0,0,12,0.35);">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                <span style="color: #22d3ee; font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.12em;">{row['Kategori']}</span>
                                <span style="background: rgba(244,63,94,0.12); color: #f43f5e; font-size: 10px; font-weight: 600; padding: 3px 10px; border-radius: 4px; letter-spacing: 0.08em; border: 1px solid rgba(244,63,94,0.2);">RİSKLİ</span>
                            </div>
                            <p style="color: #7e8db0; font-size: 14px; line-height: 1.65; margin: 0; font-style: italic;">"{row[aktif_yorum_sutunu]}"</p>
                        </div>
                        """, unsafe_allow_html=True)
                # 7. YAPAY ZEKA DESTEKLİ YÖNETİCİ ÖZETİ
                st.divider()
                st.markdown("<p style='color: #3d4d70; font-size: 11px; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 16px;'>Yapay Zeka Destekli Yönetici Özeti</p>", unsafe_allow_html=True)

                toplam_kayit = len(df_sonuc)
                poz_oran_ui = round((len(df_sonuc[df_sonuc['Duygu_Etiketi']=='Pozitif']) / toplam_kayit) * 100, 1)
                neg_oran_ui = round((len(df_sonuc[df_sonuc['Duygu_Etiketi']=='Negatif']) / toplam_kayit) * 100, 1)
                en_sorunlu_kat = df_sonuc[df_sonuc['Duygu_Etiketi'] == 'Negatif']['Kategori'].value_counts()
                en_sorunlu_kat_adi = en_sorunlu_kat.index[0] if not en_sorunlu_kat.empty else 'Genel'

                # Genel durum satırı
                if neg_oran_ui > 40:
                    durum_renk, durum_etiket, durum_mesaj = "#f43f5e", "KRİTİK", f"Müşterilerin %{neg_oran_ui}'i olumsuz deneyim bildiriyor. Bu oran sektör ortalamasının belirgin şekilde üzerinde — acil operasyonel müdahale gerektirir."
                elif neg_oran_ui > 20:
                    durum_renk, durum_etiket, durum_mesaj = "#fbbf24", "DİKKAT", f"Olumsuz yorum oranı %{neg_oran_ui} seviyesinde. Artış eğilimi izlenmeli, önleyici aksiyonlar planlanmalı."
                else:
                    durum_renk, durum_etiket, durum_mesaj = "#34d399", "SAĞLIKLI", f"Müşteri memnuniyeti %{poz_oran_ui} ile güçlü bir seviyede. Mevcut kalite standartlarının korunması yeterlidir."

                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #0f1330 0%, #141838 100%); border: 1px solid #1d2444; border-left: 3px solid {durum_renk}; border-radius: 10px; padding: 16px 20px; margin-bottom: 10px; box-shadow: 0 4px 16px rgba(0,0,12,0.3);">
                    <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
                        <span style="background: rgba(255,255,255,0.05); color:{durum_renk}; font-size:10px; font-weight:700; padding:3px 10px; border-radius:4px; letter-spacing:0.10em; border:1px solid {durum_renk}33;">{durum_etiket}</span>
                        <span style="color:#3d4d70; font-size:10px; font-weight:600; letter-spacing:0.10em; text-transform:uppercase;">Genel Müşteri Sağlığı</span>
                    </div>
                    <p style="color:#7e8db0; font-size:14px; margin:0; line-height:1.65;">{durum_mesaj}</p>
                </div>
                """, unsafe_allow_html=True)

                # Kategoriye özel aksiyon önerileri
                aksiyon_map = {
                    'Kargo':             ("#22d3ee", "KARGO & LOJİSTİK", "Kargo şikayetleri baskın kategori. Lojistik partner SLA'ları gözden geçirilmeli; teslimat takip sistemi güçlendirilmeli ve geç teslimat bildirim mekanizması kurulmalı."),
                    'Ürün Kalitesi':     ("#fbbf24", "ÜRÜN KALİTESİ",    "Ürün kalite şikayetleri öne çıkıyor. Tedarikçi denetim süreçleri sıkılaştırılmalı, iade/değişim politikaları müşteri lehine güncellenmeli."),
                    'Fiyat/Performans':  ("#a78bfa", "FİYAT / DEĞER",    "Fiyat-performans algısı zayıf. Ürün açıklamaları ve görsel içerikler güçlendirilmeli; beklenti yönetimi iyileştirilmeli."),
                    'Paketleme':         ("#34d399", "PAKETLEME",         "Paketleme kaynaklı şikayetler tespit edildi. Kırılgan ürün protokolleri ve ambalaj standartları yeniden değerlendirilmeli."),
                    'Genel':             ("#7e8db0", "GENEL ŞİKAYETLER",  "Belirli bir kategoriye girmeyen genel şikayetler yüksek. Müşteri hizmetleri ilk temas çözüm (FCR) oranı artırılmalı."),
                }

                if en_sorunlu_kat_adi in aksiyon_map:
                    ak_renk, ak_baslik, ak_metin = aksiyon_map[en_sorunlu_kat_adi]
                else:
                    ak_renk, ak_baslik, ak_metin = aksiyon_map['Genel']

                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #0f1330 0%, #141838 100%); border: 1px solid #1d2444; border-left: 3px solid {ak_renk}; border-radius: 10px; padding: 16px 20px; margin-bottom: 10px; box-shadow: 0 4px 16px rgba(0,0,12,0.3);">
                    <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
                        <span style="background: rgba(255,255,255,0.05); color:{ak_renk}; font-size:10px; font-weight:700; padding:3px 10px; border-radius:4px; letter-spacing:0.10em; border:1px solid {ak_renk}33;">ÖNCELİKLİ AKSİYON · {ak_baslik}</span>
                    </div>
                    <p style="color:#7e8db0; font-size:14px; margin:0; line-height:1.65;">{ak_metin}</p>
                </div>
                """, unsafe_allow_html=True)

                # Stratejik öneri — her zaman gösterilir
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #0f1330 0%, #141838 100%); border: 1px solid #1d2444; border-left: 3px solid #4f7cff; border-radius: 10px; padding: 16px 20px; margin-bottom: 10px; box-shadow: 0 4px 16px rgba(0,0,12,0.3);">
                    <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
                        <span style="background: rgba(255,255,255,0.05); color:#4f7cff; font-size:10px; font-weight:700; padding:3px 10px; border-radius:4px; letter-spacing:0.10em; border:1px solid #4f7cff33;">STRATEJİK ÖNLEM</span>
                    </div>
                    <p style="color:#7e8db0; font-size:14px; margin:0; line-height:1.65;">Analiz edilen {toplam_kayit:,} yorumun <strong style='color:#dde4f5;'>%{neg_oran_ui}'i</strong> negatif içeriklidir. Riskli müşteri segmentine yönelik proaktif geri kazanım iletişimi (e-posta/SMS) başlatılması ve periyodik duygu analizi raporlamasının kurumsal KPI'lara eklenmesi önerilir.</p>
                </div>
                """, unsafe_allow_html=True)

                # 8. RAPORLAMA VE DIŞA AKTARIM MERKEZİ
                st.divider()
                st.markdown("<p style='color: #3d4d70; font-size: 11px; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 16px;'>Raporlama ve Dışa Aktarım</p>", unsafe_allow_html=True)
                
                # 1. ADIM: Önce tabloyu gösteriyoruz (Kullanıcı akışına uygun)
                with st.expander("Detaylı Veri Tablosunu Görüntüle"):
                    df_indir = df_sonuc[[aktif_yorum_sutunu, 'Kategori', 'Duygu_Etiketi']]
                    st.dataframe(df_indir, use_container_width=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # 2. ADIM: İndirme butonlarını tablonun ALTINA yerleştiriyoruz
                buton_kolon1, buton_kolon2 = st.columns(2)
                
                with buton_kolon1:
                    # ── PDF OLUŞTURMA ──────────────────────────────────────────────
                    pdf = FPDF()
                    pdf.set_auto_page_break(auto=True, margin=18)
                    pdf.add_page()

                    # Türkçe font
                    try:
                        pdf.add_font('ArialTR', '',  'C:\\Windows\\Fonts\\arial.ttf',   uni=True)
                        pdf.add_font('ArialTR', 'B', 'C:\\Windows\\Fonts\\arialbd.ttf', uni=True)
                        pdf.add_font('ArialTR', 'I', 'C:\\Windows\\Fonts\\ariali.ttf',  uni=True)
                        font_tipi = 'ArialTR'
                    except:
                        font_tipi = 'Arial'

                    # ── HEADER BANTI ──────────────────────────────────────────────
                    pdf.set_fill_color(10, 13, 36)
                    pdf.rect(0, 0, 210, 38, 'F')
                    # sol şerit aksan
                    pdf.set_fill_color(79, 124, 255)
                    pdf.rect(0, 0, 4, 38, 'F')

                    pdf.set_font(font_tipi, 'B', 20)
                    pdf.set_text_color(255, 255, 255)
                    pdf.set_xy(10, 9)
                    pdf.cell(120, 10, txt="SENTIX CORE", ln=False)

                    pdf.set_font(font_tipi, '', 9)
                    pdf.set_text_color(148, 163, 184)
                    tarih = datetime.now().strftime("%d.%m.%Y  %H:%M")
                    pdf.set_xy(130, 9)
                    pdf.cell(70, 10, txt=f"Rapor Tarihi: {tarih}", align='R', ln=False)

                    pdf.set_font(font_tipi, 'I', 10)
                    pdf.set_text_color(100, 130, 180)
                    pdf.set_xy(10, 22)
                    pdf.cell(0, 8, txt="Otomatik Müşteri İçgörü ve Duygu Analizi Raporu", ln=True)

                    pdf.set_y(46)

                    # ── YARDIMCI FONKSİYONLAR ─────────────────────────────────────
                    def bolum_baslik(pdf, no, baslik, font):
                        pdf.set_font(font, 'B', 12)
                        pdf.set_text_color(10, 13, 36)
                        pdf.set_fill_color(235, 240, 255)
                        pdf.cell(0, 9, txt=f"  {no}. {baslik}", ln=True, fill=True)
                        pdf.ln(3)

                    def bilgi_kutusu(pdf, metin, font, bg=(241,245,249), tc=(51,65,85)):
                        pdf.set_font(font, '', 10)
                        pdf.set_text_color(*tc)
                        pdf.set_fill_color(*bg)
                        pdf.multi_cell(0, 7, txt=metin, border=0, fill=True)
                        pdf.ln(4)

                    # ── ORAN HESAPLAMALARI ────────────────────────────────────────
                    poz_oran = round((len(df_sonuc[df_sonuc['Duygu_Etiketi']=='Pozitif']) / len(df_sonuc))*100, 1)
                    neg_oran = round((len(df_sonuc[df_sonuc['Duygu_Etiketi']=='Negatif']) / len(df_sonuc))*100, 1)
                    notr_oran = round((len(df_sonuc[df_sonuc['Duygu_Etiketi']=='Nötr'])    / len(df_sonuc))*100, 1)
                    en_sorunlu_pdf = df_sonuc[df_sonuc['Duygu_Etiketi']=='Negatif']['Kategori'].value_counts()
                    en_sorunlu_pdf_adi = en_sorunlu_pdf.index[0] if not en_sorunlu_pdf.empty else 'Genel'

                    # ── 1. YÖNETİCİ ÖZETİ ────────────────────────────────────────
                    bolum_baslik(pdf, 1, "YÖNETİCİ ÖZETİ", font_tipi)

                    if poz_oran > 60:
                        ozet = (f"Sisteme yüklenen {len(df_sonuc):,} adet müşteri geribildirimi Sentix AI motoru "
                                f"tarafından işlenmiştir. Genel müşteri memnuniyeti %{poz_oran} ile pazar "
                                f"standartlarının üzerindedir. Bununla birlikte %{neg_oran} oranındaki riskli "
                                f"bildirimler; ürün/hizmet geliştirme süreçlerine entegre edilmeli ve proaktif "
                                f"geri kazanım aksiyonlarıyla yönetilmelidir.")
                    else:
                        ozet = (f"Sisteme yüklenen {len(df_sonuc):,} adet müşteri geribildirimi Sentix AI motoru "
                                f"tarafından işlenmiştir. %{neg_oran} seviyesindeki riskli şikayet oranı sektör "
                                f"normlarının üzerindedir ve acil operasyonel müdahale gerektirmektedir. "
                                f"Memnuniyet oranı %{poz_oran} ile hedefin altında seyretmektedir.")
                    bilgi_kutusu(pdf, ozet, font_tipi)

                    # ── 2. KPI TABLOSU ────────────────────────────────────────────
                    bolum_baslik(pdf, 2, "TEMEL PERFORMANS GÖSTERGELERİ (KPI)", font_tipi)

                    # Başlık satırı
                    pdf.set_font(font_tipi, 'B', 10)
                    pdf.set_fill_color(218, 228, 248)
                    pdf.set_text_color(10, 13, 36)
                    for baslik_kpi, gen in [("Toplam Analiz", 47), ("Pozitif", 47), ("Nötr", 47), ("Negatif (Riskli)", 49)]:
                        pdf.cell(gen, 8, txt=baslik_kpi, border=1, fill=True, align='C')
                    pdf.ln()

                    # Veri satırı
                    pdf.set_font(font_tipi, '', 11)
                    pdf.set_fill_color(249, 250, 252)
                    pdf.set_text_color(10, 13, 36)
                    pdf.cell(47, 9, txt=f"{len(df_sonuc):,}", border=1, fill=True, align='C')
                    pdf.set_text_color(30, 100, 180)
                    pdf.cell(47, 9, txt=f"%{poz_oran}", border=1, fill=True, align='C')
                    pdf.set_text_color(80, 80, 120)
                    pdf.cell(47, 9, txt=f"%{notr_oran}", border=1, fill=True, align='C')
                    pdf.set_text_color(200, 30, 60)
                    pdf.cell(49, 9, txt=f"%{neg_oran}", border=1, fill=True, align='C')
                    pdf.set_text_color(10, 13, 36)
                    pdf.ln(12)

                    # ── 3. MODEL PERFORMANSI (F1 — sadece etiket varsa) ──────────
                    if etiket_sutun_adi:
                        bolum_baslik(pdf, 3, "MODEL PERFORMANS METRİKLERİ", font_tipi)

                        pdf.set_font(font_tipi, 'B', 10)
                        pdf.set_fill_color(218, 228, 248)
                        pdf.set_text_color(10, 13, 36)
                        for bsh, gn in [("Metrik", 95), ("Doğruluk / F1 Skoru", 95)]:
                            pdf.cell(gn, 8, txt=bsh, border=1, fill=True, align='C')
                        pdf.ln()

                        pdf.set_font(font_tipi, '', 10)
                        pdf.set_fill_color(249, 250, 252)
                        satirlar = [
                            ("Genel Doğruluk (Accuracy)", f"%{dogruluk_orani:.1f}"),
                            ("Pozitif Sınıf — F1 Skoru",  f"%{f1_pozitif:.1f}"),
                            ("Negatif Sınıf — F1 Skoru",  f"%{f1_negatif:.1f}"),
                            ("Macro Ortalama — F1 Skoru", f"%{f1_macro:.1f}"),
                        ]
                        for s_adi, s_deger in satirlar:
                            pdf.cell(95, 8, txt=f"  {s_adi}", border=1, fill=True)
                            pdf.cell(95, 8, txt=s_deger, border=1, fill=True, align='C')
                            pdf.ln()
                        pdf.ln(6)
                        sonraki_bolum = 4
                    else:
                        sonraki_bolum = 3

                    # ── 4. KÖK NEDEN ANALİZİ ─────────────────────────────────────
                    bolum_baslik(pdf, sonraki_bolum, "KÖK NEDEN ANALİZİ — İlk 5 Risk Kategorisi", font_tipi)

                    df_neg_pdf = df_sonuc[df_sonuc['Duygu_Etiketi'] == 'Negatif']
                    kat_sayilari_pdf = df_neg_pdf['Kategori'].value_counts().head(5)

                    pdf.set_font(font_tipi, 'B', 10)
                    pdf.set_fill_color(218, 228, 248)
                    pdf.set_text_color(10, 13, 36)
                    pdf.cell(130, 8, txt="Kategori", border=1, fill=True)
                    pdf.cell(30,  8, txt="Şikayet", border=1, fill=True, align='C')
                    pdf.cell(30,  8, txt="Oran",    border=1, fill=True, align='C')
                    pdf.ln()

                    pdf.set_font(font_tipi, '', 10)
                    toplam_neg = len(df_neg_pdf) if len(df_neg_pdf) > 0 else 1
                    for sira, (kat, sayi) in enumerate(kat_sayilari_pdf.items()):
                        pdf.set_fill_color(249 if sira % 2 == 0 else 242, 250, 252)
                        pdf.cell(130, 8, txt=f"  {kat}", border=1, fill=True)
                        pdf.cell(30,  8, txt=str(sayi), border=1, fill=True, align='C')
                        pdf.cell(30,  8, txt=f"%{round(sayi/toplam_neg*100,1)}", border=1, fill=True, align='C')
                        pdf.ln()
                    pdf.ln(8)

                    # ── 5. AKSİYON ÖNERİLERİ ─────────────────────────────────────
                    bolum_baslik(pdf, sonraki_bolum + 1, "STRATEJİK AKSİYON ÖNERİLERİ", font_tipi)

                    aksiyon_metinleri = {
                        'Kargo':            "Kargo şikayetleri baskın kategoride. Lojistik partner SLA'ları gözden geçirilmeli; teslimat takip sistemi güçlendirilmeli ve geç teslimat bildirim mekanizması devreye alınmalıdır.",
                        'Ürün Kalitesi':    "Ürün kalite şikayetleri kritik seviyede. Tedarikçi denetim süreçleri sıkılaştırılmalı, iade/değişim politikaları müşteri lehine revize edilmeli ve kalite kontrol checklistleri güncellenmeli.",
                        'Fiyat/Performans': "Fiyat-performans algısı zayıf. Ürün açıklamaları ve görsel içerik iyileştirilmeli; beklenti yönetimi güçlendirilmeli ve değer önerisini destekleyen kampanyalar planlanmalıdır.",
                        'Paketleme':        "Paketleme kaynaklı şikayetler tespit edildi. Kırılgan ürün protokolleri gözden geçirilmeli, ambalaj kalite standartları güncellenmeli ve kargo hasarı sigorta kapsamı değerlendirilmelidir.",
                        'Genel':            "Kategori dışı genel şikayetler yüksek. Müşteri hizmetleri ilk temas çözüm (FCR) oranı artırılmalı; self-servis destek içerikleri zenginleştirilmelidir.",
                    }
                    ak_metin_pdf = aksiyon_metinleri.get(en_sorunlu_pdf_adi, aksiyon_metinleri['Genel'])

                    pdf.set_font(font_tipi, 'B', 10)
                    pdf.set_text_color(10, 13, 36)
                    pdf.cell(0, 7, txt=f"  Öncelikli Alan: {en_sorunlu_pdf_adi}", ln=True)
                    bilgi_kutusu(pdf, ak_metin_pdf, font_tipi, bg=(240, 244, 255), tc=(30, 50, 120))

                    stratejik = (f"Analiz edilen {len(df_sonuc):,} yorumun %{neg_oran}'i negatif içeriklidir. "
                                 f"Riskli müşteri segmentine yönelik proaktif geri kazanım iletişimi "
                                 f"(e-posta/SMS) başlatılması ve periyodik duygu analizi raporlamasının "
                                 f"kurumsal KPI'lara eklenmesi önerilir.")
                    pdf.set_font(font_tipi, 'B', 10)
                    pdf.set_text_color(10, 13, 36)
                    pdf.cell(0, 7, txt="  Stratejik Önlem:", ln=True)
                    bilgi_kutusu(pdf, stratejik, font_tipi, bg=(240, 244, 255), tc=(30, 50, 120))

                    # ── 6. ÖRNEK VAKALAR ──────────────────────────────────────────
                    bolum_baslik(pdf, sonraki_bolum + 2, "ÖRNEK VAKALAR — Yapay Zeka Tarafından İşaretlendi", font_tipi)

                    ornek_pdf = df_neg_pdf.sort_values(by='Karar_Skoru', ascending=True).head(3)
                    pdf.set_font(font_tipi, 'I', 9)
                    for idx, row in ornek_pdf.iterrows():
                        temiz_yorum = str(row[aktif_yorum_sutunu])[:140] + "..."
                        kat_pdf = str(row['Kategori'])
                        pdf.set_fill_color(254, 242, 242)
                        pdf.set_text_color(160, 30, 50)
                        pdf.multi_cell(0, 6, txt=f"[{kat_pdf}]  \"{temiz_yorum}\"", fill=True, border=0)
                        pdf.ln(3)

                    # ── FOOTER ────────────────────────────────────────────────────
                    pdf.set_y(-18)
                    pdf.set_draw_color(200, 210, 230)
                    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
                    pdf.set_font(font_tipi, 'I', 8)
                    pdf.set_text_color(160, 170, 190)
                    pdf.cell(0, 10, txt="Bu belge Sentix AI Core Engine tarafından otomatik üretilmiştir. Gizli ve şirket içi kullanıma özgüdür.", align='C')

                    # ── ÇIKTI ─────────────────────────────────────────────────────
                    pdf_out = pdf.output(dest='S')
                    pdf_byte = pdf_out.encode('latin-1') if isinstance(pdf_out, str) else bytes(pdf_out)

                    st.download_button(
                        label="✦ Yönetici Özeti (PDF) İndir",
                        data=pdf_byte,
                        file_name="sentix_yonetici_ozeti.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    
                with buton_kolon2:
                    csv = df_indir.to_csv(index=False).encode('utf-8')
                    
                    # 📊 Emojisi tabloyu andıran ⊞ sembolü ile değiştirildi
                    st.download_button(
                        label="⊞ Ham Veriyi (CSV) İndir",
                        data=csv,
                        file_name='sentix_ai_analiz_raporu.csv',
                        mime='text/csv',
                        use_container_width=True
                    )
        except Exception as e:
            st.error(f"Bir hata oluştu: {e}")
    st.markdown("""
        <style>
            [data-testid="stSidebarResizer"],
            [data-testid="stSidebar"] > div:last-child {
                display: none !important;
                visibility: hidden !important;
                width: 0px !important;
                opacity: 0 !important;
                pointer-events: none !important;
            }
        </style>
    """, unsafe_allow_html=True)