
import os
import streamlit as st
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# ==============================================================================
# 1. AYARLAR BÖLÜMÜ
# ==============================================================================
# Bu sürümde API_KEY YOKTUR çünkü her şey yerel çalışır.

VERILER_KLASORU = "veriler"          # Dokümanların bulunduğu klasör adı
CHUNK_SIZE = 1000                    # Her parçanın karakter uzunluğu
CHUNK_OVERLAP = 150                  # Parçalar arası örtüşme miktarı
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Ücretsiz, yerel embedding modeli
LLM_MODEL = "google/flan-t5-base"    # Ücretsiz, yerel çalışan cevap üretme modeli
TOP_K = 4                            # Soruya cevap için getirilecek en alakalı parça sayısı
BENZERLIK_ESIGI = 0.45               # Bu değerden düşük benzerlikte sonuç "konu dışı" sayılır

# ==============================================================================
# 2. SAYFA YAPILANDIRMASI
# ==============================================================================

st.set_page_config(
    page_title="Doküman Tabanlı Yapay Zeka Asistanı",
    page_icon="📚",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        background-color: #f7f9fc;
    }
    .source-box {
        background-color: #eef3fb;
        border-left: 4px solid #4a7dfc;
        padding: 10px 15px;
        border-radius: 8px;
        margin-top: 10px;
        font-size: 0.9em;
    }
    </style>
""", unsafe_allow_html=True)


# ==============================================================================
# 3. YARDIMCI FONKSİYONLAR
# ==============================================================================

def dokumanlari_yukle(klasor_yolu: str):
    """
    Belirtilen klasördeki .txt ve .pdf dosyalarını okur.
    Her doküman parçasına hangi dosyadan geldiği bilgisini (kaynak) ekler.
    """
    yuklenen_dokumanlar = []
    durum_listesi = []

    if not os.path.exists(klasor_yolu):
        return [], []

    dosyalar = sorted(os.listdir(klasor_yolu))
    desteklenen_dosyalar = [f for f in dosyalar if f.lower().endswith((".txt", ".pdf"))]

    for dosya_adi in desteklenen_dosyalar:
        dosya_yolu = os.path.join(klasor_yolu, dosya_adi)
        try:
            if dosya_adi.lower().endswith(".txt"):
                loader = TextLoader(dosya_yolu, encoding="utf-8")
                belgeler = loader.load()
            else:
                loader = PyPDFLoader(dosya_yolu)
                belgeler = loader.load()

            for belge in belgeler:
                belge.metadata["source"] = dosya_adi

            yuklenen_dokumanlar.extend(belgeler)
            durum_listesi.append((dosya_adi, True, f"{len(belgeler)} sayfa/blok okundu"))

        except Exception as e:
            durum_listesi.append((dosya_adi, False, str(e)))

    return yuklenen_dokumanlar, durum_listesi


@st.cache_resource(show_spinner=False)
def vektor_veritabani_olustur(klasor_yolu: str):
    """
    Dokümanları yükler, parçalara ayırır ve FAISS vektör veritabanı oluşturur.
    """
    belgeler, durum_listesi = dokumanlari_yukle(klasor_yolu)

    if not belgeler:
        return None, durum_listesi

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    parcalar = text_splitter.split_documents(belgeler)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vektor_db = FAISS.from_documents(parcalar, embeddings)

    return vektor_db, durum_listesi


@st.cache_resource(show_spinner=False)
def llm_yukle():
    """
    Yerel çalışan, ücretsiz HuggingFace text2text-generation modelini yükler.
    İlk çalıştırmada internetten indirilir, sonraki çalıştırmalarda yerel
    HuggingFace önbelleğinden (cache) okunur, internet gerekmez.
    """
    return pipeline("text2text-generation", model=LLM_MODEL, max_new_tokens=300)


def yerel_llm_cevap_uret(soru: str, ilgili_parcalar: list, llm) -> str:
    """
    Getirilen doküman parçalarını bağlam (context) olarak kullanarak yerel
    modelle cevap üretir. Model talimatlara her zaman tam uymayabileceğinden,
    asıl "bilgi yok" kontrolü benzerlik skoruna göre yapılır (bkz: ana akış).
    """
    baglam = "\n".join([p.page_content for p in ilgili_parcalar])
    # Flan-T5 tipi modeller kısa ve net talimatlarla daha iyi çalışır.
    prompt = (
        f"Aşağıdaki metne dayanarak soruyu yanıtla. "
        f"Cevap metinde yoksa 'bilinmiyor' yaz.\n\n"
        f"Metin: {baglam}\n\n"
        f"Soru: {soru}\n"
        f"Cevap:"
    )
    try:
        sonuc = llm(prompt)
        return sonuc[0]["generated_text"].strip()
    except Exception as e:
        return f"⚠️ Yerel model çalıştırılırken hata oluştu: {e}"


# ==============================================================================
# 4. OTURUM DURUMU (SESSION STATE) BAŞLATMA
# ==============================================================================

if "mesajlar" not in st.session_state:
    st.session_state.mesajlar = []

if "vektor_db" not in st.session_state:
    st.session_state.vektor_db = None


# ==============================================================================
# 5. SOL PANEL (SIDEBAR) - DOKÜMAN DURUMU
# ==============================================================================

with st.sidebar:
    st.markdown("## 📁 Doküman Durumu")
    st.markdown(f"Kaynak klasör: `{VERILER_KLASORU}/`")
    st.caption("Bu uygulama %100 yerel çalışır, API anahtarı gerekmez.")
    st.divider()

    if st.button("🔄 Dokümanları Yeniden Tara", use_container_width=True):
        st.cache_resource.clear()
        st.session_state.vektor_db = None
        st.rerun()

    with st.spinner("Dokümanlar işleniyor, lütfen bekleyin..."):
        vektor_db, durum_listesi = vektor_veritabani_olustur(VERILER_KLASORU)
        st.session_state.vektor_db = vektor_db

    with st.spinner("Yerel dil modeli yükleniyor (ilk seferde biraz sürebilir)..."):
        llm = llm_yukle()

    st.divider()

    if not os.path.exists(VERILER_KLASORU):
        st.error(f"'{VERILER_KLASORU}' klasörü bulunamadı! Lütfen app.py ile aynı dizinde oluşturun.")
    elif not durum_listesi:
        st.warning("Klasörde desteklenen (.txt / .pdf) doküman bulunamadı.")
    else:
        basarili_sayisi = sum(1 for _, basarili, _ in durum_listesi if basarili)
        st.markdown(f"**Toplam doküman:** {len(durum_listesi)}")
        st.markdown(f"**Başarıyla yüklenen:** {basarili_sayisi}")
        st.markdown("---")

        for dosya_adi, basarili, mesaj in durum_listesi:
            if basarili:
                st.markdown(f"✅ **{dosya_adi}**")
                st.caption(mesaj)
            else:
                st.markdown(f"❌ **{dosya_adi}**")
                st.caption(f"Hata: {mesaj}")

    st.divider()
    st.caption("💡 İpucu: 5 adet .txt veya .pdf dosyanızı 'veriler' klasörüne koyup uygulamayı yenileyin.")
    st.caption(f"🧠 Kullanılan yerel model: {LLM_MODEL}")


# ==============================================================================
# 6. ORTA ALAN - SOHBET ARAYÜZÜ
# ==============================================================================

st.markdown("# 📚 Doküman Tabanlı Yapay Zeka Asistanı")
st.markdown("Yüklenen dokümanlara dayalı sorularınızı aşağıya yazabilirsiniz. *(Tamamen yerel, ücretsiz çalışır)*")
st.divider()

for soru, cevap, kaynaklar in st.session_state.mesajlar:
    with st.chat_message("user"):
        st.markdown(soru)
    with st.chat_message("assistant"):
        st.markdown(cevap)
        if kaynaklar:
            kaynak_metni = ", ".join(sorted(set(kaynaklar)))
            st.markdown(
                f"<div class='source-box'>📄 <b>Kaynak:</b> {kaynak_metni}</div>",
                unsafe_allow_html=True
            )

soru = st.chat_input("Sorunuzu buraya yazın...")

BILGI_YOK_MESAJI = "Bu bilgi yüklenen kaynak dokümanlarda bulunmamaktadır."

if soru:
    with st.chat_message("user"):
        st.markdown(soru)

    with st.chat_message("assistant"):
        if st.session_state.vektor_db is None:
            cevap = "Şu anda işlenmiş bir doküman bulunmuyor. Lütfen 'veriler' klasörüne .txt veya .pdf dosyalarınızı ekleyip uygulamayı yenileyin."
            st.warning(cevap)
            st.session_state.mesajlar.append((soru, cevap, []))
        else:
            with st.spinner("İlgili doküman parçaları aranıyor..."):
                # Skorlu arama: düşük skor (FAISS'te düşük = daha benzer, L2 distance)
                sonuclar = st.session_state.vektor_db.similarity_search_with_relevance_scores(soru, k=TOP_K)

            # En iyi eşleşmenin alaka skoru çok düşükse, dokümanlarda bilgi yok diyoruz.
            en_iyi_skor = max([skor for _, skor in sonuclar]) if sonuclar else 0

            if not sonuclar or en_iyi_skor < BENZERLIK_ESIGI:
                cevap = BILGI_YOK_MESAJI
                st.warning(cevap)
                st.session_state.mesajlar.append((soru, cevap, []))
            else:
                ilgili_parcalar = [doc for doc, skor in sonuclar]

                with st.spinner("Cevap oluşturuluyor (yerel model)..."):
                    cevap = yerel_llm_cevap_uret(soru, ilgili_parcalar, llm)

                # Yerel model bazen kendisi de "bilinmiyor" diyebilir, bu durumu yakalıyoruz.
                if not cevap or "bilinmiyor" in cevap.lower() or len(cevap.strip()) < 2:
                    cevap = BILGI_YOK_MESAJI
                    st.warning(cevap)
                    st.session_state.mesajlar.append((soru, cevap, []))
                else:
                    st.markdown(cevap)
                    kaynaklar = [p.metadata.get("source", "bilinmiyor") for p in ilgili_parcalar]
                    kaynak_metni = ", ".join(sorted(set(kaynaklar)))
                    st.markdown(
                        f"<div class='source-box'>📄 <b>Kaynak:</b> {kaynak_metni}</div>",
                        unsafe_allow_html=True
                    )
                    st.session_state.mesajlar.append((soru, cevap, kaynaklar))
