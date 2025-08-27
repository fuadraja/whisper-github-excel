import os, tempfile
from pathlib import Path
import streamlit as st
from faster_whisper import WhisperModel

st.set_page_config(page_title="تفريغ الصوت", page_icon="🎧")
st.title("🎧 تفريغ صوت بالعربية (مرحلة ب)")

SUPPORTED = {".m4a",".mp3",".wav",".mp4",".aac",".flac",".ogg"}
model_size = st.selectbox("اختر حجم النموذج", ["tiny","base","small","medium"], index=2)

@st.cache_resource(show_spinner=False)
def load_model(size):
    return WhisperModel(size, device="cpu")

uploaded = st.file_uploader("ارفع ملفًا صوتيًا", type=[e[1:] for e in sorted(SUPPORTED)])

if st.button("تفريغ الآن", type="primary"):
    if not uploaded:
        st.error("ارفع ملفًا أولاً."); st.stop()
    ext = Path(uploaded.name).suffix.lower()
    if ext not in SUPPORTED:
        st.error(f"صيغة غير مدعومة: {ext}"); st.stop()

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(uploaded.getbuffer()); tmp_path = tmp.name

    try:
        with st.spinner("تحميل النموذج…"):
            model = load_model(model_size)
        with st.spinner("جاري التفريغ (ar)…"):
            segments, info = model.transcribe(tmp_path, language="ar")
            text = " ".join(seg.text for seg in segments).strip()
        st.success("تم التفريغ ✅")
        st.text_area("النص:", text, height=200)
    except Exception as e:
        st.error(f"خطأ: {e}")
    finally:
        try: os.remove(tmp_path)
        except: pass

st.caption("هذه مرحلة تجريبية بدون حفظ إلى GitHub بعد.")
