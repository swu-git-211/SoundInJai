import streamlit as st
import pandas as pd
from datetime import datetime
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# ─── PAGE CONFIG ────────────────────────────
st.set_page_config(page_title="เสียงในใจ", layout="wide")

# ─── LOAD PIPELINE ──────────────────────────
@st.cache_resource
def load_thai_pipeline():
    model_name = "phoner45/wangchan-sentiment-thai-text-model"
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("text-classification", model=mdl, tokenizer=tok)

sentiment_pipe = load_thai_pipeline()

# ─── HELPERS ────────────────────────────────
DATA_FILE = "diary_records.csv"

def analyze_sentiment(text: str):
    res = sentiment_pipe(text)[0]
    return res["label"], res["score"]

def load_data():
    # ถ้าไฟล์ยังไม่มีให้สร้าง header เปล่าๆ
    if not os.path.exists(DATA_FILE):
        pd.DataFrame(columns=["date","text","sentiment","score"]) \
          .to_csv(DATA_FILE, index=False)
    # อ่านแล้วแปลง date เป็น datetime, ตั้งเป็น index
    df = pd.read_csv(DATA_FILE)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")         # แปลงไม่แมตช์เป็น NaT
    df = df.dropna(subset=["date"])                                 # ตัด row ที่ date ผิด
    df = df.set_index("date").sort_index()                          # สร้าง DatetimeIndex
    return df

def save_entry(date, text, sentiment, score):
    df = load_data().reset_index()  # คืนเป็น DataFrame ปกติก่อน append
    df.loc[len(df)] = {
        "date": date, "text": text,
        "sentiment": sentiment, "score": score
    }
    df.to_csv(DATA_FILE, index=False)

# ─── STREAMLIT LAYOUT ───────────────────────
st.title("🧠 เสียงในใจ — Diary Sentiment Tracker")
col1, col2 = st.columns([1,2])

with col1:
    entry_date = st.date_input("วันที่", datetime.now().date())
    diary_text  = st.text_area("บันทึกความรู้สึก…", height=200)
    if st.button("💾 บันทึกและวิเคราะห์"):
        if diary_text.strip():
            label, score = analyze_sentiment(diary_text)
            save_entry(entry_date, diary_text, label, score)
            st.success(f"ผลการวิเคราะห์: **{label}** ({score:.0%})")
        else:
            st.error("กรุณาใส่ข้อความก่อน")

with col2:
    df = load_data()
    if df.empty:
        st.info("ยังไม่มีข้อมูล ลองเพิ่มไดอารี่แล้วกลับมาดูกราฟ")
    else:
        # ตอนนี้ df.index คือ DatetimeIndex แล้ว
        weekly = df["score"].resample("W-MON").mean().rename("avg_score")
        st.subheader("แนวโน้มคะแนนอารมณ์เฉลี่ยรายสัปดาห์")
        st.line_chart(weekly)

        st.subheader("บันทึกล่าสุด")
        recent = df.reset_index().tail(5)[["date","text","sentiment","score"]]
        recent["score"] = recent["score"].apply(lambda x: f"{x:.0%}")
        st.table(recent)

        st.subheader("📌 คำแนะนำสำหรับคุณ")
        last = df.iloc[-1]
        if last["sentiment"]=="NEGATIVE" and last["score"]>=0.6:
            st.info("ลองฟังเพลงสบาย ๆ เพื่อผ่อนคลาย")
        elif last["sentiment"]=="POSITIVE":
            st.success("จด gratitude list เพื่อเสริมความสุข")
        else:
            st.warning("ลองเดินเล่นสั้น ๆ ในธรรมชาติ")
