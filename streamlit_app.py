import streamlit as st
import pandas as pd
import calendar
from datetime import datetime, timedelta
import os
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# ─── CONFIG ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="เสียงในใจ — Mood Diary", layout="wide")
DATA_FILE = "diary_records.csv"
EMOJI_MAP = {
    "pos": "😊",
    "neu": "😐",
    "neg": "😢",
}
EMOJI_OPTIONS = list(EMOJI_MAP.values())

st.markdown(
    """
    <style>
        .stApp {
            background-color: #5AC8B8; /* สีเขียวมิ้นฟ้าเข้มๆ เหมือนไอติม */
        }
        .block-container {
            background-color: #A0E1D7; /* สีอ่อนตัดกัน */
            padding: 2rem;
            border-radius: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)





# ─── LOAD MODEL ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    model_name = "phoner45/wangchan-sentiment-thai-text-model"
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("text-classification", model=mdl, tokenizer=tok)

sentiment_pipe = load_pipeline()

# ─── HELPERS ────────────────────────────────────────────────────────────────────
def analyze_sentiment(text: str):
    r = sentiment_pipe(text)[0]
    return r["label"], r["score"]

def load_data():
    if not os.path.exists(DATA_FILE):
        pd.DataFrame(columns=["date","text","sentiment","score","emoji"]) \
          .to_csv(DATA_FILE, index=False)
    df = pd.read_csv(DATA_FILE)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    df["emoji"] = df["sentiment"].map(EMOJI_MAP).fillna("")
    return df

def save_entry(date, text, sentiment, score, emoji=None):
    df = load_data().reset_index()
    if emoji is None:
        emoji = EMOJI_MAP.get(sentiment, "")
    same_day = df["date"].dt.date == date
    if same_day.any():
        df.loc[same_day, ["text","sentiment","score","emoji"]] = [
            text, sentiment, score, emoji
        ]
    else:
        df.loc[len(df)] = {
            "date":      date,
            "text":      text,
            "sentiment": sentiment,
            "score":     score,
            "emoji":     emoji
        }
    df.to_csv(DATA_FILE, index=False)

# ─── LAYOUT ────────────────────────────────────────────────────────────────────
st.title("🧠 เสียงในใจ — Mood Diary w/ Edit Emoji")

col1, col2 = st.columns([1,2])

# Left: Entry
with col1:
    entry_date = st.date_input("วันที่", datetime.now().date())
    diary_text = st.text_area("บันทึกความรู้สึก…", height=200)
    if st.button("💾 บันทึกและวิเคราะห์"):
        if diary_text.strip():
            sent, score = analyze_sentiment(diary_text)
            save_entry(entry_date, diary_text, sent, score)
            st.success(f"🔔 {EMOJI_MAP[sent]} บันทึกเรียบร้อย! ({sent.upper()} {score:.0%})")

             
              # 🔸 ข้อความให้กำลังใจถ้าเศร้า
        if sent == "neg":
            st.info("⚡ วันนี้อาจจะไม่ดีนัก ลองพักผ่อน ฟังเพลงที่ชอบ หรือคุยกับเพื่อนดูนะครับ 💙")
        else:
            st.error("กรุณาใส่ข้อความก่อนบันทึก")

# Right: Tabs
with col2:
    df = load_data()
    if df.empty:
        st.info("ยังไม่มีบันทึก ลองเพิ่มไดอารี่ก่อน")
    else:
        tab1, tab2, tab3 = st.tabs(["Summary","Calendar","Stats"])
        
        st.markdown("---")
        st.download_button(
        label="📥 ดาวน์โหลดบันทึกทั้งหมด (CSV)",
        data=df.reset_index().to_csv(index=False).encode('utf-8'),
        file_name='mood_diary.csv',
         mime='text/csv'
)


        # Summary w/ edit
        with tab1:
            st.subheader("บันทึกล่าสุด")
            recent = df.reset_index().tail(5)[["date","text","emoji","score"]]
            recent["score"] = recent["score"].apply(lambda x: f"{x:.0%}")
            st.table(recent)

            st.markdown("---")
            st.subheader("✏️ แก้ไขไดอารีย้อนหลัง & Emoji")
            edit_date = st.date_input(
                "เลือกวันที่จะแก้ไข",
                value=recent["date"].iloc[-1].date(),
                min_value=recent["date"].min().date(),
                max_value=recent["date"].max().date()
            )
            # load old
            old_row = df.loc[pd.to_datetime(edit_date)]
            new_text = st.text_area("ข้อความใหม่", value=old_row["text"], height=150)
            # sentiment override?
            new_sent = st.selectbox(
                "อัปเดต Sentiment",
                ["pos","neu","neg"],
                index=["pos","neu","neg"].index(old_row["sentiment"])
            )
            # emoji override
            new_emoji = st.selectbox(
                "อัปเดต Emoji",
                EMOJI_OPTIONS,
                index=EMOJI_OPTIONS.index(old_row["emoji"]) if old_row["emoji"] in EMOJI_OPTIONS else 0
            )
            if st.button("💾 บันทึกการแก้ไข"):
                # if sentiment changed, you might re-run analysis or keep override
                # here we trust new_sent and new_emoji
                _, new_score = analyze_sentiment(new_text)
                save_entry(edit_date, new_text, new_sent, new_score, emoji=new_emoji)
                st.success(f"{new_emoji} อัปเดตเรียบร้อย! ({new_sent.upper()} {new_score:.0%})")
                st.rerun()
            
            if st.button("🗑️ ลบบันทึกนี้"):
                df = df.reset_index()
                df = df[df["date"] != pd.to_datetime(edit_date)]
                df.to_csv(DATA_FILE, index=False)
                st.success("ลบบันทึกเรียบร้อยแล้ว")
                st.rerun()


        # Calendar
        with tab2:
            st.subheader("📅 ปฏิทิน Mood")
            month = st.selectbox("เดือน", list(range(1,13)), index=datetime.now().month-1)
            year  = st.number_input("ปี", 2000, 2100, datetime.now().year)
            cal = calendar.monthcalendar(year, month)
            table = []
            for week in cal:
                row = []
                for d in week:
                    if d == 0:
                        row.append("")
                    else:
                        dt = datetime(year, month, d)
                        row.append(df.loc[dt,"emoji"] if dt in df.index else "")
                table.append(row)
            st.table(pd.DataFrame(table, columns=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]))

        # Stats
        # ─── Stats ─────────────────────────────────────────────────────────────────────
        with tab3:
            st.subheader("📊 Mood & Sentiment (7 วันล่าสุด)")
            cutoff = datetime.now() - timedelta(days=7)
            recent7 = df[df.index >= cutoff]

            # กราฟจำนวนของ Emoji และ Sentiment
            st.bar_chart(recent7["emoji"].value_counts())
            st.bar_chart(recent7["sentiment"].value_counts())

            # ค่าเฉลี่ยคะแนนความรู้สึก
            if not recent7.empty:
                avg_score = recent7["score"].mean()
                st.metric("🎯 คะแนนความรู้สึกเฉลี่ย", f"{avg_score:.0%}")

                # อารมณ์ที่เกิดบ่อยที่สุด
                most_common = recent7["sentiment"].value_counts().idxmax()
                st.info(f"อารมณ์ที่เจอบ่อยที่สุดช่วงนี้คือ: **{EMOJI_MAP[most_common]} {most_common.upper()}**")

                # กราฟแนวโน้มคะแนนอารมณ์
                st.line_chart(recent7["score"])

                # ข้อความช่วง 7 วัน
                st.markdown("### 📅 ข้อความในช่วง 7 วันล่าสุด")
                for dt, row in recent7.iterrows():
                    st.markdown(
                        f"- **{dt.date()}** {row['emoji']} ({row['sentiment'].upper()} {row['score']:.0%}) → {row['text']}"
                    )
            else:
                st.warning("ยังไม่มีบันทึกในช่วง 7 วันที่ผ่านมา")