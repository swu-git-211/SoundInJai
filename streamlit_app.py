import streamlit as st
import pandas as pd
import calendar
from datetime import datetime, timedelta
import os
import uuid
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


# ─── CONFIG ─────────────────────────────────────────────────────────
st.set_page_config(page_title="เสียงในใจ — Diary", layout="wide")
DATA_FILE = "diary_records.csv"
EMOJI_MAP = {"pos": "😊", "neu": "😐", "neg": "😢"}

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300&display=swap');

        html, body, [class*="css"] {
            font-family: 'Kanit', sans-serif;
            background-color: #fff0f5;
        }

        .block-container {
            padding: 2rem 2rem 2rem;
        }

        h1, h2, h3 {
            color: #d63384;
        }

        .stButton button {
            background-color: #f7c3e0;
            color: black;
            border-radius: 12px;
            border: none;
            padding: 8px 20px;
            font-weight: bold;
        }

        .stButton button:hover {
            background-color: #ffcfe0;
        }

        .stTabs [role="tab"] {
            font-weight: bold;
            padding: 10px 20px;
        }

        .stTabs [aria-selected="true"] {
            background: #fce4ec;
            border-bottom: 3px solid #d63384;
        }
    </style>
""", unsafe_allow_html=True)



# ─── MODEL ──────────────────────────────────────────────────────────
@st.cache_resource
def load_pipe():
    model_name = "phoner45/wangchan-sentiment-thai-text-model"
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("text-classification", model=mdl, tokenizer=tok)

sentiment_pipe = load_pipe()

def analyze_sentiment(text: str):
    out = sentiment_pipe(text)[0]
    label = out["label"].lower()
    if label.startswith("pos"):
        label = "pos"
    elif label.startswith("neg"):
        label = "neg"
    else:
        label = "neu"
    return label, out["score"]

def suggest_message(sentiment, score):
    suggestions = {
        "pos": [
            "วันนี้คุณดูสดใสมาก! 🌟 ลองแบ่งปันรอยยิ้มให้คนรอบข้างดูสิ",
            "รักษาความรู้สึกดี ๆ แบบนี้ไว้นาน ๆ นะ 😊",
            "เยี่ยมเลย! เก็บโมเมนต์ดี ๆ ไว้ในใจ ❤️"
        ],
        "neu": [
            "วันกลาง ๆ ก็โอเคนะ ลองทำสิ่งใหม่ ๆ ดูไหม?",
            "ลองเขียนหาอะไรทำดูสิ เช่นเล่นเกม ดูหนัง อาจทำให้รู้สึกดีขึ้น",
            "อารมณ์นิ่ง ๆ แบบนี้ ลองฟังเพลงชิล ๆ ก็ไม่เลวนะ"
        ],
        "neg": [
            "คุณเก่งมาก วันนี้พยายามได้ดีมากได้เวลาพักผ่อนนน อย่าลืมฟังเพลงโปรดก่อนนอนละ",
            "ส่งกำลังใจให้คุณผ่านวันนี้ไปได้ ✨",
            "อย่าลืมหายใจลึก ๆ แล้วค่อย ๆ ก้าวต่อไปนะ 💛"
        ]
    }
    return random.choice(suggestions[sentiment])

# ─── DATA ──────────────────────────────────────────────────────────
def load_data():
    if not os.path.exists(DATA_FILE):
        pd.DataFrame(columns=["id", "date", "text", "sentiment", "score", "emoji"]) \
            .to_csv(DATA_FILE, index=False)

    df = pd.read_csv(DATA_FILE)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)
    if "id" not in df.columns:
        df["id"] = [str(uuid.uuid4()) for _ in range(len(df))]
    else:
        df["id"] = df["id"].fillna("").apply(lambda x: str(uuid.uuid4()) if x == "" else x)
    return df

def save_entry(date, text, sentiment, score, emoji):
    df = load_data()
    mask = df["date"] == date
    if mask.any():
        idx = df[mask].index[0]
        df.loc[idx, ["text", "sentiment", "score", "emoji"]] = [text, sentiment, score, emoji]
    else:
        df.loc[len(df)] = {
            "id": str(uuid.uuid4()),
            "date": date,
            "text": text,
            "sentiment": sentiment,
            "score": score,
            "emoji": emoji
        }
    df.to_csv(DATA_FILE, index=False)

def delete_entry(eid):
    df = load_data()
    df = df[df["id"] != eid]
    df.to_csv(DATA_FILE, index=False)

if st.query_params.get("scroll") == "edit":
    st.write('<script>window.scrollTo(0, document.body.scrollHeight);</script>', unsafe_allow_html=True)
    st.query_params.clear()  # reset query params

def toggle_edit(rid):
    if st.session_state.get("edit_id") == rid:
        st.session_state.edit_id = None
    else:
        st.session_state.edit_id = rid
        st.query_params["scroll"] = "edit"

# ─── UI ─────────────────────────────────────────────────────────────

import calendar
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import date, datetime, timedelta
import streamlit as st
import streamlit.components.v1 as components

# Title
st.markdown("""
<h1 style="text-align;">
  <span class="emoji">🌸</span>
  <span class="vertical-gradient-text">SoundInJai — Diary づ❤︎ど</span>
</h1>
""", unsafe_allow_html=True)



# Custom UI Styling
st.markdown("""
<style>
    .vertical-gradient-text {
        background: linear-gradient(to bottom, #f78fb3, #a29bfe);  /* ชมพู ➜ ม่วง */
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 40px;
    }

    .emoji {
        font-size: 40px;
        vertical-align: middle;
        margin-right: 0.3rem;
    }
    
    /* พื้นหลังไล่สีพาสเทลนุ่มๆ */
    .stApp {
        background: linear-gradient(to right, #fceefc, #e0f7fa);
        background-attachment: fixed;
        font-family: 'Kanit', sans-serif;
        color: #d63384;
    }

    .block-container {
        padding: 5rem;
    }

    /* หัวข้อทุกระดับสีชมพูอ่อน */
    h1, h2, h3, h4, h5, h6, p, span, label, div {
        color: #d63384 ;
    }
      
    /* ปุ่มสีชมพูพาสเทล */
    .stButton > button {
        background-color: #fbb6ce !important;
        color: white !important;
        border-radius: 14px;
        padding: 0.6rem 1.4rem;
        font-weight: bold;
        font-size: 16px;
        border: none;
        box-shadow: 2px 2px 8px rgba(251, 182, 206, 0.4);
        transition: background-color 0.3s ease;
    }

    .stButton > button:hover {
        background-color: #f687b3 !important;
    }

    /* กล่องใส่ข้อความ / input ต่าง ๆ */
    .stTextArea textarea,
    .stTextInput input,
    .stDateInput input {
        background-color: #fff0f5 !important;
        color: #b03060 !important;
        border-radius: 12px;
        border: 2px solid #fbb6ce !important;
        font-size: 16px;
        font-weight: 500;
        padding: 0.8rem;
        box-shadow: 0 2px 5px rgba(255, 182, 193, 0.25);
    }

    .stTextArea textarea {
        resize: none;
        min-height: 150px;
    }

    /* แท็บเมนู */
    .stTabs [role="tablist"] > div {
        background-color: pink;
        border-radius: 100px;
        padding: 0.4rem 1rem;
        color: #d63384 !important;
    }
    .stTabs [role="tablist"] > button:nth-child(1) {
        background-color: #d0b3ff !important; /* สีม่วงอ่อน */
    }
    .stTabs [role="tablist"] > button:nth-child(2) {
        background-color: #a0f0ed !important; /* สีมิ้น */
    }
    .stTabs [role="tablist"] > button:nth-child(3) {
        background-color: #fff3b0 !important; /* สีเหลืองอ่อน */
    }
            
    .summary-title {
        font-size: 35px;
        font-weight: bold;
        background: linear-gradient(to bottom, #a29bfe, #a0f0ed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 10px;
    }

    .emoji {
        font-size: 40px;
        margin-right: 8px;
        color: inherit;
        background: none !important;
        -webkit-background-clip: unset !important;
        -webkit-text-fill-color: initial !important;
    }

            
    /* กล่องแสดงข้อความแจ้งเตือน */
    .stAlert {
        background-color: #fff0f5;
        color: #b03060;
        border-radius: 12px;
        padding: 1rem;
        border-left: 6px solid #f687b3;
    }

    /* สไตล์สำหรับบันทึกแต่ละรายการ */
    .entry-container {
        background-color: #fff9fb;
        border-left: 6px solid #fbb6ce;
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.06);
    }

    .entry-date {
        font-size: 14px;
        color: #c71585;
        margin-bottom: 0.3rem;
    }

    .entry-text {
        font-size: 16px;
        color: #d63384;
        line-height: 1.5;
    }

    /* ปรับความกว้างของหน้าจอแสดงผลให้ไม่กว้างเกิน */
    .main .block-container {
        max-width: 900px;
        margin: auto;
    }
            
    /* ปรับสไตล์ของ input ปฏิทิน */
    .stDateInput input {
        background-color: #fff0f5 !important;
        color: #d63384 !important;
        border-radius: 12px !important;
        border: 2px solid #fbb6ce !important;
        font-weight: 500;
        font-size: 16px;
        box-shadow: 0px 2px 5px rgba(251, 182, 206, 0.25);
        padding: 0.6rem 1rem;
    }
                    
    /* ปรับขนาด input ปฏิทินให้เล็กลง */
    .stDateInput {
        max-width: 250px !important;
    }
            
    /* ปรับขนาดกล่องข้อความให้ไม่ยาวเกินไป */
    .stTextArea {
        max-width: 600px !important;
    }
            
    /* สีของคำ "สรุปค่าเฉลี่ยระดับความรู้สึก","แนวโน้ม Sentiment" */        
    .highlight-yellow {
        font-size: 20px;
        font-weight: bold;
        color: #F9A825 !important;  /* สีเหลืองพาสเทล */
        padding-bottom: 10px;
    }
    .high-score {
        color: #43a047 !important; /* เขียว */
    }
    .medium-score {
        color: #f9a825 !important; /* เหลือง */
    }
    .low-score {
        color: #e53935 !important; /* แดง */
    }
    .sentiment-pos {
        color: #4CAF50 !important; /* เขียว */
        font-weight: bold;
        font-size: 18px;
    }

    .sentiment-neu {
        color: #FFC107 !important; /* เหลือง */
        font-weight: bold;
        font-size: 18px;
    }

    .sentiment-neg {
        color: #F44336 !important; /* แดง */
        font-weight: bold;
        font-size: 18px;
    }            

</style>
""", unsafe_allow_html=True)

# Load data
df = load_data()
if "entry_date" not in st.session_state:
    st.session_state.entry_date = datetime.now().date()
if "entry_text" not in st.session_state:
    st.session_state.entry_text = ""

# ---- Diary Input Section ----

st.subheader("🌼 Welcome to Your Diary")
entry_date = st.date_input(
    "📅 Select Date",
    value=st.session_state.get("entry_date", datetime.now().date()),
    key="entry_date"
)
existing = df[df["date"] == entry_date]
default_text = existing.iloc[0]["text"] if not existing.empty else ""
entry_text = st.text_area("🌷 บันทึกความรู้สึกประจำวัน", value=st.session_state.get("entry_text", default_text), height=200, key="entry_text")

def on_new_save():
    if st.session_state.entry_text.strip():
        lab, sc = analyze_sentiment(st.session_state.entry_text)
        em = EMOJI_MAP[lab]
        save_entry(entry_date, st.session_state.entry_text, lab, sc, em)
        st.success(f"{em} บันทึกเรียบร้อย! ({lab.upper()} {sc:.0%})")
        st.info(f"💡 คำแนะนำวันนี้: {suggest_message(lab, sc)}")
        st.session_state.entry_text = ""
        st.session_state.entry_date = datetime.now().date()
    else:
        st.error("กรุณาใส่ข้อความก่อนบันทึก")

st.button("💾 บันทึกและวิเคราะห์", on_click=on_new_save)

# ---- Tabs Section (Moved below input) ----
st.markdown("---")

if df.empty:
    st.info("ยังไม่มีบันทึกเลย ลองเพิ่มดูสิ")
else:
    tab1, tab2, tab3 = st.tabs(["Summary", "Calendar", "Stats"])

    with tab1:
    
        st.markdown("""
        <h2>
            <span class="emoji">📝</span>
            <span class="summary-title">บันทึกย้อนหลัง</span>
        </h2>
        """, unsafe_allow_html=True)

        df2 = df.sort_values("date", ascending=False).reset_index(drop=True)
        if "edit_id" not in st.session_state:
            st.session_state.edit_id = None

        for _, row in df2.iterrows():
            c1, c2, c3, c4, c5, c6 = st.columns([1.3, 4, 1, 1, 1, 0.6])
            c1.write(str(row["date"]))
            c2.write(row["text"])
            c3.write(row["emoji"])
            
            if row["sentiment"] == "neg" and row["score"] > 0.7:
                color_class = "very-neg-score"
            elif row["score"] > 0.7:
                color_class = "high-score"
            elif row["score"] > 0.4:
                color_class = "medium-score"
            else:
                color_class = "low-score"
    
            c4.markdown(f"<div class='{color_class}'>{row['score']:.0%}</div>", unsafe_allow_html=True)

            sentiment = row["sentiment"]
            sentiment_class = (
                "sentiment-pos" if sentiment == "pos"
                else "sentiment-neu" if sentiment == "neu"
                else "sentiment-neg"
            )
            c5.markdown(f"<div class='{sentiment_class}'>{sentiment.upper()}</div>", unsafe_allow_html=True)
            c6.button("✏️", key=f"edit_{row['id']}", on_click=toggle_edit, args=(row["id"],))

        if st.session_state.edit_id:
            st.markdown("---")
            old = df[df["id"] == st.session_state.edit_id].iloc[0]
            st.subheader("🔄 แก้ไขบันทึกย้อนหลัง")
            new_text = st.text_area("ข้อความใหม่", old["text"], height=150)

            def on_apply_edit():
                lab, sc = analyze_sentiment(new_text)
                em = EMOJI_MAP[lab]
                save_entry(old["date"], new_text, lab, sc, em)
                st.success(f"{em} แก้ไขเรียบร้อย! ({lab.upper()} {sc:.0%})")
                st.info(f"💡 คำแนะนำวันนี้: {suggest_message(lab, sc)}")
                st.session_state.edit_id = None
                st.session_state.should_rerun = True

            st.button("💾 บันทึกการแก้ไข", on_click=on_apply_edit, key=f"save_{old['id']}")

            def on_apply_delete():
                delete_entry(old["id"])
                st.success("🗑️ ลบบันทึกเรียบร้อยแล้ว")
                st.session_state.edit_id = None
                st.session_state.should_rerun = True

            st.button("🗑️ ลบบันทึกนี้", on_click=on_apply_delete)

    with tab2:
        st.markdown("<h2 class='summary-title'><span class='emoji'>📅</span> ปฏิทิน Mood</h2>", unsafe_allow_html=True)
        coly, colm = st.columns(2)
        with coly:
            y = st.number_input("ปี", 2000, 2100, datetime.now().year)
        with colm:
            m = st.selectbox("เดือน", list(range(1, 13)), index=datetime.now().month - 1)

        cal = calendar.monthcalendar(y, m)
        last_emo = df.groupby("date")["emoji"].last()

        table = [
            [
                f"{d}\n{last_emo.get(datetime(y, m, d).date(), '')}" if d != 0 else ""
                for d in week
            ]
            for week in cal
        ]
        df_calendar = pd.DataFrame(table, columns=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        df_calendar.index = [""] * len(df_calendar)

        st.table(df_calendar)

    with tab3:
        st.markdown("<h2 class='summary-title'><span class='emoji'>📊</span> สถิติอารมณ์ 7 วันล่าสุด</h2>", unsafe_allow_html=True)
        today = datetime.now().date()
        start_of_week = today - timedelta(days=today.weekday())
        recent = df[df["date"] >= start_of_week]

        if recent.empty:
            st.warning("ยังไม่มีบันทึกในช่วง 7 วัน")
        else:
            sentiment_score_map = {"pos": 1.0, "neu": 0.5, "neg": 0.0}
            recent["scaled_score"] = recent["sentiment"].map(sentiment_score_map)
            avg = recent["scaled_score"].mean()

            st.markdown("<h3 class='highlight-yellow'> สรุปค่าเฉลี่ยระดับความรู้สึก</h3>", unsafe_allow_html=True)
            st.metric("ค่าเฉลี่ยความรู้สึกโดยรวมในช่วง 7 วันล่าสุด", f"{avg * 100:.2f} %")

            emoji, summary = ("😊", "สัปดาห์นี้คุณดูอารมณ์ดีสุด ๆ ไปเลย! เก็บพลังงานดี ๆ ไว้ให้ตัวเองและแบ่งให้คนรอบข้างนะ 💖") if avg >= 0.75 else ("😐", "สัปดาห์นี้อารมณ์ค่อนข้างกลาง ๆ ลองหาเวลาออกไปเที่ยวเผื่อจะเป็นสัปดาห์ที่ดีสุดๆเลยก็ได้ ✨") if avg >= 0.4 else ("😢", "สัปดาห์นี้ดูเหนื่อย ๆ 🫂 อย่าลืมพักผ่อน ทำสิ่งที่ชอบ กินของอร่อยเยอะๆ จะช่วยฮีลใจคุณได้ 💛")
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f"<div style='background:#ffe6f2;border-radius:10px;padding:30px;text-align:center;'><div style='font-size:60px'>{emoji}</div><div style='font-size:18px;'>อารมณ์สัปดาห์นี้</div></div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div style='background:#e8f5e9;border-radius:10px;padding:20px;font-size:18px;'>{summary}</div>", unsafe_allow_html=True)

            emoji_sentiment_df = recent.groupby(["emoji", "sentiment"]).size().reset_index(name="count")
        
            # กำหนดสีพาสเทลที่ใช้
            pastel_colors = {
                "pos": "#A7F3D0",  # สีเขียวพาสเทล
                "neu": "#FEF9C3",  # สีเหลืองพาสเทล
                "neg": "#FECACA"  # สีแดงพาสเทล
            }

            # แสดงกราฟจำนวนอีโมจิ (แยกตามความรู้สึก) ด้วยสีพาสเทล
            col1, col2 = st.columns(2)
            with col1:
                fig_emoji = px.bar(emoji_sentiment_df, x="emoji", y="count", color="sentiment", 
                               title="จำนวนอีโมจิ (แยกตามความรู้สึก)",
                               color_discrete_map=pastel_colors)  # ใช้สีพาสเทล
                st.plotly_chart(fig_emoji, use_container_width=True)
        
            # แสดงกราฟสัดส่วนความรู้สึก (Pie chart) ด้วยสีพาสเทล
            with col2:
                sentiment_counts = recent["sentiment"].value_counts().reset_index()
                sentiment_counts.columns = ["sentiment", "count"]
                fig_sentiment = px.pie(sentiment_counts, names="sentiment", values="count", 
                                   title="สัดส่วนความรู้สึก",
                                   color="sentiment", 
                                   color_discrete_map=pastel_colors)  # ใช้สีพาสเทล
                st.plotly_chart(fig_sentiment, use_container_width=True)

            st.markdown("<h4 class='highlight-yellow'>แนวโน้ม Sentiment</h4>", unsafe_allow_html=True)
            recent["mood_level"] = recent["sentiment"].map({"neg": 1, "neu": 2, "pos": 3})
            recent["day_label"] = recent["date"].apply(lambda d: d.strftime("%a %d %b"))
            mood_trend = recent.groupby("day_label", sort=False)["mood_level"].mean().reset_index()
            fig = px.line(mood_trend, x="day_label", y="mood_level", markers=True, title="📊 Mood Trend (Past 7 Days)")
            fig.update_yaxes(tickvals=[1, 2, 3], ticktext=["😢 NEG", "😐 NEU", "😊 POS"], range=[0.8, 3.2])
            fig.update_traces(line_color="#FF69B4", marker=dict(color="#FFB6C1", size=10))
            st.plotly_chart(fig, use_container_width=True)

# Optional: Auto-refresh after save/edit/delete
if st.session_state.get("should_rerun", False):
    st.session_state.should_rerun = False
    st.markdown("""
        <script>
        setTimeout(function() {
            window.location.reload();
        }, 2000);
        </script>
    """, unsafe_allow_html=True)
