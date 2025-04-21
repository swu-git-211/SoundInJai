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
            "คุณดูไม่นิดหน่อย ลองพักผ่อน ฟังเพลงโปรด หรือคุยกับเพื่อนดูนะ",
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
st.title(" เสียงในใจ — Diary づ❤︎ど ")
df = load_data()
col1, col2 = st.columns([1, 2])

if "entry_date" not in st.session_state:
    st.session_state.entry_date = datetime.now().date()

if "entry_text" not in st.session_state:
    st.session_state.entry_text = ""

# ─── LEFT ───────────────────────────────────────────────────────────
with col1:
    st.subheader("เขียนไดอารี่")
    entry_date = st.date_input(
        "วันที่",
        value=st.session_state.get("entry_date", datetime.now().date()),
        key="entry_date"
    )
    existing = df[df["date"] == entry_date]
    default_text = existing.iloc[0]["text"] if not existing.empty else ""
    entry_text = st.text_area(
        "บันทึกความรู้สึก…", 
        value=st.session_state.get("entry_text", default_text), 
        height=200,
        key="entry_text"
    )

    def on_new_save():
        if st.session_state.entry_text.strip():
            lab, sc = analyze_sentiment(st.session_state.entry_text)
            em = EMOJI_MAP[lab]
            save_entry(entry_date, st.session_state.entry_text, lab, sc, em)
            st.success(f"{em} บันทึกเรียบร้อย! ({lab.upper()} {sc:.0%})")
            # โชว์คำแนะนำ
            suggestion = suggest_message(lab, sc)
            st.info(f"💡 คำแนะนำวันนี้: {suggestion}")
            # ล้างข้อความหลังจากบันทึก
            st.session_state.entry_text = ""  # reset textarea
            st.session_state.entry_date = datetime.now().date()  # reset date to today
        else:
            st.error("กรุณาใส่ข้อความก่อนบันทึก")

    st.button("💾 บันทึกและวิเคราะห์", on_click=on_new_save)

# ─── RIGHT ──────────────────────────────────────────────────────────
with col2:
    if df.empty:
        st.info("ยังไม่มีบันทึกเลย ลองเพิ่มดูสิ")
    else:
        tab1, tab2, tab3 = st.tabs(["Summary", "Calendar", "Stats"])

        # ── Summary ───────────────────────────────
        with tab1:
            st.subheader("📝 บันทึกย้อนหลัง")
            df2 = df.sort_values("date", ascending=False).reset_index(drop=True)
            if "edit_id" not in st.session_state:
                st.session_state.edit_id = None

            for _, row in df2.iterrows():
                c1, c2, c3, c4, c5, c6 = st.columns([1.3, 4, 1, 1, 1, 0.6])
                c1.write(str(row["date"]))
                c2.write(row["text"])
                c3.write(row["emoji"])
                c4.write(f"{row['score']:.0%}")
                c5.write(row["sentiment"].upper())
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
                    # โชว์คำแนะนำ
                    suggestion = suggest_message(lab, sc)
                    st.info(f"💡 คำแนะนำวันนี้: {suggestion}")
                    st.session_state.edit_id = None
                    st.session_state.should_rerun = True  # ✅ ตั้ง flag

                # ปุ่มกดอยู่นอกฟังก์ชัน
                st.button("💾 บันทึกการแก้ไข", on_click=on_apply_edit, key=f"save_{old['id']}")

                def on_apply_delete():
                  delete_entry(old["id"])
                  st.success("🗑️ ลบบันทึกเรียบร้อยแล้ว")
                  st.session_state.edit_id = None
                  st.session_state.should_rerun = True  # ✅ ตั้ง flag สำหรับ rerun
                
                
                # ปุ่มต้องอยู่นอกฟังก์ชัน!
                st.button("🗑️ ลบบันทึกนี้", on_click=on_apply_delete)



                # ── Calendar ───────────────────────────────
with tab2:
    st.subheader("📅 ปฏิทิน Mood")
    
    # ใช้ st.columns() เพื่อแสดงปีและเดือนในคอลัมน์ข้างๆ กัน
    col1, col2 = st.columns(2)

    with col1:
        y = st.number_input("ปี", 2000, 2100, datetime.now().year)
        
    with col2:
        m = st.selectbox("เดือน", list(range(1, 13)), index=datetime.now().month - 1)

    # สร้างปฏิทินตามปีและเดือนที่เลือก
    cal = calendar.monthcalendar(y, m)
    last_emo = df.groupby("date")["emoji"].last()
    
    # สร้างตารางปฏิทิน
    table = []
    for week in cal:
        row = []
        for d in week:
            if d == 0:
                row.append("")
            else:
                row.append(last_emo.get(datetime(y, m, d).date(), ""))
        table.append(row)

    # แสดงตารางปฏิทิน
    st.table(pd.DataFrame(table, columns=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]))

# ── Stats ─────────────────────────────────
import plotly.express as px

# ── Stats ─────────────────────────────────
with tab3:
    st.subheader("📊 สถิติอารมณ์ 7 วันล่าสุด")

    today = datetime.now().date()
    weekday = today.weekday()
    start_of_week = today - timedelta(days=weekday)
    recent = df[df["date"] >= start_of_week]

    if recent.empty:
        st.warning("ยังไม่มีบันทึกในช่วง 7 วัน")
    else:
        # ── สถิติต่างๆ ─────────────────────────────────

        # ── ค่าเฉลี่ยความรู้สึก ────────────────────
        sentiment_score_map = {"pos": 1.0, "neu": 0.5, "neg": 0.0}
        recent["scaled_score"] = recent["sentiment"].map(sentiment_score_map)

        # คำนวณค่าเฉลี่ย (แบบไม่ติดลบ)
        avg = recent["scaled_score"].mean()

        # แสดงผลลัพธ์
        st.markdown("### สรุปค่าเฉลี่ยระดับความรู้สึก")
        st.metric(
            label=" ค่าเฉลี่ยความรู้สึกโดยรวมในช่วง 7 วันล่าสุด",
            value=f"{avg * 100:.2f} %",
            help="คะแนน: POS = 100%, NEU = 50%, NEG = 0%"
        )

        # สรุปอารมณ์รวมทั้งสัปดาห์ พร้อมข้อความ
        if avg >= 0.75:
            emoji = "😊"
            summary = "อาทิตย์นี้คุณดูอารมณ์ดีสุด ๆ ไปเลย 💖 อย่าลืมดูแลตัวเองและแบ่งปันความสุขให้คนรอบข้างนะ!"
        elif avg >= 0.4:
            emoji = "😐"
            summary = "อารมณ์ในสัปดาห์นี้ค่อนข้างกลาง ๆ ลองหาเวลาพักผ่อนหรือทำสิ่งที่คุณชอบเพื่อชาร์จพลังดูนะ ✨"
        else:
            emoji = "😢"
            summary = "ดูเหมือนว่าสัปดาห์นี้จะค่อนข้างหนักหน่วง 🫂 ลองให้เวลากับตัวเองเยอะขึ้น พักใจ และขอความช่วยเหลือได้เสมอนะ 💛"

        col1, col2 = st.columns([1, 3])  # หรือจะ [1, 2] ก็ได้ถ้าอยากให้ emoji เล็กหน่อย

        with col1:
            st.markdown(
                f"""
                <div style='
                    background-color:#ffe6f2;
                    border-radius:10px;
                    padding:30px;
                    min-height:160px;
                    display:flex;
                    flex-direction:column;
                    justify-content:center;
                    align-items:center;
                    box-shadow: 2px 2px 10px #f3c6d1;
                '>
                    <div style='font-size:60px;'>{emoji}</div>
                    <div style='font-size:18px; margin-top:10px; color:#333;'>อารมณ์สัปดาห์นี้</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                f"""
                <div style='
                    background-color:#e8f5e9;
                    border-radius:10px;
                    padding:20px;
                    min-height:160px;
                    display:flex;
                    flex-direction:column;
                    justify-content:center;
                    font-size:18px;
                    line-height:1.6;
                    color:#333;
                    box-shadow: 2px 2px 10px #bde0c0;
                '>
                    ✦ <strong>คำแนะนำประจำสัปดาห์:</strong><br>
                    {summary}
                </div>
                """,
                unsafe_allow_html=True
            )

        emoji_sentiment_df = recent.groupby(["emoji", "sentiment"]).size().reset_index(name="count")

        sentiment_colors = {
            "positive": "green",
            "neutral": "gray",
            "negative": "red"
        }

        col1, col2 = st.columns(2)

        with col1:
            fig_emoji = px.bar(
                emoji_sentiment_df,
                x="emoji",
                y="count",
                color="sentiment",
                color_discrete_map=sentiment_colors,  # ใช้สีเดียวกัน
                title="จำนวนอีโมจิ (แยกตามความรู้สึก)"
            )
            st.plotly_chart(fig_emoji, use_container_width=True)

        with col2:
            sentiment_counts = recent["sentiment"].value_counts().reset_index()
            sentiment_counts.columns = ["sentiment", "count"]
            fig_sentiment = px.pie(
                sentiment_counts,
                names="sentiment",
                values="count",
                title="สัดส่วนความรู้สึก",
                color="sentiment",
                color_discrete_map=sentiment_colors  # ใช้สีเดียวกัน
            )
            st.plotly_chart(fig_sentiment, use_container_width=True)

        # ✨ แนวโน้มความรู้สึก (ตัวเลขแปลงจาก sentiment)
        st.markdown("#### แนวโน้ม Sentiment ")

        # แปลง sentiment เป็นระดับอารมณ์ 1 = neg, 2 = neu, 3 = pos
        recent["mood_level"] = recent["sentiment"].map({"neg": 1, "neu": 2, "pos": 3})

        # แสดงชื่อวันแบบย่อ + วันที่ เช่น "Mon 22 Apr"
        recent["day_label"] = recent["date"].apply(lambda d: d.strftime("%a %d %b"))

        # หาค่าเฉลี่ยถ้ามีหลายบันทึกต่อวัน
        mood_trend = recent.groupby("day_label", sort=False)["mood_level"].mean().reset_index()

        # วาดกราฟ
        fig = px.line(
            mood_trend,
            x="day_label",
            y="mood_level",
            markers=True,
            title="📊 Mood Trend (Past 7 Days)",
            labels={"day_label": "Day", "mood_level": "Mood Level"},
            template="plotly_white"
        )

        # กำหนดระดับแกน Y
        fig.update_yaxes(
            tickvals=[1, 2, 3],
            ticktext=["😢 NEG", "😐 NEU", "😊 POS"],
            range=[0.8, 3.2]
        )

        # แต่งสีพาสเทลน่ารัก ๆ
        fig.update_traces(
            line_color="#FF69B4",  # hot pink line
            marker=dict(color="#FFB6C1", size=10)  # pastel pink dots
        )

        st.plotly_chart(fig, use_container_width=True)


if st.session_state.get("should_rerun", False):
    st.session_state.should_rerun = False
    st.markdown("""
        <script>
        setTimeout(function() {
            window.location.reload();
        }, 2000);
        </script>
    """, unsafe_allow_html=True)