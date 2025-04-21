import streamlit as st
import pandas as pd
import calendar
from datetime import datetime, timedelta
import os
import uuid
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# ─── CONFIG ─────────────────────────────────────────────────────────
st.set_page_config(page_title="เสียงในใจ — Mood Diary", layout="wide")
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
st.title("🧠 เสียงในใจ — Mood Diary")
df = load_data()
col1, col2 = st.columns([1, 2])

if "entry_date" not in st.session_state:
    st.session_state.entry_date = datetime.now().date()

if "entry_text" not in st.session_state:
    st.session_state.entry_text = ""

# ─── LEFT ───────────────────────────────────────────────────────────
with col1:
    st.subheader("✍️ เขียนไดอารี่")
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
            st.subheader("📝 บันทึกย้อนหลัง (ใหม่ → เก่า)")
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
                    st.session_state.edit_id = None
                    st.session_state.should_rerun = True  # ✅ ตั้ง flag

                # ปุ่มกดอยู่นอกฟังก์ชัน
                st.button("💾 บันทึกการแก้ไข", on_click=on_apply_edit, key=f"save_{old['id']}")


                def on_apply_delete():
                    delete_entry(old["id"])
                    st.success("🗑️ ลบบันทึกเรียบร้อยแล้ว")
                    st.session_state.edit_id = None
                    st.rerun()
                st.button("🗑️ ลบบันทึกนี้", on_click=on_apply_delete)

        # ── Calendar ───────────────────────────────
        with tab2:
            st.subheader("📅 ปฏิทิน Mood")
            y = st.number_input("ปี", 2000, 2100, datetime.now().year)
            m = st.selectbox("เดือน", list(range(1, 13)), index=datetime.now().month - 1)
            cal = calendar.monthcalendar(y, m)
            last_emo = df.groupby("date")["emoji"].last()
            table = []
            for week in cal:
                row = []
                for d in week:
                    if d == 0:
                        row.append("")
                    else:
                        row.append(last_emo.get(datetime(y, m, d).date(), ""))
                table.append(row)
            st.table(pd.DataFrame(table, columns=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]))

        # ── Stats ─────────────────────────────────
        with tab3:
            st.subheader("📊 สถิติอารมณ์ 7 วันล่าสุด")
            cutoff = datetime.now().date() - timedelta(days=7)
            recent = df[df["date"] >= cutoff]
            if recent.empty:
                st.warning("ยังไม่มีบันทึกในช่วง 7 วัน")
            else:
                st.bar_chart(recent["emoji"].value_counts())
                st.bar_chart(recent["sentiment"].value_counts())
                avg = recent["score"].mean()
                st.metric("🎯 ค่าเฉลี่ยความรู้สึก", f"{avg:.0%}")
                st.markdown("#### ✉️ ข้อความ 7 วัน")
                for _, r in recent.sort_values("date").iterrows():
                    st.markdown(
                        f"- **{r['date']}** {r['emoji']} ({r['sentiment'].upper()} {r['score']:.0%}) → {r['text']}"
                    )

if st.session_state.get("should_rerun", False):
    st.session_state.should_rerun = False  # reset
    st.rerun()