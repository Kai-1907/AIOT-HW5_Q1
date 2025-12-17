import streamlit as st
import joblib
import numpy as np

# 設定頁面資訊
st.set_page_config(page_title="AI vs Human Detector", page_icon="🤖")

st.title("🔍 AI / Human 文章偵測器")
st.markdown("輸入一段英文文本，讓我們分析它是 AI 生成還是人類撰寫的。")

# 載入模型
@st.cache_resource
def load_models():
    model = joblib.load('model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    return model, tfidf

try:
    model, tfidf = load_models()
except:
    st.error("請先執行 train_model.py 產生模型檔案！")
    st.stop()

# 使用者輸入
text_input = st.text_area("請貼上文章內容：", height=250, placeholder="Once upon a time...")

if st.button("立即偵測"):
    if text_input.strip() == "":
        st.warning("請輸入內容再進行分析")
    else:
        # 預測處理
        vectorized_text = tfidf.transform([text_input])
        prediction_proba = model.predict_proba(vectorized_text)[0]
        
        human_score = prediction_proba[0] * 100
        ai_score = prediction_proba[1] * 100

        # 顯示結果
        st.divider()
        col1, col2 = st.columns(2)
        col1.metric("👨‍💻 Human Score", f"{human_score:.1f}%")
        col2.metric("🤖 AI Score", f"{ai_score:.1f}%")

        # 進度條可視化
        st.progress(ai_score / 100)
        
        if ai_score > 50:
            st.error(f"這篇文章看起來有 {ai_score:.1f}% 的機率是由 AI 撰寫的。")
        else:
            st.success(f"這篇文章看起來有 {human_score:.1f}% 的機率是由人類撰寫的。")

st.sidebar.info("本工具基於 TF-IDF 與 Logistic Regression 實作，僅供參考。")