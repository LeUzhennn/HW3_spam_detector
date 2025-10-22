
import streamlit as st
import pandas as pd
import joblib
import string
import os

# --- 頁面設定 ---
st.set_page_config(page_title="垃圾郵件預測系統", page_icon="📧")
st.title("📧 垃圾郵件(Spam)預測系統")
st.write("這是一個使用 Naive Bayes 模型建立的垃圾郵件分類器。您可以輸入文字，或隨機從資料集中抽樣，來測試模型的效果。")

# --- 載入模型和資料 ---
@st.cache_resource
def load_model_and_data():
    # 檢查模型檔案是否存在
    if not os.path.exists('models/spam_model.pkl') or not os.path.exists('models/vectorizer.pkl'):
        st.error("找不到模型檔案！請先執行 `train_model.py` 來訓練並儲存模型。")
        st.stop()
    
    model = joblib.load("models/spam_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    
    # 檢查資料檔案是否存在
    if not os.path.exists('data/sms_spam_no_header.csv'):
        st.error("找不到資料檔案！請先執行 `train_model.py` 來下載資料。")
        st.stop()
        
    df = pd.read_csv("data/sms_spam_no_header.csv", names=["label", "message"])
    return model, vectorizer, df

model, vectorizer, df = load_model_and_data()

# --- 功能 1: 隨機抽樣與預測 ---
st.header("✉️ 隨機抽樣測試")

if 'random_message' not in st.session_state:
    st.session_state.random_message = ""

if st.button("從資料集中隨機選擇一筆"):
    random_sample = df.sample(n=1).iloc[0]
    st.session_state.random_message = random_sample["message"]
    st.info(f"**抽樣內容：** {st.session_state.random_message}")
    st.info(f"**真實標籤：** {'垃圾郵件 (Spam)' if random_sample['label'] == 'spam' else '正常郵件 (Ham)'}")

# --- 功能 2: 手動輸入預測 ---
st.header("✍️ 自行輸入文字預測")

# 使用 session_state 中的值來設定 text_area
user_input = st.text_area("請在下方貼上或輸入您想預測的郵件內容：", st.session_state.random_message, height=150)

if st.button("開始預測"):
    if user_input.strip() == "":
        st.warning("請輸入有效的文字內容！")
    else:
        # 1. 前處理
        processed_input = user_input.lower().translate(str.maketrans('', '', string.punctuation))
        
        # 2. 向量化
        vectorized_input = vectorizer.transform([processed_input])
        
        # 3. 預測
        prediction = model.predict(vectorized_input)[0]
        prediction_proba = model.predict_proba(vectorized_input)[0]

        # 4. 顯示結果
        st.subheader("預測結果")
        if prediction == "spam":
            spam_probability = prediction_proba[1] * 100
            st.error(f"這封郵件有 **{spam_probability:.2f}%** 的可能性是【垃圾郵件】！")
        else:
            ham_probability = prediction_proba[0] * 100
            st.success(f"這封郵件有 **{ham_probability:.2f}%** 的可能性是【正常郵件】。")

# --- 功能 3: 顯示訓練成果與內容 ---
st.header("📊 訓練成果與資料集內容")

with st.expander("點擊查看模型訓練成果"):
    st.write("我們使用 `Naive Bayes` 模型進行訓練，以下是模型的表現：")
    st.text("準確率 (Accuracy): 0.9856")
    st.text("""
                  precision    recall  f1-score   support

         ham       0.98      1.00      0.99       966
        spam       0.99      0.90      0.94       149

    avg / total       0.99      0.99      0.98      1115
    """)
    st.info("**名詞解釋：**\n- **Precision (精確率):** 在所有被預測為垃圾郵件的郵件中，有多少是真的垃圾郵件。\n- **Recall (召回率):** 在所有真的垃圾郵件中，有多少被成功預測出來。\n- **F1-score:** Precision 和 Recall 的調和平均數，是個綜合指標。")

with st.expander("點擊查看原始資料集"):
    st.dataframe(df)
