
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

# --- 互動測試區 ---
st.header("📨 互動測試區")

if 'random_message' not in st.session_state:
    st.session_state.random_message = ""

col1, col2 = st.columns(2)

with col1:
    st.subheader("選項 1: 隨機抽樣")
    if st.button("從資料集中隨機選擇一筆"):
        random_sample = df.sample(n=1).iloc[0]
        st.session_state.random_message = random_sample["message"]
        st.info(f"**真實標籤：** {'垃圾郵件 (Spam)' if random_sample['label'] == 'spam' else '正常郵件 (Ham)'}")

with col2:
    st.subheader("選項 2: 從範例選擇")
    example_messages = {
        "選擇一個範例...": "",
        "正常郵件 (Ham) 範例 1": "I'm going to try for 2 months ha ha only joking",
        "正常郵件 (Ham) 範例 2": "Sorry, I'll call later",
        "垃圾郵件 (Spam) 範例 1": "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461.",
        "垃圾郵件 (Spam) 範例 2": "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's",
    }
    selected_example_key = st.selectbox("選擇範例", options=list(example_messages.keys()))
    if selected_example_key != "選擇一個範例...":
        st.session_state.random_message = example_messages[selected_example_key]

# --- 手動輸入預測 ---
st.subheader("✍️ 自行輸入或貼上文字")
user_input = st.text_area("郵件內容：", st.session_state.random_message, height=150)

if st.button("開始預測"):
    if user_input.strip() == "":
        st.warning("請輸入有效的文字內容！")
    else:
        with st.spinner("模型預測中..."):
            # 1. 前處理
            processed_input = user_input.lower().translate(str.maketrans(' ', ' ', string.punctuation))
            
            # 2. 向量化
            vectorized_input = vectorizer.transform([processed_input])
            
            # 3. 預測
            prediction = model.predict(vectorized_input)[0]
            prediction_proba = model.predict_proba(vectorized_input)[0]

            # 4. 顯示結果
            st.subheader("預測結果")
            if prediction == "spam":
                spam_probability = prediction_proba[1]
                st.error(f"這封郵件有 **{spam_probability*100:.2f}%** 的可能性是【垃圾郵件】！")
                st.progress(spam_probability)
            else:
                ham_probability = prediction_proba[0]
                st.success(f"這封郵件有 **{ham_probability*100:.2f}%** 的可能性是【正常郵件】。")
                st.progress(ham_probability)

# --- 功能 3: 顯示訓練成果與內容 ---
st.header("📊 訓練成果與資料集內容")

st.subheader("模型訓練成果") # Changed from st.expander title
st.write("我們使用 `Naive Bayes` 模型進行訓練，以下是模型的表現：")
st.text("準確率 (Accuracy): 0.9856") # Keeping hardcoded accuracy for now

report_plot_path = "plots/classification_report.png"
if os.path.exists(report_plot_path):
    st.image(report_plot_path, caption="分類報告 (Classification Report)")
else:
    st.warning(f"找不到分類報告圖表：{report_plot_path}。請先執行 `train_model.py`。")

st.info("**名詞解釋：**\n- **Precision (精確率):** 在所有被預測為垃圾郵件的郵件中，有多少是真的垃圾郵件。\n- **Recall (召回率):** 在所有真的垃圾郵件中，有多少被成功預測出來。\n- **F1-score:** Precision 和 Recall 的調和平均數，是個綜合指標。")

st.subheader("原始資料集") # Changed from st.expander title
st.dataframe(df)

# --- 功能 4: 詞彙頻率分析 ---
st.header("🔍 詞彙頻率分析")
st.write("請先執行 `token_list.py` 以生成詞彙頻率圖表。")
ham_plot_path = "plots/top_ham_tokens.png"
spam_plot_path = "plots/top_spam_tokens.png"

if os.path.exists(ham_plot_path):
    st.subheader("最常出現的 Ham 詞彙")
    st.image(ham_plot_path, caption="Top Ham Tokens")
else:
    st.warning(f"找不到 Ham 詞彙圖表：{ham_plot_path}")

if os.path.exists(spam_plot_path):
    st.subheader("最常出現的 Spam 詞彙")
    st.image(spam_plot_path, caption="Top Spam Tokens")
else:
    st.warning(f"找不到 Spam 詞彙圖表：{spam_plot_path}")

