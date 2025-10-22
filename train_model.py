
import os
import pandas as pd
import string
import joblib
import requests
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# --- 1. 資料準備 ---

# 建立資料夾
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# 下載資料集
url = "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/master/Chapter03/datasets/sms_spam_no_header.csv"
csv_path = "data/sms_spam_no_header.csv"

if not os.path.exists(csv_path):
    print("正在下載資料集...")
    response = requests.get(url)
    response.raise_for_status() 
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(response.text)
    print("資料集下載完成。")

# 讀取資料
print("正在讀取資料...")
df = pd.read_csv(csv_path, names=["label", "message"])

# --- 2. 資料前處理 ---
print("正在進行資料前處理...")
# 將文字轉為小寫並移除標點符號
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

df['message'] = df['message'].apply(preprocess_text)

# 分割訓練與測試集
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# --- 3. 特徵工程 (TF-IDF) ---
print("正在進行特徵工程...")
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# --- 4. 模型訓練 ---
print("正在訓練 Naive Bayes 模型...")
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# --- 5. 模型評估 ---
print("正在評估模型...")
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\n--- 模型評估結果 ---")
print(f"準確率 (Accuracy): {accuracy:.4f}")
print("分類報告 (Classification Report):")
print(report)
print("---------------------\n")

# --- 6. 儲存模型與向量器 ---
print("正在儲存模型與向量器...")
joblib.dump(model, "models/spam_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("訓練完成！模型與向量器已成功儲存於 'models' 資料夾中。")
