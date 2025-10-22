import pandas as pd
import joblib
from collections import Counter
import os
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_tokens(data_path='data/sms_spam_no_header.csv', vectorizer_path='models/vectorizer.pkl', top_n=20):
    """
    分析垃圾郵件資料集中的詞彙，統計 'ham' 和 'spam' 類別中出現頻率最高的詞彙。
    """
    print("載入資料集...")
    df = pd.read_csv(data_path, encoding='latin-1')
    df = df.iloc[:, :2] # 只取前兩列
    df.columns = ['label', 'message']

    print("載入 CountVectorizer...")
    if not os.path.exists(vectorizer_path):
        print(f"錯誤：找不到向量化器檔案 '{vectorizer_path}'。請確保已執行 train_model.py。")
        return

    with open(vectorizer_path, 'rb') as f:
        vectorizer = joblib.load(f)

    # 分離 ham 和 spam 訊息
    ham_messages = df[df['label'] == 'ham']['message']
    spam_messages = df[df['label'] == 'spam']['message']

    print("分析 ham 訊息中的詞彙頻率...")
    ham_transformed = vectorizer.transform(ham_messages)
    ham_token_counts = ham_transformed.sum(axis=0)
    ham_token_freq = Counter(
        {token: ham_token_counts[0, idx] for idx, token in enumerate(vectorizer.get_feature_names_out())}
    )

    print("分析 spam 訊息中的詞彙頻率...")
    spam_transformed = vectorizer.transform(spam_messages)
    spam_token_counts = spam_transformed.sum(axis=0)
    spam_token_freq = Counter(
        {token: spam_token_counts[0, idx] for idx, token in enumerate(vectorizer.get_feature_names_out())}
    )

    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)

    # Plotting Ham tokens
    plt.figure(figsize=(12, 6))
    sns.barplot(x=[token for token, count in ham_token_freq.most_common(top_n)],
                y=[count for token, count in ham_token_freq.most_common(top_n)],
                palette='viridis')
    plt.title(f'Top {top_n} Most Common Ham Tokens')
    plt.xlabel('Token')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('plots/top_ham_tokens.png')
    plt.close()

    # Plotting Spam tokens
    plt.figure(figsize=(12, 6))
    sns.barplot(x=[token for token, count in spam_token_freq.most_common(top_n)],
                y=[count for token, count in spam_token_freq.most_common(top_n)],
                palette='magma')
    plt.title(f'Top {top_n} Most Common Spam Tokens')
    plt.xlabel('Token')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('plots/top_spam_tokens.png')
    plt.close()

    print(f"詞彙頻率圖表已儲存至 'plots/' 資料夾。")

if __name__ == "__main__":
    # 確保在正確的目錄下執行
    current_dir = os.getcwd()
    if not current_dir.endswith('spam_detector'):
        print(f"警告：目前工作目錄不是 'spam_detector'。請確保資料和模型路徑正確。目前目錄: {current_dir}")
    
    analyze_tokens()
