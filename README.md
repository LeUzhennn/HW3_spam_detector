# Spam Detector 專案摘要與指南

## 1. 專案目標

本專案旨在建立一個基於機器學習的垃圾郵件/簡訊分類器。系統能夠：
- 訓練一個分類模型來區分正常訊息 (Ham) 和垃圾訊息 (Spam)。
- 提供一個互動式的 Web 應用程式，讓使用者可以輸入文字或使用範例資料來進行即時預測。
- 視覺化模型的訓練成果與資料集的詞彙特性。

## 2. 專案結構

```
spam_detector/
|
|-- data/
|   `-- sms_spam_no_header.csv      # 專案使用的原始資料集
|
|-- models/
|   |-- spam_model.pkl              # 訓練好的 Naive Bayes 分類模型
|   `-- vectorizer.pkl              # 訓練好的 TF-IDF 向量化器
|
|-- plots/
|   |-- classification_report.png   # 模型評估報告的視覺化圖表
|   |-- top_ham_tokens.png          # 正常訊息中最常見的詞彙圖表
|   `-- top_spam_tokens.png         # 垃圾訊息中最常見的詞彙圖表
|
|-- app.py                          # Streamlit Web 應用程式主體
|-- train_model.py                  # 用於訓練模型、評估並儲存的腳本
|-- token_list.py                   # 用於分析詞彙頻率並生成圖表的腳本
|-- requirements.txt                # 專案所需的 Python 套件
|-- prompt.txt                      # 原始的 Gemini 互動紀錄
`-- prompt2.txt                     # 本摘要檔案
```

## 3. 核心腳本說明

- **`train_model.py`**:
  - 自動從網路上下載 `sms_spam_no_header.csv` 資料集。
  - 對文字資料進行前處理 (轉小寫、移除標點符號)。
  - 使用 `TfidfVectorizer` 將文字轉換為數字特徵。
  - 訓練一個 `Multinomial Naive Bayes` 模型。
  - 評估模型效能，並將視覺化的分類報告儲存於 `plots/classification_report.png`。
  - 將訓練完成的模型 (`spam_model.pkl`) 和向量化器 (`vectorizer.pkl`) 儲存到 `models/` 資料夾。

- **`token_list.py`**:
  - 載入 `train_model.py` 產生的向量化器。
  - 分析資料集中正常 (Ham) 與垃圾 (Spam) 訊息的詞彙頻率。
  - 產生並儲存兩個長條圖 (`top_ham_tokens.png`, `top_spam_tokens.png`) 到 `plots/` 資料夾，顯示兩類訊息中最高頻的詞彙。

- **`app.py`**:
  - 使用 Streamlit 建立的互動式 Web 介面。
  - 載入 `models/` 中的模型和向量化器。
  - 提供兩種測試模式：
    1.  **隨機抽樣**: 從資料集中隨機抽取一筆訊息進行預測。
    2.  **手動輸入**: 讓使用者自行輸入文字內容進行預測。
  - 顯示預測結果，包含判斷的類別與其機率。
  - 整合並顯示 `plots/` 資料夾中的所有圖表，視覺化呈現模型效能與詞彙分析結果。

## 4. 如何執行

請依照以下步驟來完整執行本專案：

1.  **安裝依賴套件**:
    在您的終端機中，切換到 `spam_detector` 目錄，然後執行：
    ```bash
    pip install -r requirements.txt
    ```

2.  **訓練模型**:
    執行訓練腳本來產生模型相關檔案。
    ```bash
    python train_model.py
    ```
    此步驟會建立 `models/` 和 `plots/` 資料夾，並填入 `spam_model.pkl`, `vectorizer.pkl` 和 `classification_report.png`。

3.  **分析詞彙頻率**:
    執行詞彙分析腳本來產生對應的圖表。
    ```bash
    python token_list.py
    ```
    此步驟會產生 `top_ham_tokens.png` 和 `top_spam_tokens.png`。

4.  **啟動 Web 應用**:
    所有檔案都備妥後，啟動 Streamlit 應用程式。
    ```bash
    streamlit run app.py
    ```
    您的瀏覽器將會自動開啟一個網頁，顯示垃圾郵件預測系統的操作介面。
