# 🔍 AI vs Human 文章偵測器 (AI Detector)

這是一個基於機器學習的簡易 AI 文章偵測工具。使用者輸入一段英文文本後，系統會分析並給出「人類撰寫」與「AI 生成」的機率百分比。

## 🌟 專案特色
- **即時判斷**：輸入文字後立即顯示 AI% / Human%。
- **機器學習驅動**：使用 Scikit-learn 的 TF-IDF 向量化技術與邏輯迴歸 (Logistic Regression)。
- **現代化 UI**：使用 Streamlit 打造簡潔的網頁介面。
- **視覺化結果**：透過進度條與量表清楚展示偵測數據。

## 🚀 如何在本地端執行 (Demo 步驟)

請按照以下步驟在你的電腦上運行此專案：

### 1. 複製專案
```bash
git clone <你的 GitHub 專案網址>
cd ai-detector-app
### 1. 複製專案
```bash
git clone <你的 GitHub 專案網址>
cd ai-detector-app
2. 安裝必要套件
建議先建立虛擬環境：

Bash

python -m venv venv
# 啟動虛擬環境 (Windows)
.\venv\Scripts\activate
# 啟動虛擬環境 (Mac/Linux)
source venv/bin/activate

pip install -r requirements.txt
3. 訓練模型
在執行網頁前，需要先產生模型檔案：

Bash

python train_model.py
4. 啟動 Streamlit Demo
Bash

streamlit run app.py
啟動後，瀏覽器會自動開啟 http://localhost:8501。

🌐 線上 Demo 方式
本專案支援 Streamlit Cloud 快速部署：

將專案推送到 GitHub。

登入 Streamlit Cloud。

連結此 Repository 並選擇 app.py 進行部署。

🛠 技術棧
Language: Python 3.9+

ML Library: Scikit-learn, Joblib

Web Framework: Streamlit

Feature Extraction: TF-IDF (Term Frequency-Inverse Document Frequency)