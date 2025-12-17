import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 1. 準備簡易數據 (實際應用請匯入大型 Dataset，如 Kaggle 的 AI vs Human 數據)
data = {
    'text': [
        "The sun rises in the east and sets in the west.", # Human
        "Artificial intelligence is transforming the world of technology.", # AI
        "I love eating pizza on a rainy Sunday afternoon.", # Human
        "The Large Language Model is trained on a vast corpus of text.", # AI
        # ... 這裡應該加入更多樣本
    ],
    'label': [0, 1, 0, 1] # 0: Human, 1: AI
}

df = pd.DataFrame(data)

# 2. 特徵提取 (TF-IDF)
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X = tfidf.fit_transform(df['text'])
y = df['label']

# 3. 訓練模型
model = LogisticRegression()
model.fit(X, y)

# 4. 儲存模型
joblib.dump(model, 'model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
print("模型訓練完成並已儲存！")