import pandas as pd
import numpy as np
import regex as re
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords
import os

def download_resources():
    """下載必要的 NLTK 資源"""
    try:
        nltk.download('stopwords', quiet=True)
        print("NLTK stopwords downloaded.")
    except Exception as e:
        print(f"Warning: Could not download stopwords: {e}")

def clean_text_v1(text):
    """
    針對競賽資料的精細化前處理：
    1. 還原被拆開的縮寫 (如 didn t -> didnnot)
    2. 保護特殊標記 (如 num_num -> TOKEN_NUM)
    3. 移除雜訊符號
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    
    # 1. 還原縮寫 (對情感分析極其重要)
    text = re.sub(r"(\w+)\s+t\b", r"\1not", text) 
    text = re.sub(r"i\s+m\b", "iam", text)
    text = re.sub(r"it\s+s\b", "its", text)
    text = re.sub(r"(\w+)\s+s\b", r"\1s", text)
    text = re.sub(r"(\w+)\s+re\b", r"\1are", text)
    text = re.sub(r"(\w+)\s+ve\b", r"\1have", text)
    text = re.sub(r"(\w+)\s+ll\b", r"\1will", text)
    
    # 2. 標記保護
    text = text.replace("num_num", "TOKEN_NUM")
    text = text.replace("num_extend", "TOKEN_EXT")
    
    # 3. 移除標點，保留字母、數字與受保護的底線
    text = re.sub(r"[^a-z0-9\s_]", " ", text)
    
    # 4. 去除多餘空格
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def main():
    # 0. 準備環境
    download_resources()
    
    # 1. 載入資料 (路徑相對於專案根目錄執行)
    train_path = 'data/train_2022.csv'
    test_path = 'data/test_no_answer_2022.csv'
    
    if not os.path.exists(train_path):
        # 嘗試在 src 目錄執行的情況下修正路徑
        train_path = '../data/train_2022.csv'
        test_path = '../data/test_no_answer_2022.csv'

    print(f"Loading data from: {train_path}")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    
    # 2. 前處理
    print("Preprocessing text...")
    train_df['CLEAN_TEXT'] = train_df['TEXT'].apply(clean_text_v1)
    test_df['CLEAN_TEXT'] = test_df['TEXT'].apply(clean_text_v1)
    
    # 3. 特徵工程
    # 自定義停用詞：排除具備情感強烈特徵的詞
    stop_words = set(stopwords.words('english'))
    sentiment_keywords = {'not', 'no', 'but', 'however', 'never', 'don', 'didn', 'doesn', 'won', 'couldn', 'shouldn'}
    custom_stopwords = list(stop_words - sentiment_keywords)
    
    print("Vectorizing text with TF-IDF...")
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2), 
        max_features=5000, 
        min_df=2, 
        stop_words=custom_stopwords
    )
    
    X = tfidf.fit_transform(train_df['CLEAN_TEXT'])
    y = train_df['LABEL']
    
    # 4. 模型評估 (5-Fold Cross Validation)
    print("Evaluating model with 5-Fold CV...")
    lr = LogisticRegression(C=1.2, class_weight='balanced', max_iter=1000)
    
    f1_scores = cross_val_score(lr, X, y, cv=5, scoring='f1')
    acc_scores = cross_val_score(lr, X, y, cv=5, scoring='accuracy')
    
    print("-" * 30)
    print(f"CV F1-score: {f1_scores.mean():.4f} (+/- {f1_scores.std()*2:.4f})")
    print(f"CV Accuracy: {acc_scores.mean():.4f} (+/- {acc_scores.std()*2:.4f})")
    print("-" * 30)
    
    # 5. 訓練最終模型並產生預測
    print("Training final model and generating predictions...")
    lr.fit(X, y)
    X_test = tfidf.transform(test_df['CLEAN_TEXT'])
    predictions = lr.predict(X_test)
    
    # 建立提交檔
    submission = pd.DataFrame({
        'row_id': test_df['row_id'],
        'LABEL': predictions
    })
    
    submission_name = 'submission_v1.csv'
    submission.to_csv(submission_name, index=False)
    print(f"Successfully generated '{submission_name}'")

if __name__ == "__main__":
    main()
