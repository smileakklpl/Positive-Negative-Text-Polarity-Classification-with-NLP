import pandas as pd
import numpy as np
import regex as re
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
import nltk
from nltk.corpus import stopwords
import os

def download_resources():
    try:
        nltk.download('stopwords', quiet=True)
    except Exception as e:
        print(f"Warning: Could not download stopwords: {e}")

def clean_text_v2(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    
    # 1. 還原縮寫
    text = re.sub(r"(\w+)\s+t\b", r"\1not", text) 
    text = re.sub(r"i\s+m\b", "iam", text)
    text = re.sub(r"it\s+s\b", "its", text)
    text = re.sub(r"(\w+)\s+s\b", r"\1s", text)
    text = re.sub(r"(\w+)\s+re\b", r"\1are", text)
    text = re.sub(r"(\w+)\s+ve\b", r"\1have", text)
    text = re.sub(r"(\w+)\s+ll\b", r"\1will", text)
    
    # 2. 處理過度重複的字母 (如 sooooo -> soo)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # 3. 標記保護與雜訊移除
    text = text.replace("num_num", "TOKEN_NUM")
    text = text.replace("num_extend", "TOKEN_EXT")
    text = re.sub(r"[^a-z0-9\s_]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def main():
    download_resources()
    
    # 載入資料
    train_path = 'data/train_2022.csv'
    test_path = 'data/test_no_answer_2022.csv'
    if not os.path.exists(train_path):
        train_path = '../data/train_2022.csv'
        test_path = '../data/test_no_answer_2022.csv'

    print("Loading data and preprocessing...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    train_df['CLEAN_TEXT'] = train_df['TEXT'].apply(clean_text_v2)
    test_df['CLEAN_TEXT'] = test_df['TEXT'].apply(clean_text_v2)
    
    # 特徵工程：結合 Word N-grams 與 Char N-grams
    stop_words = set(stopwords.words('english'))
    sentiment_keywords = {'not', 'no', 'but', 'however', 'never', 'don', 'didn', 'doesn', 'won', 'couldn', 'shouldn'}
    custom_stopwords = list(stop_words - sentiment_keywords)

    # 建立 FeatureUnion 以結合不同層次的 TF-IDF
    features = FeatureUnion([
        ('word_tfidf', TfidfVectorizer(
            ngram_range=(1, 2), max_features=10000, min_df=2, stop_words=custom_stopwords
        )),
        ('char_tfidf', TfidfVectorizer(
            analyzer='char_wb', ngram_range=(3, 5), max_features=5000, min_df=2
        ))
    ])

    pipeline = Pipeline([
        ('features', features),
        ('clf', LogisticRegression(C=1.5, class_weight='balanced', max_iter=2000, solver='lbfgs'))
    ])

    X = train_df['CLEAN_TEXT']
    y = train_df['LABEL']
    
    print("Evaluating Version 2 with 5-Fold CV...")
    f1_scores = cross_val_score(pipeline, X, y, cv=5, scoring='f1')
    acc_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
    
    print("-" * 30)
    print(f"CV F1-score: {f1_scores.mean():.4f} (+/- {f1_scores.std()*2:.4f})")
    print(f"CV Accuracy: {acc_scores.mean():.4f} (+/- {acc_scores.std()*2:.4f})")
    print("-" * 30)
    
    # 訓練最終模型
    print("Training final model for submission...")
    pipeline.fit(X, y)
    predictions = pipeline.predict(test_df['CLEAN_TEXT'])
    
    output_dir = 'result'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    submission_name = os.path.join(output_dir, 'submission_v2.csv')
    pd.DataFrame({'row_id': test_df['row_id'], 'LABEL': predictions}).to_csv(submission_name, index=False)
    print(f"Successfully generated '{submission_name}'")

if __name__ == "__main__":
    main()
