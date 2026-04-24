# v2 策略說明

## 目標
- 在傳統基線上再往前推一步，透過詞層級與字元層級特徵的結合提升表現。

## 策略
1. 延續 v1 的文字清洗流程。
2. 使用 FeatureUnion 組合兩種特徵：
   - word TF-IDF，n-grams `(1, 2)`
   - character TF-IDF，使用 `char_wb` n-grams `(3, 5)`
3. 將總特徵量提升至 15,000。
4. 以 Logistic Regression 搭配 class balancing 訓練模型。

## 為什麼有進步
- 字元 n-grams 能補回拼字變化、縮寫型態與局部字形資訊，這些是純 word TF-IDF 容易漏掉的。

## 限制
- 即使 sparse features 更豐富，模型仍缺乏真正的語意理解，因此效果最終會趨於飽和。
