# v3 策略說明

## 目標
- 從稀疏特徵的傳統方法，轉向能理解語意的預訓練 Transformer。

## 策略
1. 採用適合 Transformer 的清洗方式：
   - 還原被拆開的縮寫
   - 壓縮重複字母
   - 在有用時保留標點與自然語句
2. 使用 DistilBERT 作為基礎模型。
3. 以固定最大長度進行 tokenizer 切分，並使用 train/validation split 訓練。
4. 以較低 learning rate 與 regularization 進行 fine-tuning，降低過擬合。

## v3.1 的修正
- 增加正則化強度，讓訓練更穩定。
- 改善前處理流程，並調整超參數。

## 為什麼有效
- 這個模型可以比 TF-IDF 更好地根據上下文推斷語意，對於脫離上下文的文字片段特別重要。

## 限制
- 單一 validation split 的評估仍然偏不穩定，而且模型在資料量不大的情況下仍可能過擬合。
