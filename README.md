# Positive-Negative-Text-Polarity-Classification-with-NLP
Positive/Negative: Text Polarity Classification with NLP
The dataset comprises a training set and test set. The texts come from multiple sources and are taken out of context.

The label attribute in the training set tells you whether the corresponding text is positive or negative class, 1 being positive and 0 being negative. However, this is no definite boundary between positiveness and negativeness in expressing one's feelings or situations.

You are encouraged to study the training set to get a good understanding of how or why a piece of text is labeled positive or negative.

Once again, you are reminded that you are allowed to manipulate the training set in any way that helps you with the competition.

Note: the row_id is zero-indexed.

Note: the text in the dataset is semi-processed. This makes the pre-processing more difficult.

## 實驗紀錄 (Experiment Tracking)

| 版本 (Version) | 策略 (Strategy) | Eval F1 | Eval Acc | 備註 (Notes) |
| :--- | :--- | :--- | :--- | :--- |
| **v1** | Baseline: TF-IDF (Word 1-2) + Logistic Regression | 0.6622 | 0.6665 | 基礎文字清洗 + 5k 特徵 |
| **v2** | Feature Expansion: Word + Char TF-IDF (15k features) | 0.6697 | 0.6735 | 達到詞袋模型天花板 |
| **v3** | **Pre-trained Transformer (DistilBERT)** | 0.8196 | 0.8033 | 語義理解帶來顯著提升，但存在嚴重過擬合 (Loss 0.75) |
| **v3.1** | **Refined BERT: Preprocessing + Regularization** | **0.8239** | **0.8133** | 成功控制過擬合 (Loss 0.45)，泛化能力增強 |

### 結論與後續建議
1. **Transformer 是正解**: 從 67% 到 81% 的跳躍證明了預訓練模型在處理「脫離上下文」文字時的強大語意補全能力。
2. **正則化是關鍵**: 在小樣本上使用 BERT，低學習率 (2e-5) 與權重衰減 (0.05) 是防止模型走火入魔的關鍵。
3. **後續方向**: 若想進階到 85%+，可考慮使用更大的模型（如 `roberta-large`）或採用 **K-Fold Cross Validation** 進行模型集成（Ensemble）。

