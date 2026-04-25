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
| **v4 (正式版)** | **RoBERTa-base + Stratified 5-Fold + Soft Voting Ensemble** | **0.8348** | **0.8305** | GPU 訓練完成，泛化與穩定性均優於 v3.1 |
| **v4.1** | **RoBERTa-base + Stratified 5-Fold + Class Weights** | **0.8348** | **0.8305** | 引入 WeightedTrainer 與 balanced 權重處理資料不平衡 |
| **v5** | **RoBERTa-large + Pseudo-labeling + FP16** | **0.8715** | **0.8695** | 引入 v4 預測作偽標籤，大幅突破準確率天花板。Train Loss 收斂至 ~0.255 |

### 結論與後續建議
1. **模型容量與資料量的突破**: v5 升級 `roberta-large` 並引入 Pseudo-labeling 後，不僅訓練穩定，**Eval F1 更是大幅躍升至 0.8715**，成功突破先前的瓶頸。
2. **硬體資源最佳化**: 透過 FP16 與梯度累積，RTX 5070 Ti 成功扛住 Large 模型。單 Fold 訓練僅需約 6.5~9 分鐘 (總時長約 39 分鐘)。
3. **後續方向**: 針對 Pseudo-labeling，未來可實作信心度過濾 (Confidence Thresholding)，只取機率 > 0.9 或 < 0.1 的高信心樣本作為偽標籤。

## 檔案層級與用途 (Project Structure)

| Path | Purpose |
| :--- | :--- |
| `data/` | 原始訓練資料、測試資料與提交範例 |
| `doc/` | 額外報告、筆記與說明文件 |
| `result/` | 各版本模型輸出與 submission 檔 |
| `src/version1.py` | v1 基線：TF-IDF + Logistic Regression |
| `src/version2.py` | v2 改良：Word + Char TF-IDF + Logistic Regression |
| `src/version3.py` | v3 Transformer：DistilBERT 微調 |
| `src/version4.py` | v4 正式版：RoBERTa-base + 5-Fold + Soft Voting |
| `src/version5.py` | v5 進階版：RoBERTa-large + Pseudo-labeling + FP16 |
| `src/docs/v1_strategy.md` | v1 詳細策略說明 |
| `src/docs/v2_strategy.md` | v2 詳細策略說明 |
| `src/docs/v3_strategy.md` | v3 / v3.1 詳細策略說明 |
| `src/docs/v4_strategy.md` | v4 詳細策略說明 |
| `src/docs/v5_strategy.md` | v5 詳細策略說明 |
| `README.md` | 專案摘要、成績總覽與文件導覽 |
| `requirements.txt` | Python 套件依賴 |
| `pyproject.toml` | 專案設定與建置資訊 |

## 詳細策略文件索引

各版本的完整策略與實作筆記已拆分至 `src/docs/`：

- [v1_strategy.md](src/docs/v1_strategy.md)
- [v2_strategy.md](src/docs/v2_strategy.md)
- [v3_strategy.md](src/docs/v3_strategy.md)
- [v4_strategy.md](src/docs/v4_strategy.md)