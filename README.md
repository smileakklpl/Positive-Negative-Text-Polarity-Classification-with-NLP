# SentimentForge：多階段 Transformer 集成文字極性分類

> **SentimentForge** — Iterative Multi-Stage Transformer Ensemble for Text Polarity Classification
>
> 從傳統詞袋模型出發，歷經六個版本的系統性迭代，最終以 RoBERTa-large、DeBERTa-v3-large、ELECTRA-large 三模型集成，在競賽資料集上達到 OOF F1 **0.8864**。

---

## 資料集說明

訓練集與測試集文本來自多種來源，且已脫離原本語境。訓練集的 `label` 欄位標示極性：**1 = 正面**、**0 = 負面**。分類界線並非絕對，鼓勵深入研究訓練集的標注規律。資料集為半處理狀態，前處理難度略高於原始文本。`row_id` 從 0 開始計數。

---

## 實驗紀錄 (Experiment Tracking)

| 版本 | 策略 | OOF F1 | OOF Acc | 備註 |
| :--- | :--- | :--- | :--- | :--- |
| **v1** | TF-IDF (Word 1-2) + Logistic Regression | 0.6622 | 0.6665 | 基礎文字清洗 + 5k 特徵 |
| **v2** | Feature Expansion：Word + Char TF-IDF (15k features) | 0.6697 | 0.6735 | 達到詞袋模型天花板 |
| **v3** | Pre-trained Transformer (DistilBERT) | 0.8196 | 0.8033 | 語義理解帶來顯著提升，但存在嚴重過擬合 (Loss 0.75) |
| **v3.1** | Refined BERT：Preprocessing + Regularization | 0.8239 | 0.8133 | 成功控制過擬合 (Loss 0.45)，泛化能力增強 |
| **v4** | RoBERTa-base + Stratified 5-Fold + Soft Voting Ensemble | 0.8348 | 0.8305 | GPU 訓練完成，泛化與穩定性均優於 v3.1 |
| **v4.1** | RoBERTa-base + 5-Fold + Class Weights (WeightedTrainer) | 0.8348 | 0.8305 | 引入 balanced 權重處理資料不平衡 |
| **v5** | RoBERTa-large + Pseudo-labeling + FP16 | 0.8715 | 0.8695 | 引入 v4 預測作偽標籤，大幅突破準確率天花板 |
| **v6** | Two-Stage Self-Training + Confidence/Margin Filtering + OOF Threshold | 0.8546 | 0.8520 | Stage2 fold 波動下降；最佳 OOF 閾值 0.71 |
| **v6.1** | v6 + pseudo_loss_weight=0.3（降低偽標籤影響力） | 0.8694 | — | OOF F1 +0.0117；閾值從 0.71 回落至 0.61 |
| **v6.5** | DeBERTa-v3-large 替換 RoBERTa-large（同 Two-Stage 框架） | 0.8797 | — | RTD 預訓練 + Disentangled Attention，Stage2 OOF +0.0103 |
| **v6.6** | **三模型 Ensemble（RoBERTa + DeBERTa + ELECTRA）** | **0.8864** | **0.8865** | 等權重機率平均；ELECTRA 多樣性補足 MLM 系列誤差 |

### 關鍵里程碑

| 階段 | 代表版本 | 核心突破 |
| :--- | :--- | :--- |
| 傳統 ML 基線 | v1 → v2 | TF-IDF + LR，確立可比較基線 |
| Transformer 躍遷 | v3 → v3.1 | DistilBERT 引入語義理解，正則化控制過擬合 |
| 工程化穩定 | v4 → v4.1 | K-Fold + Soft Voting，從單次切分升級為穩定評估 |
| Large Model 突破 | v5 | RoBERTa-large + Pseudo-labeling，F1 突破 0.87 |
| 泛化優先迭代 | v6 → v6.1 | Two-Stage Self-Training + OOF 閾值校準，抑制過擬合 |
| 多樣性集成 | v6.5 → v6.6 | 跨架構（MLM × RTD）Ensemble，F1 達 0.8864 |

---

## 最終模型架構 (v6.6 Ensemble)

```
訓練資料 (gold labels)
       │
       ├─ Stage 1：K-Fold（gold-only）
       │       └─ OOF 預測 → pseudo label 篩選
       │              ├ confidence ≥ 0.92
       │              └ margin ≥ 0.35（類別平衡抽樣）
       │
       └─ Stage 2：K-Fold（gold + soft pseudo labels）
               └─ OOF 機率 → 最佳決策閾值搜尋

三個獨立模型（不同預訓練架構）：
  ① roberta-large      (Masked LM)        OOF F1 = 0.8694
  ② deberta-v3-large   (MLM + Disentangled Attn)  OOF F1 = 0.8797
  ③ electra-large      (Replaced Token Detection) OOF F1 = 0.8481

等權重機率平均 → OOF 最佳閾值 (0.61) → 最終預測
Ensemble OOF F1 = 0.8864  │  OOF Acc = 0.8865
```

---

## 快速開始

### 環境安裝

```bash
pip install -r requirements.txt
```

### 執行各版本

```bash
# v1 / v2：傳統 ML 基線
python src/version1.py
python src/version2.py

# v4：RoBERTa-base + K-Fold（需 GPU）
python src/version4.py --model_name roberta-base --num_folds 5 --epochs 3 --batch_size 16

# v5：RoBERTa-large + Pseudo-labeling（需 v4 輸出）
python src/version5.py --model_name roberta-large --pseudo_label_path result/submission_v4.csv \
    --batch_size 8 --grad_accum 2 --learning_rate 1e-5 --epochs 3

# v6.1：Two-Stage Self-Training（RoBERTa-large）
python src/version6.py \
    --model_name roberta-large \
    --num_folds 5 --epochs_stage1 2 --epochs_stage2 3 \
    --batch_size 8 --grad_accum 2 \
    --confidence_threshold 0.92 --margin_threshold 0.35 \
    --max_pseudo_per_class 3000 --pseudo_loss_weight 0.3 \
    --label_smoothing 0.05

# v6.5：DeBERTa-v3-large（關閉 FP16）
python src/version6.py \
    --model_name microsoft/deberta-v3-large \
    --output_dir result/version65_model \
    --submission_path result/submission_v65.csv \
    --disable_fp16 \
    --num_folds 5 --epochs_stage1 2 --epochs_stage2 3 \
    --batch_size 8 --grad_accum 2 \
    --confidence_threshold 0.92 --margin_threshold 0.35 \
    --max_pseudo_per_class 3000 --pseudo_loss_weight 0.3 \
    --label_smoothing 0.05

# v6.6：三模型 Ensemble
python src/ensemble_v66.py \
    --model_dirs result/version6_model result/version65_model result/version66_electra_model_v2 \
    --output_dir result/version66_ensemble_3model \
    --submission_path result/final_submission_v6.6_3model_equal.csv
```

---

## 檔案結構 (Project Structure)

```
.
├── data/
│   ├── train_2022.csv              # 原始訓練資料
│   ├── test_no_answer_2022.csv     # 測試資料（無標籤）
│   └── sample_submission.csv       # 提交格式範例
├── doc/
│   ├── Research Report.pdf         # 專案研究報告
│   └── figures/                    # 報告圖表
│       ├── fig1_class_distribution.png
│       ├── fig2_text_length.png
│       ├── fig3_top_words.png
│       ├── fig4_pipeline.png
│       ├── fig5_ensemble_arch.png
│       ├── fig6_public_progression.png
│       ├── fig7_oof_vs_public.png
│       └── generate_figures.py     # 圖表生成腳本
├── result/
│   ├── submission_v2.csv
│   ├── submission_v3.csv / v3.1.csv
│   ├── submission_v4.csv
│   ├── submission_v5.csv
│   ├── submission_v6.csv           # v6 / v6.1 提交檔
│   └── final_submission_v6.6_3model_equal.csv  # 最終提交（三模型集成）
├── src/
│   ├── version1.py                 # TF-IDF + Logistic Regression
│   ├── version2.py                 # Word + Char TF-IDF + LR
│   ├── version3.py                 # DistilBERT 微調
│   ├── version4.py                 # RoBERTa-base + 5-Fold + Soft Voting
│   ├── version5.py                 # RoBERTa-large + Pseudo-labeling + FP16
│   ├── version6.py                 # Two-Stage Self-Training（支援 RoBERTa / DeBERTa / ELECTRA）
│   ├── ensemble_v66.py             # 多模型機率平均集成腳本
│   └── docs/
│       ├── v1_strategy.md
│       ├── v2_strategy.md
│       ├── v3_strategy.md
│       ├── v4_strategy.md
│       ├── v5_strategy.md
│       └── v6_strategy.md          # 含 v6.1–v6.6 完整實測紀錄
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 詳細策略文件索引

各版本的完整策略、調參紀錄與實驗分析：

- [v1_strategy.md](src/docs/v1_strategy.md) — TF-IDF 基線設計
- [v2_strategy.md](src/docs/v2_strategy.md) — 字元 n-gram 擴充特徵
- [v3_strategy.md](src/docs/v3_strategy.md) — DistilBERT 微調與正則化
- [v4_strategy.md](src/docs/v4_strategy.md) — RoBERTa-base + K-Fold + Class Weights
- [v5_strategy.md](src/docs/v5_strategy.md) — RoBERTa-large + Pseudo-labeling
- [v6_strategy.md](src/docs/v6_strategy.md) — Two-Stage Self-Training 完整迭代（v6 ~ v6.6）

---

## 結論與設計心得

1. **模型架構升級是最有效的單一槓桿**：從 DistilBERT → RoBERTa-base → RoBERTa-large，每次升級帶來遠大於超參數調整的增益。
2. **Pseudo-labeling 需搭配嚴格篩選**：v6 的 confidence + margin 雙重過濾（門檻 0.92/0.35）比 v5 的直接使用所有偽標籤更能防止噪音污染。
3. **OOF 閾值校準不可省略**：固定 0.5 在資料分佈偏移時會帶來明顯偏差；v6.1 的最佳閾值 0.61 使 OOF F1 多提升 +0.012。
4. **跨架構集成勝過同架構堆疊**：ELECTRA（RTD 預訓練）單模 OOF F1 雖低於 RoBERTa/DeBERTa，但其系統性錯誤與 MLM 系列低相關，加入集成後整體 F1 仍提升 +0.002。
5. **Stage3 循環自訓練易過擬合**：v6.4 的 OOF F1 雖大幅提升，但 public score 反而下降，根本原因是循環確認偏差，應以更換基底模型取代不斷自我增強。
