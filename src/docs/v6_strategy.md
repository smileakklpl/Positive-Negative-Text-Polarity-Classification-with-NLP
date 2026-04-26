# v6 策略：以泛化能力優先

## 1. 目標
- 以提升 Private Leaderboard 穩定性為主，而非過度擬合 Public 33% 切分。
- 只挑選高信心且高區分度的測試樣本，降低 pseudo-label 噪音。
- 在 out-of-fold (OOF) 機率上調整分類閾值，讓 F1 校準更準確。

## 2. 核心設計
1. 二階段訓練：
- Stage 1：僅用 gold labels 進行 K-Fold 訓練。
- Stage 2：使用 gold labels + 篩選後 pseudo labels 重新進行 K-Fold 訓練。

2. 信心度過濾的 pseudo labeling：
- 只保留滿足 `max(prob) >= confidence_threshold` 的 pseudo 樣本。
- 同時要求 `abs(prob_1 - prob_0) >= margin_threshold`。
- 使用類別平衡抽樣，避免單一類別主導 pseudo 資料。

3. Soft pseudo labels：
- 以機率分佈向量作為 pseudo 目標，而不是硬標籤。
- 透過 `pseudo_loss_weight` 降低 pseudo 樣本在 loss 中的影響力。

4. OOF 閾值調整：
- 在 OOF 機率上搜尋最佳 decision threshold。
- 最終測試預測使用最佳閾值（非固定 0.5）。

5. 泛化正則化：
- 在 gold labels 使用 label smoothing。
- soft-label loss 可選擇加入 class weights。
- 使用 early stopping 並回載最佳 checkpoint。

## 3. 主要輸出檔案
- `result/submission_v6.csv`：最終提交檔。
- `result/version6_model/cv_metrics_v6.csv`：stage 1 與 stage 2 的 fold 指標。
- `result/version6_model/oof_stage2_probs_v6.csv`：用於閾值分析的 OOF 機率。
- `result/version6_model/test_probs_v6.csv`：最終平均後的測試集機率。
- `result/version6_model/pseudo_selected_v6.csv`：被選中的 pseudo 樣本（含 confidence、margin）。
- `result/version6_model/best_threshold_v6.txt`：最佳 OOF 閾值與分數摘要。

## 4. 建議執行指令
```bash
python src/version6.py \
  --model_name roberta-large \
  --num_folds 5 \
  --epochs_stage1 2 \
  --epochs_stage2 3 \
  --batch_size 8 \
  --grad_accum 2 \
  --confidence_threshold 0.92 \
  --margin_threshold 0.35 \
  --max_pseudo_per_class 3000 \
  --pseudo_loss_weight 0.5 \
  --label_smoothing 0.05
```

## 5. 實務調參順序
1. 先調 pseudo 篩選：
- `confidence_threshold` 建議範圍 [0.90, 0.96]
- `margin_threshold` 建議範圍 [0.25, 0.45]

2. 再調閾值校準：
- `threshold_grid_step` 建議維持較小（0.01 或 0.005）。
- 確認 OOF F1 相較固定 0.5 是否有提升。

3. 最後調 pseudo 影響力：
- `pseudo_loss_weight` 建議範圍 [0.3, 0.8]
- 太高可能造成模型漂移，太低則無法有效利用 pseudo 資料。

## 6. 為什麼 v6 更穩健
- 不會對每筆 pseudo label 給予相同信任度。
- 類別平衡抽樣可避免 pseudo 樣本向單一類別崩塌。
- 透過 OOF 閾值校準，使預測更貼合競賽評分指標。
- 設計目標是穩定與泛化，而不是單次切分運氣分。

## 7. 本次實測摘要（2026-04-26）
- Stage1（gold-only）: F1 mean = 0.8350，F1 std = 0.0124，Acc mean = 0.8320
- Stage2（gold+pseudo）: F1 mean = 0.8546，F1 std = 0.0100，Acc mean = 0.8520
- OOF 最佳閾值：
  - Stage1: 0.420（OOF F1 = 0.8422）
  - Stage2: 0.710（OOF F1 = 0.8577）
- Pseudo 樣本：
  - 選入 814 筆（0 類 407 / 1 類 407，類別平衡）
  - confidence: min 0.9200 / avg 0.9432 / max 0.9750
  - margin: min 0.8401 / avg 0.8864 / max 0.9500

以上結果顯示：v6 在提升平均分數的同時，也降低了 fold 波動，符合「泛化優先」目標。
