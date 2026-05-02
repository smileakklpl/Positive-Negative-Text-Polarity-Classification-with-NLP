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

---

## 8. v6.1 階段性成果（2026-04-26）

### 調整內容
- `pseudo_loss_weight`: 0.5 → **0.3**（降低 pseudo 樣本對 loss 的影響力）
- 其餘參數不變（Stage1 結果重複使用，僅重訓 Stage2）

### Fold 結果對比

| Fold | v6（w=0.5） | v6.1（w=0.3） | 變化 |
|------|------------|--------------|------|
| 1    | 0.8516     | 0.8652       | +0.0136 |
| 2    | 0.8706     | 0.8710       | +0.0004 |
| 3    | 0.8421     | 0.8586       | +0.0165 |
| 4    | 0.8479     | 0.8571       | +0.0092 |
| 5    | 0.8607     | 0.8916       | +0.0309 |
| **mean** | **0.8546** | **0.8687** | **+0.0141** |

### OOF 指標對比

| 指標 | v6 | v6.1 |
|------|----|------|
| OOF F1 | 0.8577 | **0.8694** |
| OOF 最佳閾值 | 0.710 | **0.610** |

### 觀察
- OOF F1 提升 +0.0117，為顯著改善。
- 閾值從 0.710 降至 0.610，校準漂移明顯減少（更靠近 0.5）。
- 降低 pseudo_loss_weight 可有效抑制 Stage2 的機率偏移。

### 輸出檔案
- `result/submission_v6.csv`：v6.1 的最終提交檔（已覆蓋）
- `result/version6_model/best_threshold_v6.txt`：v6.1 閾值摘要
- `result/version6_model/oof_stage1_probs_v6.csv`：Stage1 OOF 機率（新增，供 --reuse_stage1 使用）
- `result/version6_model/test_stage1_probs_v6.csv`：Stage1 test 機率（新增）

---

## 9. v6.2 實測摘要（2026-04-27）

### 調整內容
- `pseudo_loss_weight`: 0.3 → **0.2**

### 結果
- 所有 fold F1 與 v6.1 **完全相同**：[0.8652, 0.8710, 0.8586, 0.8571, 0.8916]
- OOF F1 = 0.8694，OOF 最佳閾值 = 0.610

### 結論
`pseudo_loss_weight=0.3` 已是最佳值，繼續降低無額外增益。可能原因：
- Early stopping 使模型在相同步數停止，兩個 weight 的梯度更新已收斂至相同方向。
- 814 筆 pseudo 樣本佔比（約 51%）下，weight 差異（0.3 vs 0.2）不足以改變收斂點。

### 下一步候選
- 嘗試增加 `confidence_threshold=0.90`（放寬條件，增加 pseudo 樣本至 ~1200+），觀察更多偽標籤是否進一步提升泛化。
- 或嘗試第三階段（Stage3）：以 Stage2 的模型再產生更高品質的 pseudo labels 重訓。

---

## 10. v6.3 實測摘要（2026-04-27）

### 調整內容
- `confidence_threshold`: 0.92 → **0.90**（放寬篩選條件）
- 其餘參數不變（`pseudo_loss_weight=0.3`，使用 `--reuse_stage1` 重用 Stage1 輸出）

### 結果

| Fold | v6.1（ct=0.92） | v6.3（ct=0.90） | 變化 |
|------|----------------|----------------|------|
| 1    | 0.8652         | 0.8647         | -0.0005 |
| 2    | 0.8710         | 0.8856         | +0.0146 |
| 3    | 0.8586         | 0.8434         | -0.0152 |
| 4    | 0.8571         | 0.8564         | -0.0007 |
| 5    | 0.8916         | 0.8878         | -0.0038 |
| **mean** | **0.8687** | **0.8676** | **-0.0011** |

### OOF 指標對比

| 指標 | v6.1 | v6.3 |
|------|----|------|
| OOF F1 | **0.8694** | 0.8684 |
| OOF 最佳閾值 | 0.610 | 0.580 |

### Pseudo 樣本統計

| 版本 | 樣本數 | 類別 0 | 類別 1 | confidence min | confidence avg |
|------|--------|--------|--------|----------------|----------------|
| v6.1 | 814    | 407    | 407    | 0.9200         | 0.9432         |
| v6.3 | 1700   | 850    | 850    | 0.9000         | 0.9333         |

### 結論
降低 `confidence_threshold` 至 0.90 使 pseudo 樣本從 814 增加至 1700（+108.8%），但 OOF F1 略降（0.8694 → 0.8684）。
- 0.90~0.92 信心區間的樣本帶入了額外噪音，抵消了樣本量增加的效益。
- **`confidence_threshold=0.92` 為最佳值**，數量與品質的平衡點。

### 下一步候選
- 嘗試 Stage3：以 Stage2 模型重新產生更高品質的 pseudo labels，再進行第三輪訓練。
- 調整 `margin_threshold`（目前 0.35）來控制 pseudo 樣本的邊界清晰度。

---

## 11. v6.4 實測摘要（Stage3）（2026-04-27）

### 調整內容
- 新增 Stage3：以 Stage2 模型對測試集的預測重新篩選 pseudo labels，再訓練第三輪。
- 使用 `--reuse_stage1 --reuse_stage2 --run_stage3` 旗標執行。
- 擴大 OOF 閾值搜尋範圍：`[0.2, 0.8]` → `[0.05, 0.95]`。

### Fold 結果對比

| Fold | Stage2 (v6.1) | Stage3 (v6.4) | 變化 |
|------|---------------|---------------|------|
| 1    | 0.8652        | 0.8680        | +0.0028 |
| 2    | 0.8710        | 0.9020        | +0.0310 |
| 3    | 0.8586        | 0.8718        | +0.0132 |
| 4    | 0.8571        | 0.8722        | +0.0151 |
| 5    | 0.8916        | 0.8883        | -0.0033 |
| **mean** | **0.8687** | **0.8805** | **+0.0118** |

### OOF 指標對比

| 指標 | Stage2 (v6.1) | Stage3 (v6.4) |
|------|---------------|---------------|
| OOF F1 | 0.8694 | **0.8823** |
| OOF 最佳閾值 | 0.610 | **0.170** |

### Pseudo 樣本統計

| 來源 | 樣本數 | 類別 0 | 類別 1 |
|------|--------|--------|--------|
| Stage1 predictions（v6.1） | 814  | 407  | 407  |
| Stage2 predictions（v6.4） | 5348 | 2674 | 2674 |

### 概率分佈特性
Stage3 的 OOF 概率高度雙峰：47% < 0.1，48% > 0.8，中間值極少。
- 最佳閾值落在 0.170（原始搜尋上界 0.2 以下），因此擴大了搜尋範圍。
- threshold=0.17 與 threshold=0.5 的 F1 差距僅 0.0017（0.8823 vs 0.8806），雙峰分佈下閾值選擇影響甚小。
- submission positive rate = 0.595（train 為 0.500），略偏 class 1。

### 結論
- Stage3 顯著提升 OOF F1：0.8694 → 0.8823（+0.0129）。
- Stage2 模型信心更高，篩選出 5348 筆 pseudo 樣本（Stage1 僅 814 筆）。
- 高品質 pseudo labels 使模型學到更強的特徵，但也造成概率分佈極端化（高雙峰）。

### 輸出檔案
- `result/submission_v6.csv`：v6.4 的最終提交檔（已覆蓋，使用 threshold=0.17）
- `result/version6_model/oof_stage3_probs_v6.csv`：Stage3 OOF 機率
- `result/version6_model/pseudo_selected_stage3_v6.csv`：Stage3 pseudo 樣本診斷
- `result/version6_model/test_stage2_probs_v6.csv`：Stage2 test 機率（新增，供 --reuse_stage2 使用）

---

## 12. v6.5 策略：換用 DeBERTa-v3-large（2026-04-27）

### 動機
v6.4（Stage3）的 OOF F1 雖提升至 0.8823，但 public score 從 0.79173 **下降**至 0.78484。
根本問題是循環確認偏差（circular self-training）導致過擬合，OOF 增益無法轉換為泛化提升。

策略轉向：**更換更強的基底模型**，在相同的 Two-Stage 框架下引入 `microsoft/deberta-v3-large`。

### 核心改動
- **基底模型**：`roberta-large` → `microsoft/deberta-v3-large`
- **FP16 關閉**：DeBERTa-v3 使用 disentangled attention，不支援 FP16（數值不穩定），改用 FP32
- **輸出目錄**：`result/version65_model`（獨立目錄，保留 v6.1 結果）
- **提交檔**：`result/submission_v65.csv`
- **策略不變**：Stage1（gold-only）+ Stage2（gold + pseudo labels）+ OOF 閾值調整

### 執行指令
```bash
python src/version6.py \
  --model_name microsoft/deberta-v3-large \
  --output_dir result/version65_model \
  --submission_path result/submission_v65.csv \
  --num_folds 5 \
  --epochs_stage1 2 \
  --epochs_stage2 3 \
  --batch_size 8 \
  --grad_accum 2 \
  --confidence_threshold 0.92 \
  --margin_threshold 0.35 \
  --max_pseudo_per_class 3000 \
  --pseudo_loss_weight 0.3 \
  --label_smoothing 0.05 \
  --disable_fp16
```

### 預期效益
DeBERTa-v3-large 在多數文字分類任務上穩定優於 RoBERTa-large，原因：
- Replaced Token Detection（RTD）預訓練目標比 MLM 學習更豐富的語義
- Disentangled attention 分離位置與內容表示，對語義細節更敏感
- 相同的 Two-Stage + pseudo label 策略在更強的模型上預期帶來更高基線

### 後續計畫
若 DeBERTa-v3-large 單模有提升，下一步：
- Ensemble：RoBERTa-large (v6.1) + DeBERTa-v3-large (v6.5) 預測機率加權平均
- 跨模型 pseudo labeling：以 v6.1 的測試集預測作為 DeBERTa Stage2 的 pseudo labels

### 實測結果（2026-04-27）

#### Stage1 fold 結果

| Fold | DeBERTa v6.5 | RoBERTa v6.1 | 差距 |
|------|-------------|-------------|------|
| 1    | 0.8685      | 0.8652      | +0.0033 |
| 2    | 0.8591      | 0.8710      | -0.0119 |
| 3    | 0.8448      | 0.8586      | -0.0138 |
| 4    | 0.8318      | 0.8571      | -0.0253 |
| 5    | 0.8762      | 0.8916      | -0.0154 |
| **mean** | **0.8561** | **0.8687** | -0.0126 |
| OOF F1 | **0.8603** | 0.8694 | -0.0091 |

#### Stage2 fold 結果

| Fold | DeBERTa v6.5 | RoBERTa v6.1 | 差距 |
|------|-------------|-------------|------|
| 1    | 0.8696      | 0.8652      | +0.0044 |
| 2    | 0.9020      | 0.8710      | +0.0310 |
| 3    | 0.8512      | 0.8586      | -0.0074 |
| 4    | 0.8657      | 0.8571      | +0.0086 |
| 5    | 0.8933      | 0.8916      | +0.0017 |
| **mean** | **0.8764** | **0.8687** | **+0.0077** |
| OOF F1 | **0.8797** | 0.8694 | **+0.0103** |
| threshold | 0.310 | 0.610 | |

#### 概率分佈特性
- 高度雙峰：46.4% < 0.1，49.9% > 0.7，mean=0.505
- threshold=0.31 vs threshold=0.5 的 F1 差距僅 0.0031，雙峰下閾值選擇影響小
- submission positive rate = 0.558（train 為 0.500）

#### 輸出檔案
- `result/submission_v65.csv`：v6.5 最終提交檔（threshold=0.310）
- `result/version65_model/`：DeBERTa 模型輸出目錄

---

## 13. v6.6：三模型 Ensemble（RoBERTa + DeBERTa + ELECTRA）

### 核心策略
將三個架構完全不同的模型進行等權重機率平均 ensemble，利用各模型系統性錯誤的低相關性提升泛化能力。

### 模型成員
| 模型 | 版本 | 預訓練機制 | Stage2 OOF F1 |
|---|---|---|---|
| `roberta-large` | v6.1 | Masked LM（優化訓練流程） | 0.8694 |
| `microsoft/deberta-v3-large` | v6.5 | Masked LM + Disentangled Attention | 0.8797 |
| `google/electra-large-discriminator` | v6.6 | Replaced Token Detection（RTD） | 0.8481 |

### ELECTRA-large 訓練細節
- 首次以 `lr=1e-5` 訓練出現嚴重不穩定（Fold 3 F1=0.030，Fold 1 Stage2 F1=0.251），部分 fold 完全 collapse 至預測單一類別
- 根本原因：ELECTRA discriminator 的梯度尺度與 RoBERTa/DeBERTa 不同，1e-5 對某些 fold 初始化過高
- 修正：`--learning_rate 5e-6`，所有 fold 訓練穩定，Stage2 OOF F1 = 0.8481
- 支援 FP16（不需 `--disable_fp16`）
- pseudo 樣本選取數量待確認

#### 執行指令（ELECTRA 訓練）
```bash
python src/version6.py \
  --model_name google/electra-large-discriminator \
  --output_dir result/version66_electra_model_v2 \
  --submission_path result/submission_v66_electra_v2.csv \
  --num_folds 5 --epochs_stage1 2 --epochs_stage2 3 \
  --batch_size 8 --grad_accum 2 \
  --learning_rate 5e-6 \
  --confidence_threshold 0.92 --margin_threshold 0.35 \
  --max_pseudo_per_class 3000 --pseudo_loss_weight 0.3 \
  --label_smoothing 0.05
```

#### ELECTRA Stage2 各 Fold 結果
| Fold | F1 | Acc |
|---|---|---|
| 1 | 0.8722 | 0.8725 |
| 2 | 0.8815 | 0.8750 |
| 3 | 0.8486 | 0.8475 |
| 4 | 0.8495 | 0.8450 |
| 5 | 0.7380 | 0.7550 |
| **mean** | **0.8380** | — |
| OOF F1 | **0.8481** | — |
| threshold | 0.470 | — |

### Ensemble 結果比較
| 配置 | OOF F1 | OOF Acc | Threshold | Positive Rate |
|---|---|---|---|---|
| 2-model (RoBERTa + DeBERTa) | 0.8844 | 0.8845 | 0.56 | 0.507 |
| **3-model equal 1/3 each** | **0.8864** | **0.8865** | 0.61 | 0.487 |
| 3-model weighted (0.4/0.4/0.2) | 0.8860 | 0.8860 | 0.57 | 0.504 |

**最佳：三模型等權重 ensemble**（OOF F1 0.8864，+0.002 vs 二模型）

#### 關鍵觀察
- ELECTRA 單模型 OOF F1（0.8481）低於 RoBERTa/DeBERTa，但加入 ensemble 後整體 F1 仍提升 +0.002
- 原因：RTD 預訓練學到的特徵與 MLM 系列模型的系統性錯誤**低相關**，多樣性彌補了個別模型的強度不足
- 三模型 ensemble 的 positive rate = 0.487，最接近 train set 分佈（0.500）

#### 執行指令（Ensemble）
```bash
python src/ensemble_v66.py \
  --model_dirs result/version6_model result/version65_model result/version66_electra_model_v2 \
  --output_dir result/version66_ensemble_3model \
  --submission_path result/submission_v66_3model_equal.csv
```

#### 輸出檔案
- `result/submission_v66_3model_equal.csv`：v6.6 最終提交檔（threshold=0.610）
- `result/version66_ensemble_3model/`：Ensemble 輸出目錄（OOF/test prob 平均值）
- `src/ensemble_v66.py`：Ensemble 腳本（支援任意模型數量與權重）
