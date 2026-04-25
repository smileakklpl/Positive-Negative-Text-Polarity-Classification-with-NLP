# v5 策略與實作說明 (RoBERTa-large + Pseudo-labeling)

## 1. 目標
- 解決「訓練集遠小於測試集」的挑戰。
- 透過大語言模型架構 (Large Model) 突破目前的 F1-score 天花板。
- 確保在有限的 GPU 資源 (16GB VRAM) 下順利完成訓練。

## 2. 核心策略
1. **Pseudo-labeling (偽標籤)**：
   - 讀取 `v4` 產出的 `submission_v4.csv` 作為偽標籤資料。
   - **安全整合**：在 K-Fold 切分後，只將偽標籤資料加入每個 fold 的「訓練集 (Train Split)」，驗證集 (Validation Split) 仍保持為純淨的原始資料，避免驗證失真與 Data Leakage。
2. **模型升級 (RoBERTa-large)**：
   - 模型參數由 base 版 (110M) 升級至 large 版 (355M)，具備更深層的語意理解能力。
   - 學習率 (Learning Rate) 由 `2e-5` 調降為 `1e-5`，適應 Large 模型的微調特性。
3. **硬體資源最佳化 (防 OOM)**：
   - 啟用 **FP16 混合精度訓練** (`fp16=True`)。
   - 調降實體 Batch Size 至 `8`。
   - 啟用 **梯度累積 (Gradient Accumulation)**，設定 `grad_accum=2`，等同於維持邏輯 Batch Size 為 16 (`8 * 2`)。

## 3. 執行環境
- 訓練腳本：`src/version5.py`
- 輸出檔案：
  - `result/submission_v5.csv`
- 執行環境：GPU（RTX 5070 Ti 或同等顯卡，支援 FP16）

## 4. 建議參數與執行範例
預設參數已針對 `roberta-large` 與 RTX 5070 Ti 最佳化：

```bash
# 執行前請確保 result/submission_v4.csv 存在
python src/version5.py \
    --model_name roberta-large \
    --pseudo_label_path result/submission_v4.csv \
    --batch_size 8 \
    --grad_accum 2 \
    --learning_rate 1e-5 \
    --epochs 3
```