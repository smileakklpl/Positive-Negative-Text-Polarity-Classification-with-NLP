# v4 & v4.1 策略與實作說明

## 1. 目標
- 從單次切分驗證升級為更穩定的評估方式。
- 提升相較於 v3.1 的泛化能力。
- 保留 Hugging Face checkpoint 的彈性切換能力。

## 2. 核心策略
1. 驗證方式：使用 Stratified K-Fold（正式執行為 5 folds）。
2. 模型：以 RoBERTa-base 為主，可透過 `--model_name` 切換。
3. 訓練方式：每個 fold 各自訓練一個模型，並以 `eval_f1` 選擇最佳 checkpoint。
4. 集成方式：對測試集預測結果做 soft voting，取各類別機率平均後再決策。
5. **v4.1 新增策略 (Class Weights)：** 透過覆寫 `Trainer.compute_loss` 實作自訂的 `WeightedTrainer`，並使用 `sklearn` 算出的 `balanced` class weights 傳入 CrossEntropyLoss 中。此舉旨在增加少數類別分類錯誤時的懲罰，解決分類資料不平衡的潛在不足。

## 3. 執行環境
- 訓練腳本：`src/version4.py`
- 輸出檔案：
  - `result/submission_v4.csv`
  - `result/version4_model/cv_metrics_v4.csv`
- 執行環境：GPU（PyTorch CUDA 版本，支援 RTX 5070 Ti）

## 4. 建議參數
- `--model_name roberta-base`
- `--num_folds 5`
- `--epochs 3`
- `--learning_rate 2e-5`
- `--weight_decay 0.05`
- `--batch_size 16`（若顯存不足可降為 8）

## 5. 執行範例
```bash
python src/version4.py --model_name roberta-base --num_folds 5 --epochs 3 --batch_size 16
```

其他可切換模型：
```bash
python src/version4.py --model_name bert-base-uncased --num_folds 5 --epochs 3 --batch_size 16
python src/version4.py --model_name distilbert-base-uncased --num_folds 5 --epochs 3 --batch_size 16
```

## 6. 為什麼 v4 比 v3.1 更好
1. 驗證方式改變：從 holdout 改成 K-Fold，降低切分隨機性。
2. 預測方式改變：從單模型輸出改成 soft-voting ensemble，提升穩定性。
3. 執行方式改變：使用可用於 GPU 的 runtime，能實際完成完整 fold 訓練。
4. **v4.1 改進點：** 針對資料分類不平衡，在 Loss 函數中加入動態計算的 Class Weights，讓模型在訓練過程中不再強烈偏向佔多數的類別。
