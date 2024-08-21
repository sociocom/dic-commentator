# src/main.py
import os
from utils.train_trainer import T5FineTuner
from utils.test_trainer import T5Tester
import torch


def main(
    model_name="retrieva-jp/t5-small-long",
    train_file="data/disease_sample100_20240816.csv",
    test_file="data/disease_train200_20240816.csv",
    input_column="term",
    target_column="normalized_term",
    device="cuda:0" if torch.cuda.is_available() else "cpu",
):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用するGPUのIDを指定
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # ファインチューニング
    fine_tuner = T5FineTuner(
        model_name=model_name,
        max_length=8,
        batch_size=8,
        epochs=3,
        learning_rate=5e-5,
        val_split=0.1,  # 訓練データの10%を検証データとして分割
        input_column=input_column,
        target_column=target_column,
        device=device,
    )
    fine_tuner.load_data(train_file=train_file)
    fine_tuner.train(output_dir="results")
    # テスト
    tester = T5Tester(
        model_dir="results",
        input_column=input_column,
        target_column=target_column,
        device=device,
        max_length=8,
    )
    tester.test(test_file=test_file)


if __name__ == "__main__":
    main()
