import os
from utils.train_trainer import T5FineTuner
from utils.test_trainer import T5Tester
from utils.utils import load_csv_data
import torch
import fire


def main(
    model_name: str = "retrieva-jp/t5-small-long",
    fpath: str = "data/disease_train200_20240816.csv",
    test_fpath: str = "data/disease_sample100_20240816.csv",
    input_column: str = "term",
    predict_column: str = "normalized_term",
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    reliability_column: str = "正規形_flag",
    rank: str = "C",
    epochs: int = 1,
    batch_size: int = 8,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用するGPUのIDを指定
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # データの読み込み
    train_dataset, test_dataset = load_csv_data(
        fpath, test_fpath, reliability_column, rank, input_column, predict_column
    )
    # ファインチューニング
    fine_tuner = T5FineTuner(
        model_name=model_name,
        max_length=8,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=5e-5,
        val_split=0.1,  # 訓練データの10%を検証データとして分割
        input_column=input_column,
        target_column=predict_column,
        device=device,
    )
    fine_tuner.load_data(train_dataset)
    fine_tuner.train(output_dir="results")
    # テスト
    tester = T5Tester(
        model_dir="results",
        input_column=input_column,
        target_column=predict_column,
        device=device,
        max_length=8,
    )
    tester.test(test_dataset, epoch=epochs)


if __name__ == "__main__":
    fire.Fire(main)
