# src/utils.py

import pandas as pd
from datasets import Dataset

def load_csv_data(file_path):
    """
    CSVファイルからデータを読み込み、Hugging FaceのDataset形式に変換する関数
    """
    df = pd.read_csv(file_path)

    # データフレームをHugging Faceのデータセット形式に変換
    dataset = Dataset.from_pandas(df)

    return dataset