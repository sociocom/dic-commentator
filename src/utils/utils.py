import pandas as pd
from datasets import Dataset


def load_csv_data(
    file_path, test_path, reliability_column, rank, input_column, predict_column
):
    """
    CSVファイルからデータを読み込み、Hugging FaceのDataset形式に変換する関数
    """
    df = pd.read_csv(file_path)

    df_skipped = df[df[f"{reliability_column}"] != "E"]
    # rank <- A
    ranks = "SABCDE"
    ix_train = ranks.index(rank) + 1  # => 1 + 1 = 2
    ranks_train = ranks[:ix_train]  # => "SA"
    str_ranks_train = r"|".join(r for r in ranks_train)  # => "S|A"
    mask_train = (
        df_skipped[f"{reliability_column}"].fillna("D").str.contains(str_ranks_train)
    )
    df_train = df_skipped[mask_train][
        [input_column, predict_column, f"{reliability_column}"]
    ].reset_index(drop=True)
    # df_test = df_skipped[~mask_train][
    #     ["ID", input_column, predict_column, f"{reliability_column}"]
    # ].reset_index(
    #     drop=True
    # )  # ~mask_train == not mask_train

    # データフレームをHugging Faceのデータセット形式に変換
    train_dataset = Dataset.from_pandas(df_train)

    df_test = pd.read_csv(test_path)
    df_test.rename(
        columns={"term": input_column, "normalized_term": predict_column}, inplace=True
    )
    test_dataset = Dataset.from_pandas(df_test)

    return train_dataset, test_dataset
