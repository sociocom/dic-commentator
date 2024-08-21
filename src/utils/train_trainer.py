from transformers import Trainer, TrainingArguments
from utils.model import load_model_and_tokenizer
from datasets import DatasetDict


class T5FineTuner:
    def __init__(
        self,
        model_name="t5-small",
        max_length=512,
        batch_size=8,
        epochs=1,
        learning_rate=5e-5,
        val_split=0.1,
        input_column="input_text",
        target_column="target_text",
        device="cuda:0",
    ):
        """
        T5のファインチューニングを行うためのクラス
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.val_split = val_split  # 訓練データを分割する割合
        self.input_column = input_column
        self.target_column = target_column
        self.device = device

        # モデルとトークナイザーのロード
        self.model, self.tokenizer = load_model_and_tokenizer(model_name, device=device)

        # モデルをデバイスに移動
        self.model = self.model.to(self.device)

    def preprocess_function(self, examples):
        """
        データセットの前処理を行う関数
        """
        inputs = [
            "医療用語の出現形「" + str(term) + "」を正規形に変換してください。"
            for term in examples[self.input_column]
        ]
        targets = examples[self.target_column]
        model_inputs = self.tokenizer(
            inputs, max_length=self.max_length, truncation=True, padding="max_length"
        )

        # ラベルをトークナイズしてモデル入力に追加
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                targets,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def load_data(self, dataset: DatasetDict):
        """
        CSVファイルからデータセットをロードし、訓練データを分割して前処理を適用する
        """
        # データを訓練セットと検証セットに分割
        train_test_split = dataset.train_test_split(test_size=self.val_split)
        self.train_dataset = train_test_split["train"].map(
            self.preprocess_function,
            batched=True,
        )
        self.val_dataset = train_test_split["test"].map(
            self.preprocess_function, batched=True
        )

    def train(self, output_dir="t5_finetuned"):
        """
        モデルのファインチューニングを行う
        """
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            weight_decay=0.01,
            logging_dir="./logs",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
        )

        trainer.train()

        # モデルとトークナイザーを保存
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
