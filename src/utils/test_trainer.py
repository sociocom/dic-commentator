# src/test_trainer.py

import torch
from utils.model import load_model_and_tokenizer
from utils.utils import load_csv_data
import pandas as pd


class T5Tester:
    def __init__(
        self,
        model_dir="t5_finetuned",
        input_column="input_text",
        target_column="target_text",
        device="cuda:0",
        max_length=8,
    ):
        """
        ファインチューニングされたT5モデルのテストを行うクラス
        """
        # ファインチューニングされたモデルとトークナイザーをロード
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_dir=model_dir, device=device
        )
        self.model.eval()  # モデルを評価モードに設定
        self.model = self.model.to(device)  # モデルをデバイスに移動
        self.input_column = input_column
        self.target_column = target_column
        self.device = device
        self.max_length = max_length

    def generate_text(self, input_text):
        """
        入力テキストから生成されたテキストを取得する関数
        """
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(
            self.device
        )
        generated_ids = self.model.generate(input_ids, max_length=self.max_length)
        generated_text = self.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        )
        return generated_text

    def calculate_confidence(self, input_text, generated_text):
        """
        生成されたテキストの信頼度を計算する関数
        """
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(
            self.device
        )
        target_ids = self.tokenizer(generated_text, return_tensors="pt").input_ids.to(
            self.device
        )

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, labels=target_ids)
            log_probs = outputs.logits.log_softmax(dim=-1)

        # 各トークンの信頼度（確率）を取得
        token_confidences = torch.exp(
            log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
        )

        # シーケンス全体のログ確率の和
        sequence_log_prob = token_confidences.log().sum()

        # 信頼度（確率に変換）
        sequence_confidence = torch.exp(sequence_log_prob).item()

        return sequence_confidence

    def test(self, test_file):
        """
        テストデータを使って生成と信頼度計算を行う
        """
        test_dataset = load_csv_data(test_file)
        input_texts = [example[self.input_column] for example in test_dataset]
        target_texts = [example[self.target_column] for example in test_dataset]
        generated_texts = []
        confidences = []

        for example in test_dataset:
            input_text = (
                "医療用語の出現形「" + str(example) + "」を正規形に変換してください。"
            )
            generated_text = self.generate_text(input_text)
            confidence = self.calculate_confidence(input_text, generated_text)
            generated_texts.append(generated_text)
            confidences.append(confidence)

        time_stamp = pd.Timestamp.now().strftime("%Y:%m:%d-%H:%M:%S")
        df = pd.DataFrame(
            {
                self.input_column: input_texts,
                self.target_column: target_texts,
                "generated_text": generated_texts,
                "confidence": confidences,
            }
        )
        df.to_csv(f"results/test_results{time_stamp}.csv", index=False)
