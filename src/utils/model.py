from transformers import T5Tokenizer, T5ForConditionalGeneration


def load_model_and_tokenizer(model_name="t5-small", model_dir=None, device="cuda:0"):
    """
    T5モデルとトークナイザーをロードする関数
    """
    if model_dir:
        tokenizer = T5Tokenizer.from_pretrained(model_dir)
        model = T5ForConditionalGeneration.from_pretrained(model_dir)
    else:
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)

    # モデルをデバイスに移動
    model = model.to(device)

    return model, tokenizer
