# dic-commentator

dic-commentator 医療辞書管理コメントシステム

[dicomplete](https://github.com/sociocom/dicomplete)をもとに機能追加させたシステム

## コマンド一覧

help が見れるコマンド: ```rye run python src/main.py --help```<br>
入力すると以下の説明が見られる

```
NAME
    main.py

SYNOPSIS
    main.py <flags>

FLAGS
    -m, --model_name=MODEL_NAME
        Type: str
        Default: 'retrieva-jp/t5-small-long'
    -f, --fpath=FPATH
        Type: str
        Default: 'data/disease_t...
    -t, --test_fpath=TEST_FPATH
        Type: str
        Default: 'data/disease_...
    -i, --input_column=INPUT_COLUMN
        Type: str
        Default: 'term'
    -p, --predict_column=PREDICT_COLUMN
        Type: str
        Default: 'normalized_term'
    -d, --device=DEVICE
        Type: str
        Default: 'cuda:0'
    --reliability_column=RELIABILITY_COLUMN
        Type: str
        Default: '正規形_flag'
    --rank=RANK
        Type: str
        Default: 'C'
    -e, --epochs=EPOCHS
        Type: int
        Default: 1
    -b, --batch_size=BATCH_SIZE
        Type: int
        Default: 8
```
