import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# データの読み込み
filtered_originals = [
    "〇〇シンドロム",
    "△△シンドロム",
    "□□シンドロム",
    "◇◇シンドロム",
    "◆◆シンドロム",
    "△△症候群",
    "□□症候群",
    "◇◇症候群",
    "◆◆症候群",
]
filtered_generated = [
    "〇〇シンドロームではないですか？",
    "△△シンドロームではないですか？",
    "□□シンドロームではないですか？",
    "◇◇シンドロームではないですか？",
    "◆◆シンドロームではないですか？",
    "△△症候群",
    "□□症候群",
    "◇◇症候群",
    "◆◆症候群",
]
filtered_embeddings = np.array(
    [
        [0.2, 0.1, 0.4, 0.3, 0.2, 0.1, 0.4, 0.3, 0.2],
        [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.6, 0.7, 0.8],
    ]
)  # 抽出済みのエンベディング

y_kmeans = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1])  # クラスタID

# データが存在するかの確認
if filtered_embeddings.size == 0:
    raise ValueError(
        "No data found in `filtered_embeddings`. Please check the filtering step."
    )

# フィードバックベクトルの定義
feedback_vectors = {
    0: np.array(
        [0.05, 0.0, -0.05, 0.05, 0.0, -0.05, 0.05, 0.0, -0.05]
    ),  # クラスタ0へのフィードバック
    1: np.array(
        [0.0, -0.05, 0.05, 0.05, 0.0, -0.05, 0.05, 0.0, -0.05]
    ),  # クラスタ1へのフィードバック
}

# フィードバック適用によるエンベディングの修正
updated_embeddings = np.array(
    [
        filtered_embeddings[i] + feedback_vectors[y_kmeans[i]]
        for i in range(len(filtered_embeddings))
    ]
)

# 修正後のクラスタの再計算
if updated_embeddings.ndim == 1:
    updated_embeddings = updated_embeddings.reshape(1, -1)

kmeans_updated = KMeans(n_clusters=2)
kmeans_updated.fit(updated_embeddings)
y_kmeans_updated = kmeans_updated.predict(updated_embeddings)

# 結果の可視化（オプション）
plt.figure(figsize=(8, 6))
plt.scatter(
    updated_embeddings[:, 0],
    updated_embeddings[:, 1],
    c=y_kmeans_updated,
    s=50,
    cmap="viridis",
)
updated_centers = kmeans_updated.cluster_centers_
plt.scatter(
    updated_centers[:, 0], updated_centers[:, 1], c="red", s=200, alpha=0.75, marker="X"
)

# タイトルとラベルの設定
plt.title("Clustering of Updated Error Patterns")
plt.xlabel("Embedding Dimension 1")
plt.ylabel("Embedding Dimension 2")
plt.show()
plt.savefig("clustering_updated_error_patterns.png")

print(filtered_originals)
print(filtered_generated)
print(list(updated_embeddings))
print(y_kmeans_updated)
# 結果の出力（オプション）
corrected_patterns = pd.DataFrame(
    {
        "original_text": filtered_originals,
        "generated_text": filtered_generated,
        "updated_embedding": list(updated_embeddings),
        "cluster_id": y_kmeans_updated,
    }
)
corrected_patterns.to_csv("corrected_patterns.csv", index=False)

print(corrected_patterns)
