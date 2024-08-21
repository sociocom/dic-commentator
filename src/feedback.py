import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1. データの読み込み
# エラーパターンの抽出プロジェクトから得られたデータを想定
filtered_originals = ["〇〇シンドロム", "△△症候群"]
filtered_generated = ["〇〇シンドロームではないですか？", "△△症候群"]

# 抽出済みのエンベディングとクラスタID
filtered_embeddings = np.array([[0.2, 0.1, 0.4], [0.4, 0.5, 0.6]])
y_kmeans = np.array([0, 1])  # クラスタID

# 2. フィードバックベクトルの定義
# 各クラスタに対する修正用フィードバックベクトル
feedback_vectors = {
    0: np.array([0.05, 0.0, -0.05]),  # クラスタ0へのフィードバック
    1: np.array([0.0, -0.05, 0.05]),  # クラスタ1へのフィードバック
}

# 3. フィードバック適用によるエンベディングの修正
updated_embeddings = np.array(
    [
        filtered_embeddings[i] + feedback_vectors[y_kmeans[i]]
        for i in range(len(filtered_embeddings))
    ]
)

# 4. 修正後のクラスタの再計算（オプション）
# 必要に応じてクラスタリングを再計算する場合
kmeans_updated = KMeans(n_clusters=2)
kmeans_updated.fit(updated_embeddings)
y_kmeans_updated = kmeans_updated.predict(updated_embeddings)

# 5. 結果の可視化（オプション）
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

# 6. 結果の出力（オプション）
# 修正後のエラーパターンとクラスタIDをCSVに保存
corrected_patterns = pd.DataFrame(
    {
        "original_text": filtered_originals,
        "generated_text": filtered_generated,
        "updated_embedding": list(updated_embeddings),
        "cluster_id": y_kmeans_updated,
    }
)
corrected_patterns.to_csv("corrected_patterns.csv", index=False)

# 結果の表示（オプション）
print(corrected_patterns)
