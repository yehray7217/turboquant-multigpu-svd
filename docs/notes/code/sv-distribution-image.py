import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
import time

# 載入資料（第一次會自動下載，約 200MB）
print("Loading LFW dataset...")
faces = fetch_lfw_people(
    min_faces_per_person=1,
    resize=1.0,
    color=False,
    slice_=(slice(0, 250), slice(0, 250))  # 保留完整 250×250
)
all_images = faces.images  # shape: (n_samples, 250, 250)

n_samples, h, w = all_images.shape
print(f"Loaded {n_samples} images, each {h}×{w}")

rng = np.random.default_rng(seed=int(time.time()))
selected_15 = rng.choice(n_samples, size=15, replace=False)
selected_5  = rng.choice(selected_15, size=5, replace=False)

# ── 圖1：15 張圖各自的 singular values 衰減曲線 ──
fig, ax = plt.subplots(figsize=(10, 5))

svd_results = {}  # 存每張圖的 SVD 結果備用

for idx in selected_15:
    img = all_images[idx]  # (125, 125)
    U, s, Vt = np.linalg.svd(img, full_matrices=False)
    svd_results[idx] = (U, s, Vt)

    s_norm = s / s[0]  # 正規化，方便比較形狀
    ax.semilogy(s_norm, alpha=0.6, linewidth=1.2)

ax.set_title(f'Singular Value Distribution (15 random LFW faces, {h}×{w}, log scale, normalized)')
ax.set_xlabel('Singular Value Index')
ax.set_ylabel('Normalized Singular Value')
plt.tight_layout()
plt.savefig('sv_distribution_15faces.png', dpi=150)
plt.show()

# ── 圖2：5 張圖，原圖 vs energy 近似 ──
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

ENERGY = 0.999

for col, idx in enumerate(selected_5):
    img = all_images[idx]
    U, s, Vt = svd_results[idx]

    # 計算對應 energy 需要幾個 singular values
    cumulative_energy = np.cumsum(s**2) / np.sum(s**2)
    k_energy = int(np.searchsorted(cumulative_energy, ENERGY)) + 1

    # rank-k 近似
    img_k = U[:, :k_energy] @ np.diag(s[:k_energy]) @ Vt[:k_energy, :]

    axes[0, col].imshow(img, cmap='gray')
    axes[0, col].set_title(f'Original #{idx}')
    axes[0, col].axis('off')

    axes[1, col].imshow(img_k, cmap='gray')
    axes[1, col].set_title(f'{ENERGY * 100:.1f}% energy\n(k={k_energy}/{min(h, w)})')
    axes[1, col].axis('off')

plt.suptitle(f'Low-rank Approximation: Original vs {ENERGY * 100:.1f}% Energy (LFW {h}×{w})', y=1.02)
plt.tight_layout()
plt.savefig(f'test_face_reconstruction_{ENERGY}pct.png', dpi=150)
plt.show()

# ── 數字報告 ──
print(f"\n{ENERGY * 100:.1f}% energy approximation summary:")
for idx in selected_5:
    _, s, _ = svd_results[idx]
    cumulative_energy = np.cumsum(s**2) / np.sum(s**2)
    k_energy = int(np.searchsorted(cumulative_energy, ENERGY)) + 1
    print(f"  Face #{idx:4d} → {k_energy:3d} / {min(h, w)} singular values")
