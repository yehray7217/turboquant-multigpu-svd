import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def fwht(x):
    """Fast Walsh-Hadamard Transform, length must be power of 2"""
    n = len(x)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                a, b = x[j], x[j + h]
                x[j], x[j + h] = a + b, a - b
        h *= 2
    return x / np.sqrt(n)

def rht(x, seed=0):
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=len(x))
    return fwht(signs * x)
    #return x
    #return signs * x

def make_paper_random_rotation(d, seed=123):
    """Dense O(d^2) Gaussian transform: R_ij ~ N(0, 1/d)."""
    rng = np.random.default_rng(seed)
    std = 1.0 / np.sqrt(d)
    return rng.normal(0.0, std, size=(d, d)).astype(np.float32)

def paper_random_rotation(x, rotation):
    return rotation @ x.astype(np.float32, copy=False)

d = 4096
theoretical_std = 1.0 / np.sqrt(d)
x_range = np.linspace(-5 * theoretical_std, 5 * theoretical_std, 500)
theoretical_pdf = stats.norm.pdf(x_range, 0, theoretical_std)

paper_rotation = make_paper_random_rotation(d)

def plot_vectors(ax, vectors, colors, title, transform, xlabel):
    for idx, (name, x) in enumerate(vectors):
        y = transform(x.copy(), idx)
        outputs = y                           # 全部 d 個維度就是樣本

        mean = outputs.mean()
        std = outputs.std()
        _, p_value = stats.kstest(outputs, 'norm', args=(0, theoretical_std))
        print(f"{name:<40} mean={mean:+.6f}  std={std:.6f}  KS p={p_value:.4f}")

        kde = stats.gaussian_kde(outputs)
        kde_x = np.linspace(outputs.min(), outputs.max(), 300)
        ax.plot(kde_x, kde(kde_x), color=colors[idx], alpha=0.5, linewidth=0.8)

    ax.plot(x_range, theoretical_pdf, 'k--', linewidth=2, label=f'N(0, 1/{d})')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()

# --- Figure 1: clustered structured vectors ---
def make_structured_vectors(d, n_vectors=50, seed=7):
    rng = np.random.default_rng(seed)
    vectors = []

    for k in range(1, n_vectors + 1):
        x = np.zeros(d)
        base_group_size = d // k
        start = 0

        for group_idx in range(1, k + 1):
            if group_idx == k:
                end = d
            else:
                end = start + base_group_size

            group_size = end - start
            mean = float(group_idx)
            std = np.sqrt(group_idx)
            x[start:end] = rng.normal(mean, std, size=group_size)
            start = end

        x /= np.linalg.norm(x)
        vectors.append((f"{k} Gaussian clusters", x))

    return vectors

# --- Figure 2: random unit vectors ---
def make_random_vectors(d, n_vectors=50, seed=42):
    rng = np.random.default_rng(seed)
    vectors = []
    for i in range(n_vectors):
        x = rng.standard_normal(d)
        x /= np.linalg.norm(x)
        vectors.append((f"random unit vector {i+1}", x))
    return vectors

structured = make_structured_vectors(d, n_vectors=50)
random_vecs = make_random_vectors(d, n_vectors=50)

cmap1 = plt.cm.coolwarm
cmap2 = plt.cm.tab10
colors1 = [cmap1(i / (len(structured) - 1)) for i in range(len(structured))]
colors2 = [cmap2((i % 10) / 10) for i in range(len(random_vecs))]

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

print(f"\n=== RHT: Structured vectors (d={d}) ===")
plot_vectors(ax1, structured, colors1,
             f'RHT: clustered structured inputs\n(d={d}, one RHT each, all {d} dims)',
             lambda x, seed: rht(x, seed=seed),
             'RHT output value')

sm = plt.cm.ScalarMappable(cmap=cmap1, norm=plt.Normalize(0, 1))
sm.set_array([])
cbar = plt.colorbar(sm, ax=[ax1, ax3])
cbar.set_label('Number of clusters')
cbar.set_ticks([0, 1])
cbar.set_ticklabels(['k=1', 'k=50'])

print(f"\n=== RHT: Random unit vectors (d={d}) ===")
plot_vectors(ax2, random_vecs, colors2,
             f'RHT: random unit vectors\n(d={d}, one RHT each, all {d} dims)',
             lambda x, seed: rht(x, seed=seed),
             'RHT output value')

print(f"\n=== Dense Gaussian O(d^2): Structured vectors (d={d}) ===")
plot_vectors(ax3, structured, colors1,
             f'Dense Gaussian O(d^2): clustered structured inputs\n(d={d}, one fixed matrix, all {d} dims)',
             lambda x, seed: paper_random_rotation(x, paper_rotation),
             'Dense Gaussian output value')

print(f"\n=== Dense Gaussian O(d^2): Random unit vectors (d={d}) ===")
plot_vectors(ax4, random_vecs, colors2,
             f'Dense Gaussian O(d^2): random unit vectors\n(d={d}, one fixed matrix, all {d} dims)',
             lambda x, seed: paper_random_rotation(x, paper_rotation),
             'Dense Gaussian output value')

plt.tight_layout()
plt.savefig('rht_distribution.png', dpi=150)
plt.show()
