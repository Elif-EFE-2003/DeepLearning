"""
XOR Problemi - Yapay Sinir Ağı Çözümü
=====================================
Tek Katmanlı Perceptron (başarısız) vs
Çok Katmanlı Perceptron / MLP (başarılı)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],   # 0 XOR 0 = 0
              [1],   # 0 XOR 1 = 1
              [1],   # 1 XOR 0 = 1
              [0]])  # 1 XOR 1 = 0


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

def step(x):
    return np.where(x >= 0.5, 1, 0)

# TEK KATMANLI PERCEPTRON (Linear - XOR'u çözemez)
class SingleLayerPerceptron:
    def __init__(self, lr=0.1, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.losses = []

    def fit(self, X, y):
        np.random.seed(42)
        self.W = np.random.randn(X.shape[1], 1) * 0.1
        self.b = np.zeros((1, 1))

        for _ in range(self.epochs):
            out = sigmoid(X @ self.W + self.b)
            loss = np.mean((y - out) ** 2)
            self.losses.append(loss)
            grad = -(y - out) * sigmoid_deriv(X @ self.W + self.b)
            self.W -= self.lr * X.T @ grad
            self.b -= self.lr * grad.sum()

    def predict(self, X):
        return step(sigmoid(X @ self.W + self.b))

# ÇOK KATMANLI PERCEPTRON / MLP (XOR'u çözer!)
# Mimari: 2 → 4 → 1
class MLP:
    def __init__(self, hidden=4, lr=0.5, epochs=10000):
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.losses = []

    def fit(self, X, y):
        np.random.seed(42)
        # Ağırlıklar
        self.W1 = np.random.randn(X.shape[1], self.hidden)
        self.b1 = np.zeros((1, self.hidden))
        self.W2 = np.random.randn(self.hidden, 1)
        self.b2 = np.zeros((1, 1))

        for _ in range(self.epochs):
            # --- Ileri ---
            self.z1  = X @ self.W1 + self.b1
            self.a1  = sigmoid(self.z1)
            self.z2  = self.a1 @ self.W2 + self.b2
            self.out = sigmoid(self.z2)

            loss = np.mean((y - self.out) ** 2)
            self.losses.append(loss)

            # --- Geri ---
            d_out = -(y - self.out) * sigmoid_deriv(self.z2)
            dW2   = self.a1.T @ d_out
            db2   = d_out.sum(axis=0)

            d_a1  = d_out @ self.W2.T
            d_z1  = d_a1 * sigmoid_deriv(self.z1)
            dW1   = X.T @ d_z1
            db1   = d_z1.sum(axis=0)

            # --- Güncelleme ---
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1

    def predict(self, X):
        a1  = sigmoid(X @ self.W1 + self.b1)
        out = sigmoid(a1 @ self.W2 + self.b2)
        return step(out)

    def predict_proba(self, X):
        a1  = sigmoid(X @ self.W1 + self.b1)
        return sigmoid(a1 @ self.W2 + self.b2)

# EĞİTİM
print("=" * 50)
print("XOR PROBLEMİ - SİNİR AĞI KARŞILAŞTIRMASI")
print("=" * 50)

slp = SingleLayerPerceptron(lr=0.1, epochs=1000)
slp.fit(X, y)

mlp = MLP(hidden=4, lr=0.5, epochs=10000)
mlp.fit(X, y)

slp_preds = slp.predict(X)
mlp_preds = mlp.predict(X)

print("\n--- Tek Katmanlı Perceptron ---")
print(f"{'Girdi':<12} {'Beklenen':<12} {'Tahmin':<12} {'Doğru?'}")
print("-" * 45)
for i in range(4):
    dogru = "✓" if slp_preds[i][0] == y[i][0] else "✗"
    print(f"{str(X[i]):<12} {y[i][0]:<12} {slp_preds[i][0]:<12} {dogru}")
slp_acc = np.mean(slp_preds == y) * 100
print(f"\nDoğruluk: %{slp_acc:.1f}")

print("\n--- Çok Katmanlı Perceptron (MLP) ---")
print(f"{'Girdi':<12} {'Beklenen':<12} {'Tahmin':<12} {'Doğru?'}")
print("-" * 45)
for i in range(4):
    dogru = "✓" if mlp_preds[i][0] == y[i][0] else "✗"
    print(f"{str(X[i]):<12} {y[i][0]:<12} {mlp_preds[i][0]:<12} {dogru}")
mlp_acc = np.mean(mlp_preds == y) * 100
print(f"\nDoğruluk: %{mlp_acc:.1f}")

fig = plt.figure(figsize=(16, 12))
fig.suptitle("XOR Problemi — Tek Katmanlı vs Çok Katmanlı Perceptron",
             fontsize=16, fontweight='bold')

ax1 = fig.add_subplot(2, 3, 1)
ax1.plot(slp.losses, color='#F44336', linewidth=2, label='Tek Katmanlı')
ax1.set_title('Tek Katmanlı — Loss', fontweight='bold')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('MSE Loss')
ax1.grid(alpha=0.3); ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
ax1.annotate(f'Son Loss: {slp.losses[-1]:.4f}', xy=(0.6, 0.8), xycoords='axes fraction',
             color='#F44336', fontsize=10, fontweight='bold')

ax2 = fig.add_subplot(2, 3, 2)
ax2.plot(mlp.losses, color='#4CAF50', linewidth=2, label='MLP')
ax2.set_title('MLP — Loss', fontweight='bold')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('MSE Loss')
ax2.grid(alpha=0.3); ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
ax2.annotate(f'Son Loss: {mlp.losses[-1]:.4f}', xy=(0.6, 0.8), xycoords='axes fraction',
             color='#4CAF50', fontsize=10, fontweight='bold')

xx, yy = np.meshgrid(np.linspace(-0.3, 1.3, 300), np.linspace(-0.3, 1.3, 300))
grid = np.c_[xx.ravel(), yy.ravel()]

slp_grid = step(sigmoid(grid @ slp.W + slp.b)).reshape(xx.shape)
ax3 = fig.add_subplot(2, 3, 4)
ax3.contourf(xx, yy, slp_grid, alpha=0.3, cmap='RdYlGn')
for i, (xi, yi) in enumerate(zip(X, y)):
    color = '#4CAF50' if yi[0] == 1 else '#F44336'
    ax3.scatter(*xi, c=color, s=200, zorder=5, edgecolors='black', linewidth=1.5)
    ax3.annotate(f'({xi[0]},{xi[1]})→{yi[0]}', xi, textcoords='offset points',
                 xytext=(8, 6), fontsize=9)
ax3.set_title('Tek Katmanlı — Karar Sınırı\n(Lineer → XOR çözülemez!)',
              fontweight='bold', color='#F44336')
ax3.set_xlim(-0.3, 1.3); ax3.set_ylim(-0.3, 1.3)
ax3.grid(alpha=0.3)

mlp_grid = mlp.predict_proba(grid).reshape(xx.shape)
ax4 = fig.add_subplot(2, 3, 5)
ax4.contourf(xx, yy, mlp_grid, alpha=0.3, cmap='RdYlGn')
ax4.contour(xx, yy, mlp_grid, levels=[0.5], colors='navy', linewidths=2)
for i, (xi, yi) in enumerate(zip(X, y)):
    color = '#4CAF50' if yi[0] == 1 else '#F44336'
    ax4.scatter(*xi, c=color, s=200, zorder=5, edgecolors='black', linewidth=1.5)
    ax4.annotate(f'({xi[0]},{xi[1]})→{yi[0]}', xi, textcoords='offset points',
                 xytext=(8, 6), fontsize=9)
ax4.set_title('MLP — Karar Sınırı\n(Doğrusal olmayan → XOR çözüldü!)',
              fontweight='bold', color='#4CAF50')
ax4.set_xlim(-0.3, 1.3); ax4.set_ylim(-0.3, 1.3)
ax4.grid(alpha=0.3)

ax5 = fig.add_subplot(2, 3, 3)
models = ['Tek Katmanlı\nPerceptron', 'MLP\n(2→4→1)']
accs   = [slp_acc, mlp_acc]
colors = ['#F44336', '#4CAF50']
bars = ax5.bar(models, accs, color=colors, width=0.5, edgecolor='white', linewidth=2)
ax5.set_ylim(0, 115)
ax5.set_ylabel('Doğruluk (%)', fontsize=11)
ax5.set_title('Model Karşılaştırması', fontweight='bold')
for bar, acc in zip(bars, accs):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             f'%{acc:.1f}', ha='center', fontsize=13, fontweight='bold')
ax5.axhline(100, color='navy', linestyle='--', alpha=0.4, linewidth=1)
ax5.grid(axis='y', alpha=0.3)
ax5.spines['top'].set_visible(False); ax5.spines['right'].set_visible(False)

ax6 = fig.add_subplot(2, 3, 6)
ax6.set_xlim(0, 10); ax6.set_ylim(0, 10); ax6.axis('off')
ax6.set_title('MLP Mimarisi: 2 → 4 → 1', fontweight='bold')

def node(ax, x, y, label, color):
    ax.text(x, y, label, ha='center', va='center', fontsize=11, fontweight='bold',
            color='white', bbox=dict(boxstyle='circle,pad=0.4', facecolor=color, linewidth=2))

# Giris
for cy, lbl in zip([7, 3], ['x₁', 'x₂']):
    node(ax6, 1.5, cy, lbl, '#1565C0')

# Gizli
for i, cy in enumerate([8, 5.5, 3, 0.8]):
    node(ax6, 5, cy, f'h{i+1}', '#F57C00')

# Cikis
node(ax6, 8.5, 5, 'y', '#2E7D32')

for iy in [7, 3]:
    for hy in [8, 5.5, 3, 0.8]:
        ax6.annotate('', xy=(4.6, hy), xytext=(1.9, iy),
                     arrowprops=dict(arrowstyle='->', color='gray', lw=1))
for hy in [8, 5.5, 3, 0.8]:
    ax6.annotate('', xy=(8.1, 5), xytext=(5.4, hy),
                 arrowprops=dict(arrowstyle='->', color='gray', lw=1))

ax6.text(1.5, 9.5, 'Girdi\nKatmanı', ha='center', fontsize=9, color='#1565C0', fontweight='bold')
ax6.text(5,   9.5, 'Gizli\nKatman', ha='center', fontsize=9, color='#F57C00', fontweight='bold')
ax6.text(8.5, 9.5, 'Çıkış\nKatmanı', ha='center', fontsize=9, color='#2E7D32', fontweight='bold')

plt.tight_layout()
plt.savefig('xor_neural_network.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nGörsel 'xor_neural_network.png' olarak kaydedildi.")
print("\nKütüphane gereksinimleri: numpy, matplotlib (scikit-learn gerekmez!)")