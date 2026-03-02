"""
Meme Kanseri Sınıflandırma Analizi
Breast Cancer Wisconsin Dataset
Model: Random Forest Classifier
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 1. VERİ YÜKLEME

df = pd.read_csv('data.csv')

print("=" * 50)
print("VERİ SETİ BİLGİLERİ")
print("=" * 50)
print(f"Boyut: {df.shape}")
print(f"\nSınıf Dağılımı:\n{df['diagnosis'].value_counts()}")
print(f"\nEksik Değer:\n{df.isnull().sum().sum()}")

# 2. VERİ ÖN İŞLEME

X = df.drop(['id', 'diagnosis'], axis=1)
y = (df['diagnosis'] == 'M').astype(int)  # M=1 (Malignant), B=0 (Benign)

# Train/Test split (%80 / %20), stratified
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalizasyon (StandardScaler)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"\nEğitim seti: {X_train_s.shape}")
print(f"Test seti:   {X_test_s.shape}")

# 3. MODEL EĞİTİMİ

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_s, y_train)

# 4. TAHMİN VE DEĞERLENDİRME

y_pred = model.predict(X_test_s)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
cm   = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 50)
print("PERFORMANS METRİKLERİ")
print("=" * 50)
print(f"Accuracy  (Doğruluk)  : {acc:.4f}  ({acc*100:.2f}%)")
print(f"Precision (Kesinlik)  : {prec:.4f}  ({prec*100:.2f}%)")
print(f"Recall    (Duyarlılık): {rec:.4f}  ({rec*100:.2f}%)")
print(f"F1-Score              : {f1:.4f}  ({f1*100:.2f}%)")

print("\nConfusion Matrix:")
print(cm)

print("\nDetaylı Rapor:")
print(classification_report(y_test, y_pred, target_names=['Benign (B)', 'Malignant (M)']))

# Özellik önemleri
feat_imp = pd.Series(
    model.feature_importances_, index=X.columns
).sort_values(ascending=False)

print("Top 10 Önemli Özellik:")
print(feat_imp.head(10).to_string())

# 5. GÖRSELLEŞTİRME

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('Breast Cancer Classification — Random Forest\nWisconsin Dataset',
             fontsize=16, fontweight='bold', y=0.98)

# --- Confusion Matrix ---
ax1 = axes[0, 0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=['Benign (B)', 'Malignant (M)'],
            yticklabels=['Benign (B)', 'Malignant (M)'],
            linewidths=0.5, linecolor='gray')
ax1.set_title('Confusion Matrix', fontsize=13, fontweight='bold')
ax1.set_ylabel('Gerçek Değer', fontsize=11)
ax1.set_xlabel('Tahmin Edilen', fontsize=11)

# --- Metrik Bar Chart ---
ax2 = axes[0, 1]
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values  = [acc, prec, rec, f1]
colors  = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
bars = ax2.bar(metrics, values, color=colors, width=0.6, edgecolor='white', linewidth=1.5)
ax2.set_ylim(0.85, 1.02)
ax2.set_title('Performans Metrikleri', fontsize=13, fontweight='bold')
ax2.set_ylabel('Skor', fontsize=11)
for bar, val in zip(bars, values):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
             f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# --- Feature Importance ---
ax3 = axes[1, 0]
feat_imp.head(10).sort_values().plot(kind='barh', ax=ax3, color='#1976D2', edgecolor='white')
ax3.set_title('Top 10 Önemli Özellik', fontsize=13, fontweight='bold')
ax3.set_xlabel('Önem Skoru', fontsize=11)
ax3.grid(axis='x', alpha=0.3)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# --- Sınıf Dağılımı Pie Chart ---
ax4 = axes[1, 1]
counts = df['diagnosis'].value_counts()
wedges, texts, autotexts = ax4.pie(
    counts, labels=['Benign (B)', 'Malignant (M)'],
    autopct='%1.1f%%', colors=['#4CAF50', '#F44336'],
    startangle=90, pctdistance=0.75,
    wedgeprops={'edgecolor': 'white', 'linewidth': 2}
)
for at in autotexts:
    at.set_fontsize(12)
    at.set_fontweight('bold')
ax4.set_title(f'Sınıf Dağılımı (n={len(df)})', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('breast_cancer_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nGörsel 'breast_cancer_analysis.png' olarak kaydedildi.")