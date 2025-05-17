import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score
from sklearn.model_selection import LeaveOneOut
from skimage.feature import hog
from torchvision.datasets import EMNIST
from torchvision import transforms
import torch
import seaborn as sns

# Mapping index ke karakter ASCII
label_to_char = {
    i: chr(ascii_code) for i, ascii_code in enumerate([
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57,       # 0–9
        65, 66, 67, 68, 69, 70, 71, 72, 73, 74,       # A–J
        75, 76, 77, 78, 79, 80, 81, 82, 83, 84,       # K–T
        85, 86, 87, 88, 89, 90,                      # U–Z
        97, 98, 100, 101, 102, 103, 104, 110, 113, 114, 116  # a–t (subset)
    ])
}

# Transformasi EMNIST (rotasi agar orientasi benar)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.rot90(x.squeeze(), 1, [0, 1]))
])

# Ambil dataset (train EMNIST balanced)
dataset = EMNIST(root='.', split='balanced', train=True, download=True, transform=transform)

# Batasi jumlah data untuk keperluan demonstrasi LOOCV (misal 100 sample)
N_SAMPLES = 100
images, labels = [], []
for i in range(N_SAMPLES):
    img, label = dataset[i]
    images.append(img.numpy())
    labels.append(label)

images = np.array(images)
labels = np.array(labels)

# Ekstraksi Fitur HOG
hog_features = []
hog_params = {
    'orientations': 9,
    'pixels_per_cell': (8,8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys'
}

for img in images:
    features = hog(img, **hog_params)
    hog_features.append(features)

hog_features = np.array(hog_features)

# Leave-One-Out Cross-Validation (LOOCV)
loo = LeaveOneOut()
y_true = []
y_pred = []

svm = SVC(kernel='linear', C=10)

print("Melakukan LOOCV pada", N_SAMPLES, "data...")
for train_idx, test_idx in loo.split(hog_features):
    X_train, X_test = hog_features[train_idx], hog_features[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    svm.fit(X_train, y_train)
    pred = svm.predict(X_test)
    y_true.append(y_test[0])
    y_pred.append(pred[0])

# Evaluasi
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
conf_mat = confusion_matrix(y_true, y_pred)

print("\n=== Hasil Evaluasi LOOCV ===")
print(f"Akurasi    : {accuracy:.4f}")
print(f"Presisi    : {precision:.4f}")
print(f"F1-score   : {f1:.4f}")
print("Confusion Matrix:")
print(conf_mat)

# Tampilkan confusion matrix dalam bentuk heatmap
plt.figure(figsize=(8,8))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
