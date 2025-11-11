import os
import cv2
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score


DATASET_PATH = "../../Dataset/train/"
IMG_SIZE = 128
CLASSES = ["normal", "katarak"]

X, y = [], []

for label, cls in enumerate(CLASSES):
    folder = os.path.join(DATASET_PATH, cls)
    if not os.path.exists(folder):
        print(f"⚠️ Folder {folder} tidak ditemukan, lewati.")
        continue

    for file in os.listdir(folder):
        img_path = os.path.join(folder, file)
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Resize dan normalisasi
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype("float32") / 255.0

        # Sedikit peningkatan kontras untuk bantu PCA menangkap tepi
        img = cv2.equalizeHist((img * 255).astype(np.uint8)) / 255.0

        X.append(img.flatten())
        y.append(label)

X = np.array(X)
y = np.array(y)

print(f"Total data: {len(X)} (Normal: {sum(y == 0)}, Katarak: {sum(y == 1)})")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(whiten=True, random_state=42)),
    ('knn', KNeighborsClassifier(weights='distance'))
])


param_grid = {
    'pca__n_components': [50, 80, 100, 120, 150],
    'knn__n_neighbors': [3, 5, 7, 9],
    'knn__p': [1, 2],
    'knn__leaf_size': [10, 20, 30]
}

print("🔍 Mencari parameter terbaik (Grid Search)...")
grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=7,  # naikkan lipatan validasi biar hasilnya stabil
    n_jobs=-1,
    verbose=1,
    scoring='f1_weighted'
)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print(f"\n🌟 Parameter terbaik: {grid.best_params_}")

y_pred = best_model.predict(X_test)

print("\n📊 Hasil Evaluasi:")
print(classification_report(y_test, y_pred, target_names=CLASSES))

acc = accuracy_score(y_test, y_pred)
print(f"✅ Akurasi: {acc * 100:.2f}%")

os.makedirs("../model", exist_ok=True)
joblib.dump(best_model, "../model/model.pkl")
print("\n💾 Model berhasil disimpan ke '../model/model.pkl'")
