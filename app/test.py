import os
import cv2
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH = "../model/model.pkl"
pipeline = joblib.load(MODEL_PATH)
pipeline.named_steps["knn"].n_jobs = 1

IMG_SIZE = 128
LABELS = ["Normal", "Katarak"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(BASE_DIR, "../../dataset/test")

y_true = []
y_pred = []

def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    img = cv2.equalizeHist(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.flatten().reshape(1, -1) / 255.0
    return img

for label in LABELS:
    folder = os.path.join(TEST_DIR, label)
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        img = preprocess_image(file_path)
        if img is None:
            continue
        
        pred = pipeline.predict(img)[0]
        y_true.append(LABELS.index(label))
        y_pred.append(pred)

print("\n Evaluasi Model pada Folder Test")
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=LABELS))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

##tes push