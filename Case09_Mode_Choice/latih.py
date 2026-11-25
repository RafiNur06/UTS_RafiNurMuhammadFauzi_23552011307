
import os
import json
import joblib
import numpy as np
import pandas as pd

from fitur import feature_engineering

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


# ==========================================
# 1. KONFIGURASI
# ==========================================

RANDOM_STATE = 42

DATA_PATH = "dataset/kasus09_mode_choice.csv"

MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


# ==========================================
# 2. MEMBACA DATASET DAN FEATURE ENGINEERING
# ==========================================

print("=" * 55)
print("TAHAP 1: FEATURE ENGINEERING")
print("=" * 55)

df = pd.read_csv(DATA_PATH)

data = feature_engineering(df)

print("Ukuran dataset awal:", df.shape)
print("Ukuran setelah transformasi:", data.shape)


# ==========================================
# 3. MENENTUKAN FITUR DAN TARGET
# ==========================================

X = data.drop(
    columns=["individual", "target"]
)

y = data["target"].astype(int)

feature_names = X.columns.tolist()

print("\nJumlah fitur:", len(feature_names))
print("Jumlah kelas:", y.nunique())

print("\nDistribusi target:")
print(y.value_counts().sort_index())


# ==========================================
# 4. MEMBAGI DATA TRAINING DAN TESTING
# ==========================================

print("\n" + "=" * 55)
print("TAHAP 2: TRAIN-TEST SPLIT")
print("=" * 55)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("Data training:", X_train.shape)
print("Data testing:", X_test.shape)

print("\nDistribusi target training:")
print(y_train.value_counts().sort_index())

print("\nDistribusi target testing:")
print(y_test.value_counts().sort_index())


# ==========================================
# 5. MEMBUAT PIPELINE MODEL
# ==========================================

print("\n" + "=" * 55)
print("TAHAP 3: PEMBUATAN MODEL")
print("=" * 55)

model_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE
    ))
])

model_rf = Pipeline([
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE
    ))
])

models = {
    "Logistic Regression": model_lr,
    "Random Forest": model_rf
}

print("Logistic Regression berhasil dibuat.")
print("Random Forest berhasil dibuat.")


# ==========================================
# 6. CROSS VALIDATION
# ==========================================

print("\n" + "=" * 55)
print("TAHAP 4: STRATIFIED 5-FOLD CROSS VALIDATION")
print("=" * 55)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

scoring = {
    "accuracy": "accuracy",
    "f1_macro": "f1_macro"
}

hasil_cv = []

for nama, model in models.items():

    print(f"\nEvaluasi: {nama}")

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    hasil_cv.append({
        "Model": nama,
        "Accuracy Mean": scores["test_accuracy"].mean(),
        "Accuracy Std": scores["test_accuracy"].std(),
        "F1 Macro Mean": scores["test_f1_macro"].mean(),
        "F1 Macro Std": scores["test_f1_macro"].std()
    })


hasil_cv_df = pd.DataFrame(hasil_cv)

print("\nHasil Cross Validation:")
print(
    hasil_cv_df.sort_values(
        by="F1 Macro Mean",
        ascending=False
    ).to_string(index=False)
)


# ==========================================
# 7. MEMILIH MODEL TERBAIK
# ==========================================

print("\n" + "=" * 55)
print("TAHAP 5: PEMILIHAN MODEL")
print("=" * 55)

best_name = hasil_cv_df.sort_values(
    by="F1 Macro Mean",
    ascending=False
).iloc[0]["Model"]

best_model = models[best_name]

print("Model terbaik:", best_name)


# ==========================================
# 8. TRAINING MODEL TERBAIK
# ==========================================

print("\n" + "=" * 55)
print("TAHAP 6: FINAL TRAINING")
print("=" * 55)

best_model.fit(X_train, y_train)

print("Training berhasil.")
print("Jumlah data training:", len(X_train))


# ==========================================
# 9. EVALUASI DATA TESTING SATU KALI
# ==========================================

print("\n" + "=" * 55)
print("TAHAP 7: EVALUASI AKHIR DATA TESTING")
print("=" * 55)

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

print("\nHASIL EVALUASI")
print("-" * 35)
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")

print("\nClassification Report:")

report = classification_report(
    y_test,
    y_pred,
    zero_division=0
)

print(report)


# ==========================================
# 10. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=best_model.classes_
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=best_model.classes_
)

disp.plot(cmap="Blues")

plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.show()


# ==========================================
# 11. MENYIMPAN MODEL DAN METADATA
# ==========================================

print("\n" + "=" * 55)
print("TAHAP 8: DEPLOYMENT JOBLIB")
print("=" * 55)

metadata = {
    "model_name": best_name,
    "dataset": "kasus09_mode_choice.csv",
    "target_name": "target",
    "feature_names": feature_names,
    "class_labels": [
        int(x) for x in best_model.classes_
    ],
    "metrics": {
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1)
    }
}

artifact = {
    "pipeline": best_model,
    "metadata": metadata
}

model_path = os.path.join(
    MODEL_DIR,
    "random_forest_mode_choice.joblib"
)

joblib.dump(artifact, model_path)

# Menyimpan hasil evaluasi
hasil_cv_df.to_csv(
    os.path.join(MODEL_DIR, "hasil_cross_validation.csv"),
    index=False
)

with open(
    os.path.join(MODEL_DIR, "metadata.json"),
    "w"
) as file:
    json.dump(metadata, file, indent=4)

print("Model berhasil disimpan!")
print("Lokasi:", model_path)
print("Jumlah fitur:", len(feature_names))
print("Model terpilih:", best_name)

print("\nSEMUA PROSES SELESAI!")
