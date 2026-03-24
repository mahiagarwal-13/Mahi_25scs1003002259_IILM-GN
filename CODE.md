# Mahi_25scs1003002259_IILM-GN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import warnings
warnings.filterwarnings("ignore")

file_path = "/content/Student Stress Factors csv.csv"  # change if path/name different
df = pd.read_csv(file_path)

print("First 5 rows:")
display(df.head())

print("\nColumns in dataset:")
print(df.columns.tolist())

print("\nInfo:")
df.info()

df = df.dropna(how="all")
df = df.dropna()

print("Shape after dropping missing rows:", df.shape)

possible_targets = [c for c in df.columns if "stress" in c.lower()]
print("\nColumns that look like stress target:", possible_targets)

if not possible_targets:
    raise ValueError("No column containing the word 'stress' found. Rename your target column properly.")

TARGET_COL = possible_targets[0]
print("\nUsing target column:", TARGET_COL)

print("\nUnique values in target:")
print(df[TARGET_COL].value_counts())

feature_cols = [c for c in df.columns if c != TARGET_COL]

print("Feature columns being used:")
print(feature_cols)

X = df[feature_cols].copy()
y = df[TARGET_COL].copy()

feature_cols = [c for c in df.columns if c != TARGET_COL]

print("Feature columns being used:")
print(feature_cols)

X = df[feature_cols].copy()
y = df[TARGET_COL].copy()

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Label mapping:")
for cls, idx in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
    print(f"{cls} -> {idx}")

    X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

for col in feature_cols:
    plt.figure()
    plt.hist(df[col], bins=20, edgecolor="black")
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.show()

# Correlation heatmap (encode target as numeric)
corr_df = df[feature_cols].copy()
corr_df[TARGET_COL] = y_encoded

corr_matrix = corr_df.corr()

plt.figure(figsize=(8,6))
plt.imshow(corr_matrix, interpolation="nearest")
plt.colorbar()
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45, ha="right")
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

models = {}

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
models["Logistic Regression"] = log_reg

# Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42
)
rf.fit(X_train, y_train)  # no scaling needed
models["Random Forest"] = rf

# Neural Network
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42
)
mlp.fit(X_train_scaled, y_train)
models["Neural Network (MLP)"] = mlp

print("Models trained successfully."

def evaluate_model(name, model):
    print("=" * 60)
    print(f"Model: {name}")
    print("=" * 60)
 if name == "Random Forest":
        X_tr_use, X_te_use = X_train, X_test
    else:
        X_tr_use, X_te_use = X_train_scaled, X_test_scaled

    y_train_pred = model.predict(X_tr_use)
    y_test_pred = model.predict(X_te_use)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc  = accuracy_score(y_test, y_test_pred)

    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy : {test_acc:.4f}\n")

    # ---- FIX HERE: make target names strings ----
    class_names = [str(c) for c in label_encoder.classes_]

    print("Classification Report (Test):")
    print(classification_report(
        y_test,
        y_test_pred,
        target_names=class_names
    ))

    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(5,4))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
       	    plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.show()

    importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8,4))
plt.title("Feature Importances (Random Forest)")
plt.bar(range(len(feature_cols)), importances[indices])
plt.xticks(range(len(feature_cols)), [feature_cols[i] for i in indices], rotation=45, ha="right")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()

print("Feature importances:")
for i in indices:
    print(f"{feature_cols[i]}: {importances[i]:.4f}")

    def predict_stress_level(sample_dict, model=models["Random Forest"]):
    """
    sample_dict = {
        'col1': value,
        'col2': value,
        ...
    }
    """
    sample_df = pd.DataFrame([sample_dict])
    pred_encoded = model.predict(sample_df)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]
    return pred_label

# EXAMPLE: change keys to your real feature column names
example_input = {}

for col in feature_cols:
    example_input[col] = float(input(f"Enter value for {col}: "))

pred = predict_stress_level(example_input)
print("\nPredicted stress level:", pred)
