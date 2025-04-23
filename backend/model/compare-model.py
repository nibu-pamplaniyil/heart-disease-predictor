import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("..\\dataset\\heart_attack_prediction_dataset.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Remove nulls
df.dropna(inplace=True)

# Fix age (from 0.66 style)
df['Age'] = (df['Age'] * 100).astype(int)

# Fix Exercise Frequency column
df['Exercise Hours per week'] = (df['Exercise Hours Per Week'] * 10).astype(int)

# Encode categorical strings to numeric
df = pd.get_dummies(df, drop_first=True)

# Separate features and target
X = df.drop("Heart Attack Risk (Binary)", axis=1)
y = df["Heart Attack Risk (Binary)"]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Models to compare
models = {
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(probability=True)
}

# Evaluate each model
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else np.zeros(len(y_test))

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, y_proba)
    })

# Show results
result_df = pd.DataFrame(results)
print(result_df)

joblib.dump(result_df   , "heart_attack_model.pkl")


sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))

# Melt the dataframe for easier plotting
melted = result_df.melt(id_vars="Model", 
                        value_vars=["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
                        var_name="Metric",
                        value_name="Score")

# Barplot
sns.barplot(data=melted, x="Model", y="Score", hue="Metric")

plt.title("Model Performance Comparison", fontsize=16)
plt.xticks(rotation=15)
plt.ylim(0, 1)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()