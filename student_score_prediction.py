# ==========================================================
# 🎓 Student Score Prediction Project
# ==========================================================

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ----------------------------------------------------------
# Step 1: Load Dataset
# ----------------------------------------------------------
df = pd.read_csv("dataset_study.csv")

print("Dataset Preview:")
print(df.head())

# ----------------------------------------------------------
# Step 2: Define Features and Target
# ----------------------------------------------------------
X = df[["study_hours"]]   # change if column name different
y = df["grade"]           # change if column name different

# ----------------------------------------------------------
# Step 3: Train-Test Split
# ----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------------------------------
# Step 4: Train Model
# ----------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ----------------------------------------------------------
# Step 5: Evaluate Model
# ----------------------------------------------------------
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("R² Score:", round(r2, 3))

# ----------------------------------------------------------
# Step 6: Predict for 4 Study Hours
# ----------------------------------------------------------
new_data = pd.DataFrame({"study_hours": [4]})
predicted_score = model.predict(new_data)

print("\nPrediction:")
print("Predicted Score for 4 study hours:", round(predicted_score[0], 2))

# ----------------------------------------------------------
# Structured Console Output
# ----------------------------------------------------------

# ANSI Bold Code
BOLD = "\033[1m"
END = "\033[0m"

print("\n" + "="*65)
print(BOLD + "📊 DATASET PREVIEW".center(65) + END)
print("="*65)

print(df.head().to_string(index=False))

print("\n" + "="*65)
print(BOLD + "📈 MODEL PERFORMANCE SUMMARY".center(65) + END)
print("="*65)

print(f"{BOLD}{'R² Score:':<30}{END} {round(r2, 3)}")

print("\n" + "-"*65)
print(BOLD + "🔮 PREDICTION RESULT".center(65) + END)
print("-"*65)

print(f"{BOLD}{'Input Study Hours:':<30}{END} 4")
print(f"{BOLD}{'Predicted Grade:':<30}{END} {round(predicted_score[0], 2)}")

print("="*65)

# ----------------------------------------------------------
# Step 7: Visualization
# ----------------------------------------------------------

plt.figure(figsize=(10, 6))

# Scatter plot
plt.scatter(X, y, color="#1f77b4", alpha=0.6, label="Actual Data")

# Regression line
plt.plot(X, model.predict(X), color="#d62728", linewidth=3, label="Regression Line")

# Highlight prediction point
plt.scatter(4, predicted_score[0], color="green", s=120, label="Prediction (4 hrs)")

# Labels and styling
plt.xlabel("Study Hours", fontsize=12)
plt.ylabel("Grade", fontsize=12)
plt.title("Study Hours vs Grade Prediction", fontsize=14)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()