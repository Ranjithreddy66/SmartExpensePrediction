import pandas as pd
from sklearn.model_selection import train_test_split

# Load Dataset
df = pd.read_csv("data/expenses.csv")

# Missing Values
print("Missing Values")
print(df.isnull().sum())

# Duplicate Rows
duplicates = df.duplicated().sum()
print("\nDuplicate Rows:", duplicates)

# Features and Target
X = df.drop("NextMonthExpense", axis=1)
y = df["NextMonthExpense"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Features:", X_train.shape)
print("Testing Features :", X_test.shape)

print("\nTraining Target :", y_train.shape)
print("Testing Target  :", y_test.shape)