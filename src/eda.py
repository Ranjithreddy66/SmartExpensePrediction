import pandas as pd

# Load dataset
df = pd.read_csv("data/expenses.csv")

print("=" * 50)
print("FIRST 5 ROWS")
print("=" * 50)
print(df.head())

print("\n" + "=" * 50)
print("LAST 5 ROWS")
print("=" * 50)
print(df.tail())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nStatistical Summary:")
print(df.describe())

print("\nRandom Sample:")
print(df.sample(5))

print(df["NextMonthExpense"].head(20))
print(df["NextMonthExpense"].isnull().sum())