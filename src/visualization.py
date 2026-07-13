import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/expenses.csv")
print(df.columns)
print(df.head())
print("\nColumns:")
print(df.columns)

print("\nShape:")
print(df.shape)
monthly = df.groupby("Month")["NextMonthExpense"].mean()

print(monthly)



plt.figure(figsize=(10,5))

plt.plot(monthly.index,
         monthly.values,
         marker='o')

plt.title("Average Expense by Month")

plt.xlabel("Month")

plt.ylabel("Average Expense")

plt.grid(True)

plt.show()

df.groupby("Month")

expense_columns = [
    "Food",
    "Travel",
    "Shopping",
    "Bills",
    "Entertainment",
    "Healthcare",
    "Education",
    "Others"
]

totals = df[expense_columns].sum()

plt.figure(figsize=(8,8))

plt.pie(
    totals,
    labels=expense_columns,
    autopct="%1.1f%%"
)

plt.title("Expense Distribution")

plt.show()

plt.figure(figsize=(8,5))

plt.scatter(
    df["Income"],
    df["NextMonthExpense"]
)

plt.xlabel("Income")

plt.ylabel("Next Month Expense")

plt.title("Income vs Expense")

plt.show()

plt.figure(figsize=(8,5))

plt.hist(
    df["NextMonthExpense"],
    bins=20
)

plt.title("Expense Distribution")

plt.xlabel("Expense")

plt.ylabel("Frequency")

plt.show()