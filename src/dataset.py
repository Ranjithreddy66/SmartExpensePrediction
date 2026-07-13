import pandas as pd
import random

random.seed(42)

records = []

num_users = 100

months = 24

for user_id in range(1, num_users + 1):

    income = random.randint(30000, 100000)

    previous_total = random.randint(18000, 30000)

    for month in range(1, months + 1):

        food = int(income * random.uniform(0.10, 0.18))
        travel = random.randint(1000, 5000)
        shopping = random.randint(1000, 7000)
        bills = random.randint(5000, 12000)
        entertainment = random.randint(1000, 4000)
        healthcare = random.randint(500, 3000)
        education = random.randint(500, 4000)
        others = random.randint(500, 3000)

        total = (
            food +
            travel +
            shopping +
            bills +
            entertainment +
            healthcare +
            education +
            others
        )

        records.append([
            user_id,
            month,
            income,
            food,
            travel,
            shopping,
            bills,
            entertainment,
            healthcare,
            education,
            others,
            previous_total,
            total
        ])

        previous_total = total

df = pd.DataFrame(records, columns=[
    "UserID",
    "Month",
    "Income",
    "Food",
    "Travel",
    "Shopping",
    "Bills",
    "Entertainment",
    "Healthcare",
    "Education",
    "Others",
    "PreviousExpense",
    "TotalExpense"
])

df.to_csv("data/expenses.csv", index=False)

print(df.head())
print()
print(df.shape)

import os

print("\nDataset Created Successfully!")
print("CSV saved at: ")
print(os.path.abspath("data/expenses.csv"))