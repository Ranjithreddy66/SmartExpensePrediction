import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/best_model.pkl")

print("=" * 50)
print(" SMART EXPENSE PREDICTION USING MACHINE LEARNING ")
print("=" * 50)

# User Input
user_id = int(input("Enter User ID: "))
month = int(input("Enter Month: "))
income = float(input("Enter Monthly Income: "))
food = float(input("Enter Food Expense: "))
travel = float(input("Enter Travel Expense: "))
shopping = float(input("Enter Shopping Expense: "))
bills = float(input("Enter Bills Expense: "))
entertainment = float(input("Enter Entertainment Expense: "))
healthcare = float(input("Enter Healthcare Expense: "))
education = float(input("Enter Education Expense: "))
others = float(input("Enter Others Expense: "))
previous_expense = float(input("Enter Previous Month Expense: "))
total_expense = float(input("Enter Total Expense: "))

# Create DataFrame
new_data = pd.DataFrame({
    "UserID": [user_id],
    "Month": [month],
    "Income": [income],
    "Food": [food],
    "Travel": [travel],
    "Shopping": [shopping],
    "Bills": [bills],
    "Entertainment": [entertainment],
    "Healthcare": [healthcare],
    "Education": [education],
    "Others": [others],
    "PreviousExpense": [previous_expense],
    "TotalExpense": [total_expense]
})

# Predict
prediction = model.predict(new_data)

print("\n" + "=" * 50)
print("Prediction Result")
print("=" * 50)
print(f"Predicted Next Month Expense: ₹ {prediction[0]:.2f}")