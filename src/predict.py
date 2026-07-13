import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/best_model.pkl")

# Sample input
new_data = pd.DataFrame({

    "UserID":[101],
    "Month":[25],
    "Income":[60000],
    "Food":[7000],
    "Travel":[2500],
    "Shopping":[3000],
    "Bills":[8500],
    "Entertainment":[1800],
    "Healthcare":[1200],
    "Education":[2500],
    "Others":[1000],
    "PreviousExpense":[29000]

})

# Prediction
prediction = model.predict(new_data)

print("="*40)
print("Predicted Next Month Expense")
print("="*40)

print(f"₹ {prediction[0]:.2f}")