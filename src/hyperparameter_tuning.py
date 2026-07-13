import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Load dataset
df = pd.read_csv("data/expenses.csv")

# Features (X)
X = df.drop("NextMonthExpense", axis=1)

# Target (y)
y = df["NextMonthExpense"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

rf = RandomForestRegressor(random_state=42)

param_grid = {

    "n_estimators": [50, 100],

    "max_depth": [5, 10],

    "min_samples_split": [2, 5],

    "min_samples_leaf": [1, 2]

}

grid = GridSearchCV(

    estimator=rf,

    param_grid=param_grid,

    cv=5,

    scoring="r2",

    n_jobs=-1

)

grid.fit(X_train, y_train)

print("Best Parameters")

print(grid.best_params_)

best_model = grid.best_estimator_
predictions = best_model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(
    y_test,
    predictions
)

print("\nModel Evaluation")

print("MAE :", mae)

print("RMSE:", rmse)

print("R² :", r2)
joblib.dump(
    best_model,
    "models/best_model.pkl"
)

print("\nBest model saved successfully!")