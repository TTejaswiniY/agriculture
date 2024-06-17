import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib
import os

# Provide the correct path to your CSV file
file_path = "C:/Users/HP/agri.csv"

# Print the current working directory
print("Current working directory:", os.getcwd())

# Check if the file exists and print the directory contents
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    print("Contents of the directory:")
    print(os.listdir(os.path.dirname(file_path)))
else:
    print(f"File found: {file_path}")

    # Load the data
df = pd.read_csv(file_path)

# Preprocessing
def preprocess_data(df):
    df = df.dropna()  # Drop rows with missing values

    # Extract month and year from the date
    Dict = {1: 'january', 2: 'february', 3: 'march', 4: 'april', 5: 'may', 6: 'june', 
            7: 'july', 8: 'august', 9: 'september', 10: 'october', 11: 'november', 12: 'december'}

    month_col = []
    for date in df["date"]:
        # Ensure the date format is correct
        try:
            parts = date.split('-')
            if len(parts) == 3:
                month = int(parts[1])
                month_col.append(Dict[month])
            else:
                raise ValueError
        except (IndexError, ValueError):
            print(f"Skipping invalid date format: {date}")
            month_col.append(None)

    df.loc[:, "month_col"] = month_col  # Use .loc to assign values

    # Remove rows with invalid dates
    df = df.dropna(subset=["month_col"])

    # Assign season names
    def get_season(month):
        if month in ["january", "february", "march"]:
            return "Winter"
        elif month in ["april", "may", "june"]:
            return "Summer"
        elif month in ["july", "august", "september"]:
            return "Monsoon"
        else:
            return "Autumn"

    df["season_names"] = df["month_col"].apply(get_season)

    # Encode categorical variables
    for col in ['state', 'district', 'commodity_name', 'market', 'month_col', 'season_names']:
        df[col] = df[col].astype('category').cat.codes

    # Extract day and year from the date
    df['day'] = pd.to_datetime(df['date'], errors='coerce').dt.day
   

    # Remove rows with invalid dates after conversion
    df = df.dropna(subset=['day'])

    return df

    df = preprocess_data(df)


    # Define features and labels
    features = df[['commodity_name', 'state', 'district', 'market', 'month_col', 'season_names', 'day']]
    labels = df['modal_price']

    # Split the data into training and testing sets
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(features, labels, test_size=0.2, random_state=2)

    # Train the RandomForestRegressor model
    regr = RandomForestRegressor(max_depth=20, n_estimators=100, random_state=0)
    regr.fit(Xtrain, Ytrain)

    # Save the model
    joblib.dump(regr, 'random_forest_regressor_model.pkl')

    # Load the model (for testing)
    loaded_model = joblib.load('random_forest_regressor_model.pkl')

    # Make predictions
    y_pred = loaded_model.predict(Xtest)

    # Evaluate the model
    print("R2 Score:", r2_score(Ytest, y_pred))

    # Example prediction
    user_input = [[21, 6, 249, 1041, 3, 1, 4]]
    print("Prediction for user input:", loaded_model.predict(user_input))