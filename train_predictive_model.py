import pandas as pd  # Importing the pandas library for data manipulation
import numpy as np  # Importing the numpy library for numerical operations
from sklearn.model_selection import train_test_split  # Importing train_test_split function for data splitting
from sklearn.ensemble import GradientBoostingRegressor  # Importing GradientBoostingRegressor for regression modeling
from sklearn.metrics import mean_squared_error  # Importing mean_squared_error for performance evaluation
from sklearn.preprocessing import LabelEncoder  # Importing LabelEncoder for encoding categorical features
from joblib import dump  # Importing dump function for model and encoder serialization
import os  # Importing os for file handling

# Load data
data_path = 'dummy_ride_data.csv'  # Modify with the actual path
data = pd.read_csv(data_path)  # Loading the dataset from the CSV file

# Initialize and fit label encoders
region_encoder = LabelEncoder()  # Initializing label encoder for region
time_encoder = LabelEncoder()  # Initializing label encoder for time
data['From Region'] = region_encoder.fit_transform(data['From Region'])  # Encoding 'From Region' feature
data['Destination Region'] = region_encoder.transform(data['Destination Region'])  # Encoding 'Destination Region' feature
data['Time of Day'] = time_encoder.fit_transform(data['Time of Day'])  # Encoding 'Time of Day' feature

# Split the data
X = data[['From Region', 'Destination Region', 'Time of Day']]  # Extracting features
y = data['Cost']  # Extracting target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Splitting data into train and test sets

# Train a model for each app
models = {}  # Dictionary to store trained models
apps = ['TADA', 'GoJek', 'Zig']  # List of ride-hailing apps
for app in apps:
    app_data = data[data['App'] == app]  # Filtering data for the specific app
    X_app = app_data[['From Region', 'Destination Region', 'Time of Day']]  # Extracting features for the app
    y_app = app_data['Cost']  # Extracting target variable for the app
    X_train_app, X_test_app, y_train_app, y_test_app = train_test_split(X_app, y_app, test_size=0.2, random_state=42)  # Splitting data for the app

    model = GradientBoostingRegressor(random_state=42)  # Initializing GradientBoostingRegressor model
    model.fit(X_train_app, y_train_app)  # Training the model
    models[app] = model  # Storing the trained model in the dictionary

    # Calculate RMSE to determine the range
    predictions = model.predict(X_test_app)  # Making predictions
    mse = mean_squared_error(y_test_app, predictions)  # Calculating mean squared error
    rmse = np.sqrt(mse)  # Calculating root mean squared error
    print(f'{app} RMSE: {rmse}')  # Printing the RMSE for the app

# Prediction function with input validation and normalization
def predict_prices(from_region, destination_region, time_of_day):
    valid_regions = ['Central', 'North', 'East', 'West', 'South']  # Valid regions
    valid_times = ['8:45 AM', '12:30 PM', '6:00 PM']  # Valid times

    from_region = from_region.capitalize()  # Capitalizing from_region
    destination_region = destination_region.capitalize()  # Capitalizing destination_region
    time_of_day = time_of_day.strip()  # Stripping whitespace from time_of_day

    if from_region not in valid_regions or destination_region not in valid_regions or time_of_day not in valid_times:
        return "Invalid input. Please check your region or time input and try again."  # Returning error message for invalid input

    from_region_encoded = region_encoder.transform([from_region])[0]  # Encoding from_region
    destination_region_encoded = region_encoder.transform([destination_region])[0]  # Encoding destination_region
    time_of_day_encoded = time_encoder.transform([time_of_day])[0]  # Encoding time_of_day

    X_new = pd.DataFrame([[from_region_encoded, destination_region_encoded, time_of_day_encoded]],
                         columns=['From Region', 'Destination Region', 'Time of Day'])  # Creating new DataFrame for prediction

    results = []  # List to store prediction results
    for app in apps:
        model = models[app]  # Getting the model for the app
        predicted_cost = model.predict(X_new)[0]  # Making prediction
        mse = mean_squared_error(y_test[y_test.index.isin(X_test[(X_test['From Region'] == from_region_encoded) &  # Calculating mean squared error
                                                                 (X_test['Destination Region'] == destination_region_encoded) &
                                                                 (X_test['Time of Day'] == time_of_day_encoded)].index)],
                                 [predicted_cost] * len(
                                     y_test[y_test.index.isin(X_test[(X_test['From Region'] == from_region_encoded) &
                                                                     (X_test['Destination Region'] == destination_region_encoded) &
                                                                     (X_test['Time of Day'] == time_of_day_encoded)].index)]))
        rmse = np.sqrt(mse)  # Calculating root mean squared error
        results.append((app, round(predicted_cost - rmse, 2), round(predicted_cost + rmse, 2)))  # Appending prediction result to the list

    return results  # Returning prediction results

# User input section
while True:
    from_region = input("Which region are you from? (e.g., 'Central', 'North', 'East', 'West', 'South'): ")  # Getting user input for from_region
    destination_region = input("Which region are you going to? (e.g., 'Central', 'North', 'East', 'West', 'South'): ")  # Getting user input for destination_region
    time_of_day = input("What time of day is it? (e.g., '8:45 AM', '12:30 PM', '6:00 PM'): ")  # Getting user input for time_of_day

    price_predictions = predict_prices(from_region, destination_region, time_of_day)  # Calling the prediction function
    if isinstance(price_predictions, str):  # Checking if prediction result is a string (error message)
        print(price_predictions)  # Printing the error message
        continue  # Continuing to the next iteration of the loop

    print("Bot: Ok i got some prices for you:")  # Printing message
    for app, low_price, high_price in price_predictions:  # Iterating over prediction results
        print(f"{app}: ${low_price} - ${high_price}")  # Printing prediction result
    break  # Exiting the loop

# Ensure the models directory exists
models_dir = 'models'  # Directory to store models
os.makedirs(models_dir, exist_ok=True)  # Creating the directory if it doesn't exist

# Save each model in the models directory
for app, model in models.items():
    model_path = os.path.join(models_dir, f'{app}_model.joblib')  # Constructing model path
    dump(model, model_path)  # Saving the model to disk

# Save label encoders
dump(region_encoder, os.path.join(models_dir, 'region_encoder.joblib'))  # Saving region encoder to disk
dump(time_encoder, os.path.join(models_dir, 'time_encoder.joblib'))  # Saving time encoder to disk

print("Models and encoders have been saved in the 'models' directory.")  # Printing message
