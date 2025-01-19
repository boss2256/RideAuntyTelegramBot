import pandas as pd  # Importing the pandas library for data manipulation
import numpy as np  # Importing the numpy library for numerical operations
import matplotlib.pyplot as plt  # Importing the matplotlib library for data visualization

# Constants for the dataset
n_rows = 1000  # Number of rows for the dataset
regions = ["Central", "North", "East", "West", "South"]  # List of regions in Singapore
apps = ["TADA", "GoJek", "Zig"]  # List of ride-hailing apps
times_of_day = ["8:45 AM", "12:30 PM", "6:00 PM"]  # List of times of the day

# Base cost and price increases
base_cost = 15.6  # Base price for TADA
price_increase = {
    'TADA': 0,  # Price increase factor for TADA
    'GoJek': 0.0392,  # Price increase factor for GoJek
    'Zig': 0.129 + 0.0392  # Price increase factor for Zig
}
std_devs = {
    'TADA': 7.68,  # Standard deviation for TADA
    'GoJek': 7.88,  # Standard deviation for GoJek
    'Zig': 9.19  # Standard deviation for Zig
}

# Generate data
np.random.seed(42)  # Setting the random seed for reproducibility
data = {
    "From Region": np.random.choice(regions, n_rows),  # Generating random 'From Region' values
    "Destination Region": np.random.choice(regions, n_rows),  # Generating random 'Destination Region' values
    "App": np.random.choice(apps, n_rows),  # Generating random 'App' values
    "Time of Day": np.random.choice(times_of_day, n_rows),  # Generating random 'Time of Day' values
    "Cost": [],  # Placeholder for ride cost
    "Surge Indicator": [],  # Placeholder for surge indicator
    "Fare Type": np.random.choice(['Regular', 'Surge', 'Low Demand'], n_rows, p=[0.5, 0.3, 0.2])  # Generating random 'Fare Type' values
}

# Populate Cost and Surge Indicator
for app in data['App']:
    mean_cost = base_cost + (base_cost * price_increase[app])  # Calculating mean cost based on app
    cost = max(np.random.normal(mean_cost, std_devs[app]), 7.88)  # Generating cost with normal distribution, ensuring it's at least $7.88
    data['Cost'].append(cost)

    # Adjusting Surge Indicator logic based on app
    if app == 'GoJek':
        data['Surge Indicator'].append('Yes' if (np.random.rand() < 0.1652 and cost > 23) else 'No')
    elif app == 'Zig':
        data['Surge Indicator'].append('Yes' if (np.random.rand() < 0.6019 and cost > 23) else 'No')
    else:
        data['Surge Indicator'].append('Yes' if cost > 23 else 'No')

# Create DataFrame
df = pd.DataFrame(data)  # Creating a DataFrame using the generated data

# Function to plot the balance
def plot_distributions(dataframe):
    features = ['From Region', 'Destination Region', 'App', 'Time of Day', 'Surge Indicator', 'Fare Type']  # List of features to plot
    plt.figure(figsize=(15, 10))  # Setting figure size for plotting
    for i, feature in enumerate(features):
        plt.subplot(2, 3, i+1)  # Creating subplots
        dataframe[feature].value_counts(normalize=True).plot(kind='bar', title=f'Distribution of {feature}')  # Plotting frequency distribution of each feature
        plt.ylabel('Frequency')  # Adding y-label
    plt.tight_layout()  # Adjusting layout
    plt.show()  # Displaying the plot

# Call plotting function
plot_distributions(df)  # Calling the plotting function with the DataFrame

# Save the DataFrame to a CSV file
df.to_csv('dummy_ride_data.csv', index=False)  # Saving the DataFrame to a CSV file without index

# Output a sample of the dataframe
df.head()  # Displaying the first few rows of the DataFrame
