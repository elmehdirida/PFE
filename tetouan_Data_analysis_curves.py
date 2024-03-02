#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Assuming your CSV file is named 'your_data.csv'
data = pd.read_csv('power-consumption.csv')
print(data.columns)
# Display basic descriptive statistics for each column
statistics = data.describe()

# Display the result
print(statistics)


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV data into a pandas DataFrame, specifying 'DateTime' as a datetime column
data = pd.read_csv('S:\Mes documents\Bureau\PFE MID\Other Articls\Tetouan_Recherche\DataSet\Tetuan City power consumption.csv', parse_dates=['DateTime'])

# Time Series Plot

plt.figure(figsize=(25, 6))
plt.plot(data['DateTime'], data['Zone 1 Power Consumption'], label='Power Consumption Zone 1')
plt.xlabel('DateTime')
plt.ylabel('Values')
plt.title('Zone 1 Power Consumption over Time')
plt.legend()
plt.show()

# Time Series Plot for Zone 2 Power Consumption
plt.figure(figsize=(12, 6))
plt.plot(data['DateTime'], data['Zone 2  Power Consumption'], label='Power Consumption Zone 2', color='orange')
plt.xlabel('DateTime')
plt.ylabel('Power Consumption')
plt.title('Zone 2 Power Consumption over Time')
plt.legend()
plt.show()

# Time Series Plot for Zone 3 Power Consumption
plt.figure(figsize=(12, 6))
plt.plot(data['DateTime'], data['Zone 3  Power Consumption'], label='Power Consumption Zone 3', color='green')
plt.xlabel('DateTime')
plt.ylabel('Power Consumption')
plt.title('Zone 3 Power Consumption over Time')
plt.legend()
plt.show()
# Correlation Heatmap
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV data into a pandas DataFrame, specifying 'DateTime' as a datetime column
data = pd.read_csv('power consumption.csv', parse_dates=['DateTime'])

# Comparative Time Series Plot for Zone 1, Zone 2, and Zone 3 Power Consumption
plt.figure(figsize=(12, 6))
plt.plot(data['DateTime'], data['Zone 1 Power Consumption'], label='Power Consumption Zone 1', color='blue')
plt.plot(data['DateTime'], data['Zone 2  Power Consumption'], label='Power Consumption Zone 2', color='orange')
plt.plot(data['DateTime'], data['Zone 3  Power Consumption'], label='Power Consumption Zone 3', color='green')
plt.xlabel('DateTime')
plt.ylabel('Power Consumption')
plt.title('Comparative Power Consumption for Zone 1, Zone 2, and Zone 3 over Time')
plt.legend()
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV data into a pandas DataFrame, specifying 'DateTime' as a datetime column
data = pd.read_csv('power consumption.csv', parse_dates=['DateTime'])

# Assuming 'DateTime' is in the index for better seasonal analysis
data.set_index('DateTime', inplace=True)

# Resample the data to monthly frequency and calculate the mean
monthly_mean = data.resample('M').mean()

# Seasonal Plot for Zone 1, Zone 2, and Zone 3 Power Consumption
plt.figure(figsize=(12, 6))
plt.plot(monthly_mean.index, monthly_mean['Zone 1 Power Consumption'], label='Zone 1', color='blue')
plt.plot(monthly_mean.index, monthly_mean['Zone 2  Power Consumption'], label='Zone 2', color='orange')
plt.plot(monthly_mean.index, monthly_mean['Zone 3  Power Consumption'], label='Zone 3', color='green')
plt.xlabel('Month')
plt.ylabel('Average Power Consumption')
plt.title('Seasonal Patterns: Average Power Consumption for Zone 1, Zone 2, and Zone 3')
plt.legend()
plt.show()


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt

# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(data['Temperature'], data['Zone 1 Power Consumption'], alpha=0.5)
plt.title('Zone 1 Power Consumption vs Temperature')
plt.xlabel('Temperature (°C)')
plt.ylabel('Zone 1 Power Consumption (kWh)')
plt.show()


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt

# Attempt to convert 'Datetime' to datetime format, specifying the format
data['DateTime'] = pd.to_datetime(data['DateTime'], format='%m/%d/%Y %H:%M', errors='coerce')

# Extract month from the datetime
data['Month'] = data['DateTime'].dt.month

# Assuming seasons: Winter (12, 1, 2), Spring (3, 4, 5), Summer (6, 7, 8), Fall (9, 10, 11)
seasons = {'Winter': [12, 1, 2], 'Spring': [3, 4, 5], 'Summer': [6, 7, 8], 'Fall': [9, 10, 11]}

# Create a new column 'Season' based on the month
data['Season'] = data['Month'].apply(lambda month: next(season for season, months in seasons.items() if month in months))

# Plot for each season
for season, season_data in data.groupby('Season'):
    plt.figure(figsize=(10, 6))
    plt.scatter(season_data['Temperature'], season_data['Zone 1 Power Consumption'], alpha=0.5, label=season)
    plt.title(f'Zone 1 Power Consumption vs Temperature - {season}')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Zone 1 Power Consumption (kWh)')
    plt.legend()
    plt.show()


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Convert 'Datetime' to datetime format
data['DateTime'] = pd.to_datetime(data['DateTime'], format='%m/%d/%Y %H:%M', errors='coerce')

# Extract week from the datetime
data['Week'] = data['DateTime'].dt.isocalendar().week

# Create a box plot for electricity consumption among weeks
plt.figure(figsize=(12, 8))
sns.boxplot(x='Week', y='Zone 1 Power Consumption', data=data)
plt.title('Box Plot Comparison of Zone 1 Power Consumption Among Weeks')
plt.xlabel('Week')
plt.ylabel('Zone 1 Power Consumption (kWh)')
plt.show()


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Convert 'Datetime' to datetime format
data['DateTime'] = pd.to_datetime(data['DateTime'], format='%m/%d/%Y %H:%M', errors='coerce')

# Extract day of the week from the datetime
data['DayOfWeek'] = data['DateTime'].dt.day_name()

# Create a box plot for electricity consumption among days of the week
plt.figure(figsize=(12, 8))
sns.boxplot(x='DayOfWeek', y='Zone 1 Power Consumption', data=data, order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.title('Box Plot Comparison of Zone 1 Power Consumption Among Days of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Zone 1 Power Consumption (kWh)')
plt.show()


# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Convert 'DateTime' to datetime format
data['DateTime'] = pd.to_datetime(data['DateTime'], format='%m/%d/%Y %H:%M', errors='coerce')

# Extract relevant calendar variables
data['DayOfWeek'] = data['DateTime'].dt.day_name()
data['Month'] = data['DateTime'].dt.month
data['Hour'] = data['DateTime'].dt.hour

# Extract relevant columns for correlation analysis
correlation_data = data[['Zone 1 Power Consumption', 'DayOfWeek', 'Month', 'Hour']]

# Encode categorical variables using one-hot encoding
correlation_data = pd.get_dummies(correlation_data, columns=['DayOfWeek', 'Month', 'Hour'])

# Calculate the correlation matrix
correlation_matrix = correlation_data.corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(35, 35))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap: Zone 1 Power Consumption vs. Calendar Variables')
plt.show()


# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Convert 'DateTime' to datetime format
data['DateTime'] = pd.to_datetime(data['DateTime'], format='%m/%d/%Y %H:%M', errors='coerce')

# Extract relevant calendar variables
data['DayOfWeek'] = data['DateTime'].dt.day_name()
data['Month'] = data['DateTime'].dt.month
data['Hour'] = data['DateTime'].dt.hour

# Extract relevant columns for correlation analysis
correlation_data = data[['Zone 2  Power Consumption', 'DayOfWeek', 'Month', 'Hour']]

# Encode categorical variables using one-hot encoding
correlation_data = pd.get_dummies(correlation_data, columns=['DayOfWeek', 'Month', 'Hour'])

# Calculate the correlation matrix
correlation_matrix = correlation_data.corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(35, 35))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap: Zone 2 Power Consumption vs. Calendar Variables')
plt.show()


# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Convert 'DateTime' to datetime format
data['DateTime'] = pd.to_datetime(data['DateTime'], format='%m/%d/%Y %H:%M', errors='coerce')

# Extract relevant calendar variables
data['DayOfWeek'] = data['DateTime'].dt.day_name()
data['Month'] = data['DateTime'].dt.month
data['Hour'] = data['DateTime'].dt.hour

# Extract relevant columns for correlation analysis
correlation_data = data[['Zone 3  Power Consumption', 'DayOfWeek', 'Month', 'Hour']]

# Encode categorical variables using one-hot encoding
correlation_data = pd.get_dummies(correlation_data, columns=['DayOfWeek', 'Month', 'Hour'])

# Calculate the correlation matrix
correlation_matrix = correlation_data.corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(35, 35))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap: Zone 2 Power Consumption vs. Calendar Variables')
plt.show()


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt


# Convert 'DateTime' to datetime format
data['DateTime'] = pd.to_datetime(data['DateTime'], format='%m/%d/%Y %H:%M', errors='coerce')

# Set 'DateTime' as the index for time-based plotting
data.set_index('DateTime', inplace=True)

# Create subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 8), sharex=True)

# Plot Wind Speed
ax1.plot(data.index, data['Wind Speed'], label='Wind Speed', color='blue')
ax1.set_ylabel('Wind Speed')

# Plot Temperature
ax2.plot(data.index, data['Temperature'], label='Temperature', color='orange')
ax2.set_ylabel('Temperature (°C)')

# Plot Humidity
ax3.plot(data.index, data['Humidity'], label='Humidity', color='green')
ax3.set_ylabel('Humidity')

# Set common x-axis label
plt.xlabel('Time')

# Adjust layout for better visualization
plt.tight_layout()

# Show the plot
plt.show()

