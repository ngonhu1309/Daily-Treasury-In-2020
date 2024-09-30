#@Authors
# Name: Nora Ngo

import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Create a text file in 2020
def convert_row(row):
    # Keep the date as a string
    date = row[0]
    # Convert all other values to floats and store them in a list
    float_values = [float(value) for value in row[1:]]
    return [date] + float_values


# for 2020 data
data_2 = pd.read_csv("daily-treasury-rates-2020.csv")
daily_yield_curves_2020 = []
# Convert the DataFrame to a list of lists
daily_yield_curves_2020 = data_2.values.tolist()
# If you want to include the header (column names) as the first row
header_2020 = data_2.columns.tolist()
daily_yield_curves_2020.insert(0, header_2020)
# Define a function to convert date string to datetime object
def date_converter_2020(date_string_2020):
    return datetime.strptime(date_string_2020, '%m/%d/%Y')
# Sort the rows based on the date (assuming date is in the first column)
sorted_rows_2020 = sorted(daily_yield_curves_2020[1:], key=lambda x: date_converter_2020(x[0]))

def convert_row_2020(row):
    # Keep the date as a string
    date_2020 = row[0]
    print("Date  " ,date_2020)
    # Convert all other values to floats and store them in a list
    float_values_2020 = [float(value) for value in row[1:]]
    return [date_2020] + float_values_2020

float_sort_list_2020 =[]
for row in sorted_rows_2020:
    converted_row_2020 = convert_row_2020(row)
    float_sort_list_2020.append(converted_row_2020)


daily_yield_curves_2020 = daily_yield_curves_2020[0]
daily_yield_curves_2020 = daily_yield_curves_2020 + float_sort_list_2020

# Create a new file named 'daily_yield_curves_2020.txt'
file_path = "daily_yield_curves_2020.txt"

# 3D Surface Plot 
# Open the file in write mode ('w')
with open(file_path, 'w') as file:
    # Write header (first 13 items)
        file.write('      ')
        for header_item in daily_yield_curves_2020[:13]:
            line = ' \t  '.join(str(x) for x in header_item)
            file.write(header_item + ' \t ')
        file.write('\n')
    # Write data rows (remaining items)
        for row in daily_yield_curves_2020[13:]:
            line = '\t '.join(str(x) for x in row)
            file.write(line + '\n')

# Map the yield curve columns to the appropriate number of months to maturity
maturity_months = [1, 2, 3, 6, 12, 24, 36, 60, 84, 120, 240, 360]

# Extract the interest rate data (ignoring the header and date columns)
rate_data_2020 = [row[1:] for row in float_sort_list_2020[1:]]  # Only keep the rates, skip the dates
rate_data_2020 = np.array(rate_data_2020)  # Convert to a NumPy array

# Calculate the number of days since the starting date
start_date = datetime.strptime('01/02/2020', "%m/%d/%Y")  # Adjusted to match 4-digit year format

# Correct the date format in the list comprehension
dates_2020 = [datetime.strptime(row[0], "%m/%d/%Y") for row in float_sort_list_2020[1:]]  # Adjusted to match 4-digit year format

# Calculate the number of days since the start date
days_since_start_2020 = np.array([(date - start_date).days for date in dates_2020])

# Create meshgrid for X (days) and Y (months to maturity)
X_2020, Y_2020 = np.meshgrid(days_since_start_2020, maturity_months)

# We need to transpose rate_data to match X, Y dimensions
Z_2020 = rate_data_2020.T

# Plotting the 3D Surface Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(1, 1, 1, projection='3d')
surf = ax.plot_surface(X_2020, Y_2020, Z_2020, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Add a color bar that matches the surface plot
cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)  # shrink and aspect control size/shape of the colorbar
cbar.set_label('Rate')  # Add label to the color bar

# Set axis labels
ax.set_xlabel('Trading days since 01/02/20')
ax.set_ylabel('Months to maturity')
ax.set_zlabel('Rate')

# Set custom ticks for X-axis with a difference of 25
ax.set_xticks(np.arange(0, X_2020.max() + 1, 25))

# Show the surface plot
plt.show()

# Wireframe Plot
# Plotting the Wireframe Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(1, 1, 1, projection='3d')
wire = ax.plot_wireframe(X_2020, Y_2020, Z_2020, color='green')

# Set axis labels
ax.set_xlabel('Trading days since 01/02/20')
ax.set_ylabel('Months to maturity')
ax.set_zlabel('Rate')

# Set custom ticks for X-axis with a difference of 25
ax.set_xticks(np.arange(0, X_2020.max() + 1, 25))

# Show the wireframe plot
plt.show()

# Time series of interest rates for each maturity
# Create the DataFrame from daily_yield_curves
columns = ['1 mo', '2 mo', '3 mo', '6 mo', '1 yr', '2 yr', '3 yr', '5 yr', '7 yr', '10 yr', '20 yr', '30 yr']  # Bond maturities
rows = [row[0] for row in float_sort_list_2020[0:]]  # Dates as index
data = [row[1:] for row in float_sort_list_2020[0:]]  # Interest rate data

# Create the yield_curve_df DataFrame
yield_curve_df = pd.DataFrame(data, index=rows, columns=columns)

# Plot the time series of interest rates for each maturity
yield_curve_df.plot(figsize=(10, 6))
plt.title('Interest Rate Time Series, 2020')
plt.xlabel('Trading Date')
plt.ylabel('Interest Rate (%)')

# Set the limits to match the exact range of the data
plt.xlim(left=0, right=len(rows) - 1)  # Match x-axis limit to data range

# Set the y-limits based on the range of interest rates
plt.ylim(yield_curve_df.min().min(), yield_curve_df.max().max())

# Remove x-axis margins so lines extend to the edges
plt.margins(x=0)

# Select 10 evenly spaced tick positions along the x-axis
tick_positions = list(range(0, len(rows), len(rows) // 9))

# Set the corresponding dates for the tick positions
tick_labels = [rows[i] for i in tick_positions]

# Set the custom ticks on the x-axis
plt.xticks(ticks=tick_positions, labels=tick_labels, rotation=45)

# Ensure the plot fits tightly within the figure boundaries
plt.tight_layout()

plt.xticks(rotation=45)
plt.grid(True)

# Show the plot
plt.show()

# Transpose the DataFrame
transposed_df = yield_curve_df.T

# Select every 20th trading day
by_day_yield_curve_df = transposed_df.iloc[:, ::20]

# Update the column labels to include only the corresponding trading dates
by_day_yield_curve_df.columns = [rows[i] for i in range(0, len(rows), 20)]

by_day_yield_curve_df.index = maturity_months

# Plot the yield curves for every 20th trading day
by_day_yield_curve_df.plot(figsize=(10, 6))
plt.title('2020 Yield Curves, 20 Day Intervals')
plt.xlabel('Months to Maturity')
plt.ylabel('Interest Rate (%)')
plt.legend(loc='lower right')
plt.grid(True)

# Set the limits to match the exact range of the data
plt.xlim(left=0)  # Match x-axis limit to data range

# Show the plot
plt.show()
