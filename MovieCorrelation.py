import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.style.use("ggplot")  # Type of style to plot graphs

# -----------------------------------------------------------------------------------
# A figure in matplotlib represents the window or page where your plots are drawn.
plt.figure(figsize=(14, 14))

print("\n-----------------------------------------------------------------------------")
# Reading the Data
data = pd.read_csv("movies.csv")  # print(data.head(10).to_string())
data.drop_duplicates(inplace=True)  # Remove duplicates in-place

print("Seeing if there is any missing data present in our dataset:\n")
for col in data.columns:
    pct_missing = np.mean(data[col].isnull())
    # returns a Boolean Series indicating which values in the column are null (missing).
    print("{0:<10} : {1:.7f}% missing".format(col, pct_missing * 100))

print("\n-----------------------------------------------------------------------------")

data = data.dropna()  # Drop all rows that have missing data.
print("NOTE: NOW THE DATASET IS NOT HAVING ANY MISSING DATA ... ...")

print("\n-----------------------------------------------------------------------------")


def change_data_type(column_name):
    # Creating a function to change the data types into more readable manner
    print("Changing data type of '{0}' from '{1}' to 'int64'".format(column_name, data[column_name].dtype))
    data[column_name] = pd.to_numeric(data[column_name]).astype("int64")


# The year column and the year present in released column is not same, so we are doing this ... ..
data['year_correct'] = data['released'].str.extract(pat='([0-9]{4})').astype("int64")

change_data_type("gross")
change_data_type("budget")
change_data_type("runtime")
change_data_type("votes")

print("\n-----------------------------------------------------------------------------")

columns_of_interest = ['budget', 'runtime', 'gross', 'score', 'votes']
# pearson is by default (Another two: Kendall, Spearman)
print("Pearson:")
correlation_matrix_id = data[columns_of_interest].corr(method="pearson")
print(correlation_matrix_id)

print("\n-----------------------------------------------------------------------------")

# MAIN PROJECT STARTS FROM HERE:
# Scatter plot with budget vs Gross Revenue.
# We are trying to find out if there is any correlation present between them.

plt.scatter(x=data["budget"], y=data["gross"])
plt.xlabel("Budget")
plt.ylabel("Gross")
plt.title("Budget vs Gross")


# Define the formatter function
def millions_formatter(a, pos):
    return '{:.0f}M'.format(a / 1000000)


# Set the formatter for both axes
formatter = ticker.FuncFormatter(millions_formatter)
plt.gca().xaxis.set_major_formatter(formatter)
plt.gca().yaxis.set_major_formatter(formatter)

# Plot Budget vd Gross using seaborn
sns.regplot(data, x='budget', y='gross', scatter_kws={'color': 'red'}, line_kws={'color': 'green'})

plt.show()  # Display the plot

# ---------------------------------------------------------------------------------

sns.heatmap(correlation_matrix_id, annot=True)
plt.title("Correlation Matrix")
plt.show()

# ---------------------------------------------------------------------------------

print("Now, we are giving unique IDs to all the non-number columns:")

data_id = data

for column in data.columns:
    if data_id[column].dtype == "object":
        data_id[column] = data_id[column].astype("category")
        data_id[column] = data_id[column].cat.codes  # Random IDs are generated

print("\n-----------------------------------------------------------------------------")

# Votes and Budget have the highest correlation to gross earnings
print("Pearson: (High Correlation between all constraints)")
correlation_matrix_id = data_id.corr(method="pearson")
corr_pairs = correlation_matrix_id.unstack()
sorted_corr_pairs = corr_pairs.sort_values()

high_corr = sorted_corr_pairs[sorted_corr_pairs > 0.5]
print(high_corr)

sns.heatmap(correlation_matrix_id, annot=True)
plt.title("Correlation Matrix")
plt.show()

print("\n-----------------------------------------------------------------------------")
