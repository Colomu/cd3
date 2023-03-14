import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Loading the data
raw_data = pd.read_csv('C:/Users/olomuc/OneDrive - MHRA/COVID_ML_MODELS/covid_fat_recov.csv')

# Creating a box plot of the "Deaths" column
fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(data=raw_data, x='Recovered', color='blue')
plt.savefig('Recovery_box_plot.png', dpi=300)

# Printing a message indicating that the plot was saved
print('Box plot saved to Recovery_box_plot.png')


# Loading the data
raw_data = pd.read_csv('C:/Users/olomuc/OneDrive - MHRA/COVID_ML_MODELS/covid_fat_recov.csv')

# Identifying all numeric columns
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
n_variables = raw_data.select_dtypes(include=numerics).columns

# Create a figure with subplots for each column
fig, axs = plt.subplots(nrows=len(n_variables)+1, ncols=1, figsize=(10, 8*(len(n_variables)+1)))

# Loop over each column and create a boxplot subplot
for i, n in enumerate(n_variables):
    sns.boxplot(data=raw_data, x=n, color='blue', ax=axs[i])
    axs[i].set_title(n)

# Creating a box plot of the "Deaths" column
sns.boxplot(data=raw_data, x='Recovered', color='blue', ax=axs[len(n_variables)])
axs[len(n_variables)].set_title('Recovered')

# Save the plot to a PNG file with larger size
plt.savefig('all_box_plot.png', dpi=300)

# Print a message indicating that the plot was saved
print('All box plots saved to all_box_plot.png')

