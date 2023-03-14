
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

"""
    Loads data from a csv file and returns a pandas DataFrame.
    
    Args:
    file_path (str): Path to the csv file.
    
    Returns:
    pd.DataFrame: A pandas DataFrame containing the loaded data.
    Plots a histogram of a pandas Series and saves the plot to a file.
    Adds a vertical line indicating the mean value.
    
    Args:
    x (pd.Series): A pandas Series containing the data to plot.
    mean (float): The mean value of the data.
    file_path (str): Path to save the plot file to.
    
    Returns:
    None
    Plots histograms of all numeric columns in a pandas DataFrame and saves the plot to a file.
    Adds a vertical line indicating the mean
    """


# Loading the data
raw_data = pd.read_csv('C:/Users/olomuc/OneDrive - MHRA/COVID_ML_MODELS/covid_fat_recov.csv')

x = raw_data['Recovered'].values

sns.displot(x, color='blue', kde=True)

# Calculating the mean
mean = raw_data['Recovered'].mean()

# Plotting the mean
plt.axvline(mean, 0,1, color='red')

# Saving the plot to a PNG file with larger size
fig, ax = plt.subplots(figsize=(10, 8))
sns.displot(x, color='blue', kde=True)
plt.axvline(mean, 0, 1, color='red')
plt.savefig('Recovered_dist_plot_matrix.png', dpi=300)

# Printing a message indicating that the plot was saved
print('Barplot matrix saved to Recovered_dist_plot.png')

# Identifying all numeric columns
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
n_variables = raw_data.select_dtypes(include=numerics).columns

# Create a figure with subplots for each column
fig, axs = plt.subplots(nrows=len(n_variables), ncols=1, figsize=(10, 8*len(n_variables)))

# Loop over each column and create a histogram subplot
for i, n in enumerate(n_variables):
    sns.histplot(data=raw_data, x=n, kde=True, color='blue', ax=axs[i])
    axs[i].axvline(raw_data[n].mean(), 0, 1, color='red')
    axs[i].set_title(n)

# Save the plot to a PNG file with larger size
plt.savefig('all_histograms.png', dpi=300)

# Print a message indicating that the plot was saved
print('All histograms saved to all_histograms.png')
