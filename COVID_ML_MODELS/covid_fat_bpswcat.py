import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


"""
This code loads a dataset of COVID-19 data from a CSV file, creates a categorical plot with both boxplot and swarmplot, and saves the resulting plot to a PNG file.

The steps in the code are:

Load the COVID-19 data from a CSV file.
Create a categorical plot with both boxplot and swarmplot using the Seaborn library.
Set the title of the plot.
Save the plot to a PNG file with larger size using the Matplotlib library.
Print a message indicating that the plot was saved.
The purpose of this code is to visualize the distribution of COVID-19 deaths across different continents using a combination of a boxplot and swarmplot.
"""

# Loading the data
raw_data = pd.read_csv('C:/Users/olomuc/OneDrive - MHRA/COVID_ML_MODELS/covid_fat_recov.csv')

# Create a categorical plot with both boxplot and swarmplot
fig = sns.catplot(x='Continent', y='Deaths', data=raw_data, kind='boxen', height=6, aspect=1.5)
sns.swarmplot(x='Continent', y='Deaths', data=raw_data, color='red')

# Set the title of the plot
plt.title('Deaths by Continent - Boxplot and Swarmplot')

# Save the plot to a PNG file with larger size
plt.savefig('box_swarm_plot.png', dpi=300)

# Print a message indicating that the plot was saved
print('Boxplot and swarmplot saved to box_swarm_plot.png')
