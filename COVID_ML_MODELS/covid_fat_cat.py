import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

"""
This code loads a raw data file of COVID-19 fatality rates by country and conducts some basic data processing
operations before creating a bar plot of deaths by continent and saving the plot to a PNG file.

Functions:

None
Inputs:

A raw data file of COVID-19 fatality rates by country in CSV format.
Outputs:

A bar plot of deaths by continent saved as a PNG file.
A message indicating that the plot was saved.
"""

#1.LOADING RAW DATA

# Loading the data
raw_data = pd.read_csv('C:/Users/olomuc/OneDrive - MHRA/COVID_ML_MODELS/covid_fat_recov.csv')
print(raw_data.shape)

#runs the first 5 rows
print(raw_data.head())

#2. DATA PROCESSING

#for column in raw_data:
    #unique_vals = np.unique(raw_data[column])
    #nr_values = len(unique_vals)
    #if nr_values < 10:
        #print('The number of values for feature {} :{} -- {}'.format(column, nr_values,unique_vals))
    #else:
        #print('The number of values for feature {} :{}'.format(column, nr_values))




# enable automatic padding for the plot
plt.rcParams.update({'figure.autolayout': True})

# create a bar plot of deaths by continent
ax = sns.barplot(x="Continent", y="Recovered", data=raw_data)

# add labels to the bars
ax.bar_label(ax.containers[0])

# wrap the text of the x-axis font
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
ax.tick_params(axis="x", pad=5)

# save the plot to a PNG file
plt.savefig('Barplot_matrix.png')

# print a message indicating that the plot was saved
print('Barplot matrix saved to barplot_matrix.png')



