import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

"""
This code loads data from a CSV file and creates a heatmap of the Pearson correlation coefficient between a selection of variables.
The resulting heatmap is saved as a PNG file.

Functions:

None
Inputs:

'covid_fat_recov.csv': CSV file containing data to be analyzed
Outputs:

'correl coef.png': PNG file containing the heatmap of the Pearson correlation coefficient between the selected variables
"""S


# Loading the data
raw_data = pd.read_csv('C:/Users/olomuc/OneDrive - MHRA/COVID_ML_MODELS/covid_fat_recov.csv')


# dropping columns we don't need
#raw_data = raw_data.drop(['Country', 'Continent'], axis=1)

# Increases the size of sns plots
sns.set(rc={'figure.figsize':(15,10)})

n_variables = ['Recovered', 'Animal fats', 'Fish Seafood', 
               'Fruits - Excluding Wine', 'Milk - Excluding Butter', 
               'Vegetables', 'Obesity', 'Animal products total']

pc = raw_data[n_variables].corr(method='pearson')

cols = n_variables

ax = sns.heatmap(pc, annot=True,
                 yticklabels=cols,
                 xticklabels=cols,
                 annot_kws={'size':10},
                 cmap="Blues")

# Set the title of the plot
plt.title('correl coef ')

# Save the plot to a PNG file with larger size
plt.savefig('correl coef.png', dpi=300)

# Print a message indicating that the plot was saved
print('correl coef saved to correl coef.png')
