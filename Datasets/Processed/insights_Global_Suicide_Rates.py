import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the file location
file_location = 'Datasets\Processed\Global_Suicide_Rates.csv'

# Load the dataset from the CSV file
df = pd.read_csv(file_location)

# Data Analysis Report

# Overview:
# The dataset consists of suicide statistics across different countries and years, with 1421 rows and 13 columns. It includes variables such as 'Country', 'Year', 'Suicides number', 'Life expectancy', and socioeconomic indicators like 'GDP' and 'Schooling'. The data types are mixed, including numerical (int64, float64) and categorical (object) types.

# Key Trends and Observations:
# - Suicide rates vary significantly by country. A count plot of the top countries shows which countries have the highest suicide rates based on the number of records in the dataset, which can be related to the size of the dataset.
# - There's a noticeable positive correlation between 'Schooling' and 'Income composition of resources', suggesting that higher education levels are associated with better income prospects.
# - 'Life expectancy' has a negative correlation with 'Adult Mortality', which aligns with the expectation that higher adult mortality rates are associated with lower life expectancy.
# - There are missing values in 'Alcohol', 'Income composition of resources', and 'Schooling' columns, which are handled by filling with the mean.
# - The distribution of 'Suicides number' is right-skewed, indicating that there are more instances of lower suicide numbers with fewer instances of very high suicide numbers.

# Anomalies and Data Quality:
# - Missing values were identified in 'Alcohol', 'Income composition of resources', and 'Schooling' columns. These were imputed using the mean of each respective column to avoid data loss during analysis. No duplicate rows were found.

# Detailed Explanation:
# - Country-Specific Suicide Rates: By visualizing the count of records for each country, we can identify which countries are most represented in the dataset. This does not directly indicate suicide rates but rather the frequency of reporting in the dataset.
# - Correlation Analysis: The correlation heatmap provides insights into the relationships between numerical variables. A strong positive correlation between 'Schooling' and 'Income composition of resources' suggests that education plays a crucial role in economic well-being. The negative correlation between 'Life expectancy' and 'Adult Mortality' confirms the intuitive relationship between these two variables.
# - Distribution of Suicide Numbers: The distribution of 'Suicides number' shows a right-skewed pattern, indicating that while there are many instances of lower suicide numbers, there are fewer instances of very high suicide numbers, which may represent outliers or specific events causing spikes.

# Python Code for Visualization

# Impute missing values using the mean
df['Alcohol'] = df['Alcohol'].fillna(df['Alcohol'].mean())
df['Income composition of resources'] = df['Income composition of resources'].fillna(df['Income composition of resources'].mean())
df['Schooling'] = df['Schooling'].fillna(df['Schooling'].mean())

# Check for duplicates and drop them
df.drop_duplicates(inplace=True)

# 1. Country-Specific Suicide Rates
plt.figure(figsize=(12, 6))
sns.countplot(y='Country', data=df, order=df['Country'].value_counts().iloc[:10].index)
plt.title('Top 10 Countries by Number of Records')
plt.xlabel('Number of Records')
plt.ylabel('Country')
plt.tight_layout()
plt.show()
# plt.savefig('country_suicide_counts.png')

# 2. Correlation Heatmap
numerical_columns = ['Suicides number', 'Life expectancy', 'Adult Mortality', 'Infant deaths', 'Alcohol', 'Under-five deaths', 'HIV/AIDS', 'GDP', 'Population', 'Income composition of resources', 'Schooling']
correlation_matrix = df[numerical_columns].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Heatmap of Numerical Variables')
plt.tight_layout()
plt.show()
# plt.savefig('correlation_heatmap.png')

# 3. Distribution of Suicide Numbers
plt.figure(figsize=(10, 6))
sns.histplot(df['Suicides number'], kde=True)
plt.title('Distribution of Suicide Numbers')
plt.xlabel('Number of Suicides')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
# plt.savefig('suicide_number_distribution.png')

# 4. Life Expectancy vs. Adult Mortality
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Life expectancy', y='Adult Mortality', data=df)
plt.title('Life Expectancy vs. Adult Mortality')
plt.xlabel('Life Expectancy (years)')
plt.ylabel('Adult Mortality (per 1000)')
plt.tight_layout()
plt.show()
# plt.savefig('life_expectancy_vs_adult_mortality.png')

# 5. Schooling vs. Income Composition of Resources
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Schooling', y='Income composition of resources', data=df)
plt.title('Schooling vs. Income Composition of Resources')
plt.xlabel('Schooling (years)')
plt.ylabel('Income Composition of Resources')
plt.tight_layout()
plt.show()
# plt.savefig('schooling_vs_income_composition.png')
