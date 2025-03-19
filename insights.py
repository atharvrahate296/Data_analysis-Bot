
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import bz2

# Load the dataset
file_location = 'Datasets/Processed/customer_shopping_data.csv'

df = pd.read_csv(file_location)


# Data Analysis Report
# The analysis of the customer shopping data reveals several key aspects of customer behavior and preferences. The dataset provides a detailed view of customer demographics, purchase categories, spending habits, and preferred payment methods, all tied to specific shopping malls.
#
# Key Insights:
#
# -  Gender Distribution: The distribution of male and female customers provides insights into which gender groups frequent the shopping malls more often.
# -  Age Demographics: Analysis of customer ages helps understand the primary age groups that shop at these malls, which can inform targeted marketing strategies.
# -  Purchase Categories: Examining the most popular purchase categories allows for optimizing product offerings and inventory management.
# -  Payment Method Preferences: Understanding the preferred payment methods helps tailor transaction processes to customer convenience.
# -  Shopping Mall Popularity: Identifying the most frequented shopping malls can assist in resource allocation and marketing efforts.

# Python Code for Data Visualization
# This code performs a comprehensive visual analysis of the customer shopping data, using Pandas for data manipulation and Seaborn/Matplotlib for creating informative visualizations. The goal is to uncover trends, patterns, and relationships within the data, which can provide valuable insights for business decision-making.

# Set the style for seaborn plots
sns.set(style='whitegrid')

# 1. Gender Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='gender', data=df)
plt.title('Distribution of Customers by Gender')
plt.xlabel('Gender')
plt.ylabel('Number of Customers')
plt.show()

# 2. Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['age'], kde=True)
plt.title('Distribution of Customer Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# 3. Category Purchases
plt.figure(figsize=(12, 6))
sns.countplot(x='category', data=df, order=df['category'].value_counts().index)
plt.title('Distribution of Purchase Categories')
plt.xlabel('Category')
plt.ylabel('Number of Purchases')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 4. Payment Method Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='payment_method', data=df)
plt.title('Distribution of Payment Methods')
plt.xlabel('Payment Method')
plt.ylabel('Number of Transactions')
plt.show()

# 5. Shopping Mall Popularity
plt.figure(figsize=(12, 6))
sns.countplot(x='shopping_mall', data=df, order=df['shopping_mall'].value_counts().index)
plt.title('Distribution of Customers by Shopping Mall')
plt.xlabel('Shopping Mall')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 6. Price Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['price'], kde=True)
plt.title('Distribution of Purchase Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# 7. Quantity Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='quantity', data=df)
plt.title('Distribution of Purchase Quantities')
plt.xlabel('Quantity')
plt.ylabel('Number of Purchases')
plt.show()