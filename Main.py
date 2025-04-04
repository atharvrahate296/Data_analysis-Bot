import streamlit as st
import pandas as pd
import os
import ast
import bz2
import pickle
import google.generativeai as genai
from dotenv import load_dotenv as dtn
from sklearn.impute import SimpleImputer

def process_file(file):
    df = pd.read_csv(file)
    df = detect_anomalies(df)
    return df

def detect_anomalies(df):
    st.write("Detecting anomalies...")
    
    # Check for missing values
    for col in df.columns:
        if df[col].isnull().any():
            st.write(f"Found missing values in {col}! Imputing...")
            if df[col].dtype == 'object':
                imputer = SimpleImputer(strategy='most_frequent')
                df[col] = imputer.fit_transform(df[[col]]).ravel()
            else:
                imputer = SimpleImputer(strategy='mean')
                df[col] = imputer.fit_transform(df[[col]]).ravel()
        
    # Check for duplicates
    if len(df) != len(df.drop_duplicates()):
        st.write("Found duplicate rows! Removing...")
        df.drop_duplicates(inplace=True)
    
    return df

def filter_data(data):
    # Placeholder for the actual filter function
    dtn()
    api_key = os.getenv("API_KEY_1")
    if not api_key:
        raise ValueError("API key not found! Check your .env file.")
    genai.configure(api_key=api_key)
    SYS = """You are a Professional Data Analyst Chatbot. The user will provide you with a set of columns in a dataset. Your task is to identify and return ONLY the names of the columns that are most critical and insightful for generating meaningful analysis and actionable insights. Exclude columns that are identifiers (e.g., IDs), dates of birth, or other irrelevant metadata unless they are directly useful for analysis. Focus on columns that represent measurable, categorical, or time-based data that can reveal trends, patterns, or relationships.

    For example:
    If the columns are ['customer_id', 'name', 'age', 'city', 'purchase_amount', 'date'], you should return:
    ['age', 'city', 'purchase_amount', 'date'] as these are the most relevant for analysis.
    """
    # Create the model with the updated system prompt
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction=SYS
    )
    query = f"The dataset contains the following columns: {data.columns.tolist()}. Please identify the columns that are necessary for the analysis task." 
    # Generate the response using the query
    response = model.generate_content(query)
    # return response.text.toList()
    try:
        col =  ast.literal_eval(response.text)
        # print(col)
        # return df.columns.tolist()
        return col
    except (SyntaxError, ValueError):
        return None  # Or raise an exception, depending on your error handling preference


def compress_data(data, columns):
    # Placeholder for the actual compression function
    # Drop unnecessary columns
    selected_df = data[columns]
    # Convert DataFrame to byte string using pickle
    data_bytes = pickle.dumps(selected_df)
    # Compress the byte string using bz2
    compressed_data = bz2.compress(data_bytes, compresslevel=9)
    return compressed_data
    # return df[columns]

def gather_insights( compressed_data, columns):
    # Placeholder for the actual insights gathering function
    dtn()
    # api_key = os.getenv("API_KEY_2")
    api_key = os.getenv("API_KEY_3")
    if not api_key:
        raise ValueError("API key not found! Check your .env file.")        
    genai.configure(api_key=api_key)
    SYST = '''You are a Professional Data Analyst Chatbot. You will be provided with a Pandas DataFrame named 'df' and a list of relevant columns identified in the previous step. Your task is to generate a concise data analysis report (maximum 3 paragraphs) summarizing key insights from the data, followed by Python code for data visualization that supports and illustrates these insights.  Assume the DataFrame 'df' is read directly from the file path specified in the `file_location` variable using pandas.

    The report should:

    *   Be written in a professional and clear tone.
    *   Focus on the most important trends, patterns, and relationships within the data.
    *   Include specific observations and quantifiable metrics (e.g., averages, distributions, correlations) to support your claims.
    *   Present insights in bullet points for easy readability.

    The Python code should:

    *   Use the libraries Pandas, Matplotlib, and Seaborn.
    *   Read the DataFrame 'df' directly from the path specified in the `file_location` variable using pandas.
    *   Generate visualizations that reveal important trends, patterns, and relationships within the data.
    *   Include descriptive statistics, distributions, count plots, scatter plots, box plots, correlation heatmaps, and time series analysis (if a date column is available).
    *   Include appropriate titles, labels, and legends for clarity.
    *   Be well-commented to explain the purpose of each step.
    *   Be executable without errors, assuming the file is accessible at the path given by `file_location` and the relevant columns are present.
    *   Focus on conciseness and clarity, providing a comprehensive overview of the data's key characteristics.

    Example (for a customer shopping dataset):

    **Input:**

    *   Relevant Columns: `['age', 'city', 'purchase_amount', 'date']`
    *   `file_location = 'Datasets/Raw/customer_shopping_data.csv'`

    **Expected Output:**

    **Data Analysis Report:**

    The analysis of customer data reveals several key insights regarding purchasing behavior. Customers in the dataset range in age, with the largest group falling between 25 and 45 years old. Purchase amounts vary significantly, with a notable peak in transactions occurring during specific periods.

    Key Insights Example:

    *   The average purchase amount is $X, with a standard deviation of $Y.
    *   Customers in City A tend to spend Z% more than customers in City B.
    *   Purchase amounts show a positive correlation with age, particularly for customers over 50.
    *   There is a significant increase in purchase activity during the months of November and December.

    **Example Python Code:**

    ```python
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    file_location = 'Datasets/Raw/customer_shopping_data.csv'
    df = pd.read_csv(file_location)

    # Set the style for seaborn plots
    sns.set(style='whitegrid')

    # 1. Age Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['age'], kde=True)
    plt.title('Distribution of Customer Ages')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.show()

    # 2. Purchase Amount by City
    plt.figure(figsize=(12, 7))
    sns.boxplot(x='city', y='purchase_amount', data=df)
    plt.title('Purchase Amount Distribution by City')
    plt.xlabel('City')
    plt.ylabel('Purchase Amount')
    plt.show()

    # 3. Purchase Amount over Time
    df['date'] = pd.to_datetime(df['date')
    df['month'] = df['date'].dt.month
    monthly_purchases = df.groupby('month')['purchase_amount'].sum()
    plt.figure(figsize=(12, 7))
    monthly_purchases.plot(kind='line', marker='o')
    plt.title('Total Purchase Amount Over Time')
    plt.xlabel('Month')
    plt.ylabel('Total Purchase Amount')
    plt.show()

    # 4. Correlation Heatmap
    correlation_matrix = df[['age', 'purchase_amount']].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap of Age and Purchase Amount')
    plt.show()'''

    # Create the model with the updated system prompt
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction=SYST
    )
    query = f"df = {compressed_data}, columns = {columns}"
    # Generate the response using the query
    response = model.generate_content(query)
    return response.text
    # return "# Insights code\nprint('Insights generated')"

def write_insights(filename,content):
    with open(filename, "w") as file:
                file.write(content)
    st.write("Insights and code saved to insights.py. Save or rename the file before running the analysis task again!")

def main():
    st.set_page_config(page_title="Autonomous Data Analysis Bot", page_icon=":mag:", layout="wide")
    st.title("Autonomous Data Analysis Bot")
    st.write("Upload multiple files for analysis.")
    st.header("Upload CSV Files")
    
    uploaded_files = st.file_uploader("Choose CSV files", accept_multiple_files=True, type=["csv"])
    
    if uploaded_files:
        dataframes = []
        for file in uploaded_files:
            st.write(f"Processing {file.name}...")
            df = process_file(file)
            dataframes.append(df)
        
        combined_df = pd.concat(dataframes, ignore_index=True)
        st.write("### Combined Data Preview:")
        st.dataframe(combined_df.head())
        
        # Option to download processed data
        csv = combined_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Processed Data", csv, "processed_data.csv", "text/csv")
        
        # Do Analysis button
        if st.button("Do Analysis"):
            # columns = filter_data(combined_df)
            # st.write(f"Relevant Columns: {columns}")
            while True:
                # handle none value error
                columns = filter_data(combined_df)
                st.write(f"Relevant Columns: {columns}")
                if columns == None:
                    st.write("No relevant columns found for analysis.\nRetrying...")
                else: 
                    break
            compressed_data = compress_data(combined_df, columns)
            result = gather_insights( compressed_data, columns)
            #write insights and code to a file
            filename = "insights.py"
            write_insights(filename,result) 
            
            with open(filename, "rb") as file:
                st.download_button("Download Insights File", file, filename, "text/x-python")

if __name__ == "__main__":
    main()
