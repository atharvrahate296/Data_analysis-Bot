# LATEST VERSION

import streamlit as st
import pandas as pd
import os
import ast
import bz2
import pickle
import google.generativeai as genai
from dotenv import load_dotenv as dtn
from sklearn.impute import SimpleImputer
import lzma
import zlib
import numpy as np


def process_file(file):
    file.seek(0)
    data = pd.read_csv(file, encoding='utf-8')
    df = detect_anomalies(data)
    return df

def detect_anomalies(df):
    try:
        with st.spinner("Detecting anomalies..."):
            # Check for missing values
            for col in df.columns:
                if df[col].isnull().any():
                    if df[col].dtype == 'object':
                        imputer = SimpleImputer(strategy='most_frequent')
                        df[col] = imputer.fit_transform(df[[col]]).ravel()
                    else:
                        imputer = SimpleImputer(strategy='mean')
                        df[col] = imputer.fit_transform(df[[col]]).ravel()

            # Check for duplicates
            if len(df) != len(df.drop_duplicates()):
                df.drop_duplicates(inplace=True)

        return df

    except Exception as e:
        st.error(f"Error detecting anomalies: {e}")
        return None

def filter_data(data):
    dtn()
    API1 = os.getenv("API_KEY_1")
    if not API1:
        raise ValueError("API key not found! Check your .env file.")
    
    genai.configure(api_key=API1)
    SYS = """You are a Professional Data Analyst Chatbot.
    The user will provide you with a set of columns in a dataset. Your task is to identify and return ONLY the names of the columns that are most critical and insightful for generating meaningful analysis and actionable insights. 
    Exclude columns that are identifiers (e.g., IDs), dates of birth, or other irrelevant metadata unless they are directly useful for analysis. 
    Focus on columns that represent measurable, categorical, or time-based data that can reveal trends, patterns, or relationships. 
    return the column names in a strict list format only.
    Do not give any extra text beside that.
    """

    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction=SYS
    )

    query = f"The dataset contains the following columns: {data.columns.tolist()}. Please identify the columns that are necessary for the analysis task."
    response = model.generate_content(query)

    while True:
        # Attempt to safely evaluate the response as a Python list
        col = ast.literal_eval(response.text)
        if not col:
            st.warning("No relevant columns found by the model.")
            st.write("Retrying... ")
            filter_data(data)
        else:
            return col

def compress_data(data, columns, method='bz2'):
    try:
        selected_df = data[columns]
        data_bytes = pickle.dumps(selected_df)

        if method == 'bz2':
            compressed_data = bz2.compress(data_bytes, compresslevel=9)
        elif method == 'lzma':
            compressed_data = lzma.compress(data_bytes, format=lzma.FORMAT_XZ, preset=9)
        elif method == 'zlib':
            compressed_data = zlib.compress(data_bytes, level=9)
        else:
            raise ValueError("Invalid compression method specified.")
        return compressed_data
    except Exception as e:
        st.error(f"Error compressing data: {e}")
        return None

def gather_insights(compressed_data, columns, compression_method='bz2'):
    try:
        dtn()
        API2 = os.getenv("API_KEY_2")
        if not API2:
            raise ValueError("API key not found! Check your .env file.")        
        genai.configure(api_key=API2)
        SYST = f'''You are a Professional Data Analyst Chatbot. You will be provided with a compressed Pandas DataFrame and a list of relevant columns. The data was compressed using {compression_method}. Your task is to decompress the data, generate a concise data analysis report (maximum 3 paragraphs) summarizing key insights from the data, followed by Python code for data visualization that supports and illustrates these insights. Assume the DataFrame 'df' is read directly from a CSV file specified in the `file_location` variable using pandas.

        The report should:

        *   Be written in a professional and clear tone.
        *   Focus on the most important trends, patterns, and relationships within the data.
        *   Include specific observations and quantifiable metrics (e.g., averages, distributions, correlations) to support your claims.
        *   Present insights in bullet points for easy readability.

        The Python code should:

        *   Use the libraries Pandas, Matplotlib, and Seaborn.
        *   Decompress the provided data to create the DataFrame 'df'.
        *   Generate visualizations that reveal important trends, patterns, and relationships within the data.
        *   Include descriptive statistics, distributions, count plots, scatter plots, box plots, correlation heatmaps, and time series analysis (if a date column is available).
        *   Include appropriate titles, labels, and legends for clarity.
        *   Be well-commented to explain the purpose of each step.
        *   Be executable without errors, assuming the file is accessible at the path given by `file_location` and the relevant columns are present.
        *   Focus on conciseness and clarity, providing a comprehensive overview of the data's key characteristics.
        '''

        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=SYST
        )

        query = f"""Decompress the following data (compressed using {compression_method}) to create a Pandas DataFrame 'df': {compressed_data}. The relevant columns are: {columns}. Provide a data analysis report and Python code for visualization."""

        response = model.generate_content(query)
        return response.text
    except Exception as e:
        st.error(f"Error gathering insights: {e}")
        return None

def write_insights(filename,content):
    try:
        with open(filename, "w") as file:
            file.write(content)
        st.write("Insights and code saved to insights.py. Save or rename the file before running the analysis task again!")
    except Exception as e:
        st.error(f"Error writing insights to file: {e}")

def upload_files():
    uploaded_files = st.file_uploader("Choose CSV files", accept_multiple_files=True, type=["csv"])
    return uploaded_files

def set_navbar(uploaded_files, static_pages):
    chat_pages = [os.path.splitext(file.name)[0] for file in uploaded_files] if uploaded_files else []
    nav_options = static_pages[:2] + chat_pages + static_pages[2:]

    if 'selected_tab' not in st.session_state:
        st.session_state['selected_tab'] = static_pages[0]

    selected_tab = st.sidebar.radio("Navigation", nav_options, index=nav_options.index(st.session_state['selected_tab']))
    st.session_state['selected_tab'] = selected_tab
    return selected_tab, chat_pages

def main():
    st.set_page_config(page_title="Autonomous Data Analysis Bot", page_icon=":mag:", layout="wide")
    st.title("Autonomous Data Analysis Bot")

    static_pages = ["Home", "Dashboard", "About", "Explore"]
    uploaded_files = st.session_state.get('uploaded_files', [])
    selected_tab, chat_pages = set_navbar(uploaded_files, static_pages)

    if selected_tab == "Home":
        st.subheader("Welcome to the Autonomous Data Analysis Bot!")
        st.markdown(
            """
            <div style='display: flex; flex-wrap: wrap; justify-content: space-around;'>
                <div style='width: 45%; margin-bottom: 20px; padding: 20px; border: 1px solid #ddd; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <h3>🔍 Automated Data Insights</h3>
                    <p>
                        Uncover hidden patterns and actionable insights from your data with minimal effort. Our bot automates the entire data analysis pipeline, from cleaning to visualization.
                    </p>
                    <ul>
                        <li>Handles missing values and duplicates</li>
                        <li>Identifies key features for analysis</li>
                        <li>Generates insightful reports and visualizations</li>
                    </ul>
                </div>
                <div style='width: 45%; margin-bottom: 20px; padding: 20px; border: 1px solid #ddd; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <h3>📈 Streamlined Data Processing</h3>
                    <p>
                        Spend less time wrestling with data and more time making informed decisions. Our bot streamlines data processing, allowing you to focus on what matters most.
                    </p>
                    <ul>
                        <li>Supports various CSV file formats</li>
                        <li>Compresses data for efficient storage and transfer</li>
                        <li>Provides downloadable processed data and insights</li>
                    </ul>
                </div>
            </div>

            ## Get Started
            1.  Navigate to the **Dashboard** to upload your CSV files.
            2.  The bot will automatically process the data, clean it, and identify relevant columns.
            3.  Click **Analyze** to generate insights and visualizations.
            4.  Download the generated insights file to explore the results.

            ## Key Features
            -   **Automated Data Cleaning:** Handles missing values and removes duplicates.
            -   **Intelligent Feature Selection:** Identifies the most relevant columns for analysis.
            -   **Efficient Data Compression:** Compresses data for faster processing and storage.
            -   **Insightful Report Generation:** Generates comprehensive data analysis reports with visualizations.
            """,
            unsafe_allow_html=True
        )
    elif selected_tab == "Dashboard":
        st.subheader("Welcome to the Dashboard page!")
        st.write("Upload multiple files for analysis.")
        st.header("Upload CSV Files")
        uploaded_files = upload_files()
        if uploaded_files:
            st.session_state['uploaded_files'] = uploaded_files
            # Directly update selected_tab to the first chat page if available, otherwise keep the current tab
            chat_pages = [os.path.splitext(file.name)[0] for file in uploaded_files] if uploaded_files else []
            if chat_pages and 'selected_tab' not in st.session_state:
                st.session_state['selected_tab'] = chat_pages[0]
            # No need to call set_navbar here, it will be called at the beginning of main()

            for file in uploaded_files:
                st.markdown(f"### {os.path.splitext(file.name)[0]}")
                st.markdown(
                    f"""
                    <div style='border: 2px solid gray; border-radius: 10px; padding: 10px; margin: 10px 0;'>
                        <strong>{os.path.splitext(file.name)[0]}</strong>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:
            st.write("No files uploaded.")

    elif selected_tab == "About":
        st.subheader("About")
        st.write("This application is designed to assist with autonomous data analysis, streamlining data processing, and insights generation.")
        st.write("Developed by a Data Science team to streamline data processing and insights generation.")
        st.markdown("""
        **Our Mission**

        we are dedicated to empowering users with accessible and efficient data analysis tools. Our Autonomous Data Analysis Bot is designed to simplify complex data, uncover valuable insights, and accelerate decision-making processes.

        **Key Features**

        - **Automated Data Cleaning:** Handles missing values and removes duplicates to ensure data quality.
        - **Intelligent Feature Selection:** Uses advanced language models to identify the most relevant columns for analysis.
        - **Efficient Data Compression:** Compresses data for faster processing and storage.
        - **Insightful Report Generation:** Generates comprehensive data analysis reports with visualizations.


        **Connect With Us**

        Stay updated on our latest developments and connect with us

        **Contact Information**

        For inquiries and support, please reach out to us:

        📧 Email: atharvrahate1@gmail.com\n
        📞 Phone: [+91-9302103296](tel:+919302103296)
        """)

    elif selected_tab == "Explore":
        st.subheader("README :")
        with open("README.MD","r")as file:
            content = file.read()
        st.write(content)
        st.write("For more details Visit the Github Repository : [link](https://github.com/atharvrahate296/Data_analysis-Bot)")

    elif selected_tab in chat_pages:
        file = next(f for f in uploaded_files if os.path.splitext(f.name)[0] == selected_tab)
        with st.container():
            with st.spinner("Waiting..."):
                with st.spinner(f"Processing {file.name}..."):
                    # st.write(f"Processing {file.name}...")
                    df = process_file(file)
                    if df is not None:
                        st.write(f"### Data Preview for {file.name}:")
                        st.dataframe(df.head())
                        csv = df.to_csv(index=False).encode('utf-8')
                        df.to_csv(f"Datasets/Processed/{file.name}", index=False)
                        st.download_button(f"Download Processed Data for {file.name}", csv, f"processed_{file.name}", "text/csv")

                        compression_method = st.selectbox(f"Choose compression method for {file.name}",options=['bz2', 'lzma', 'zlib'],index=0)

                        # Filename of the final insights python file
                        outputFilePath = f"Datasets/Processed/insights_{file.name.replace('.csv', '.py')}"

                        if st.button(f"Analyze {file.name}"):
                            with st.spinner(f"Analyzing {file.name}"):
                                columns = filter_data(df)
                                if columns:
                                    st.write(f"Relevant Columns: {columns}")
                                    compressed_data = compress_data(df, columns, method=compression_method)
                                    if compressed_data:
                                        result = gather_insights(df, columns, compression_method)
                                        if result:
                                            
                                            write_insights(outputFilePath, result)
                                            with open(outputFilePath, "rb") as insights_file:
                                                st.download_button(f"Download Insights for {file.name}", insights_file, outputFilePath, "text/x-python")
                                        else:
                                            st.error("Failed to gather insights.")
                                    else:
                                        st.error("Compression failed.")
                                else:
                                    st.warning("No relevant columns found.")

if __name__ == "__main__":
    main()
