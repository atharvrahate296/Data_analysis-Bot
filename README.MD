# Autonomous Data Analysis Bot

## Overview

This project implements an autonomous data analysis bot that automates the process of data analysis. It takes a dataset as input (CSV, XLSX, TXT, JSON), performs data cleaning, feature engineering, and then generates insights and visualizations using a language model.

## Functionality

The bot performs the following key functions:

1.  **Data Loading:**  Accepts datasets in various formats (CSV, XLSX, TXT, JSON).
2.  **Data Cleaning:**
    *   Handles missing values using imputation (mean or most frequent).
    *   Removes duplicate rows.
    *   Validates data types in numeric columns.
3.  **Feature Selection:**
    *   Utilizes a language model (Gemini) to identify the most relevant columns for analysis. This step is crucial for focusing on the most important features.
4.  **Data Compression:**
    *   Compresses the data for efficient storage and transfer.  Uses `gzip` compression  in `Main.ipynb`.
5.  **Insight Generation and Visualization:**
    *   Uses a language model (Gemini) to generate a data analysis report summarizing key insights.
    *   Creates Python code for data visualization (using Pandas, Matplotlib, and Seaborn) to illustrate these insights. The code is designed to be saved to a `.py` file and executed.

## Usage

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/atharvrahate296/Data_analysis-Bot
    cd Data_analysis-Bot
    ```

2. **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv env
    Activate:
    # On Windows:
    .\env\Scripts\activate
    # On macOS and Linux:
    source env/bin/activate
    ```

3.  **Install Dependencies:** Make sure you have the necessary Python packages installed. You can install them using `pip`:

    ```bash
    pip install -r requirements.txt
    ```

4.  **API Keys:**  You will need to obtain API keys for the Gemini language model.  Set these keys as environment variables.  Create a `.env` file in the same directory as your notebook and add the following lines, replacing `"YOUR_API_KEY"` with your actual API keys (Obtain at least two or three if possible, as the bot uses different keys for different tasks):

    ```
    API_KEY_1="YOUR_API_KEY_1"
    API_KEY_2="YOUR_API_KEY_2"
    API_KEY_3="YOUR_API_KEY_3"
    ```

5.  **Run the Main file:** 
    ```bash
    streamlit run Main.py
    ```
6.  **Input Data:**
    *  Select input data in CSV format only (one file).

7.  **Output:** The `Main.py` file will generate an `insights.py` file containing the data analysis report and visualization code.  You can then run this Python file to see the visualizations:

    ```bash
    python insights.py
    ```

## Directory Structure

```
├── Analysis.ipynb       # Jupyter Notebook containing the primary analysis code
├── Analysis.ipynb       # Jupyter Notebook containing the primary analysis code 
├── Main.py              # streamlit file for loading and processing data 
├── insights.py         # Output file containing the generated insights and code (from Analysis.ipynb) 
├── Datasets 
│ ├── Raw                  # Folder for storing raw input datasets 
│    └── customer_shopping_data.csv (example) │ │
│ └── Processed             # Folder for storing processed datasets (currently not actively used) 
│    └── customer_shopping_data.csv 
├── requirements.txt    # List of Python package dependencies 
├── .gitignore 
└── README.md           # This file
```

## Environment Variables

*   `API_KEY_1`: API key for the Gemini language model (used for feature selection in `Main.py`).
*   `API_KEY_2`: API key for the Gemini language model (used for insight generation in `Main.py`).
*   `API_KEY_3`: API key for the Gemini language model (used for feature selection in `Main.py`).

## Future Improvements

*   Implement more advanced data cleaning techniques.
*   Add support for more complex feature engineering.
*   Allow users to customize the analysis process through configuration options.
*   Perform analysis on various file formats including .pdf files.
*   Generate insights on larger and more complex datasets that aren't possible now due to limitations of context window and authentic data compression techniques.
*   Improve the quality of the generated insights and visualizations.
*   Implement better error handling and logging.
*   Add unit tests.
*   Explore more efficient data compression algorithms.
```
