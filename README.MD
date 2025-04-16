# Autonomous Data Analysis Bot

## Overview

This project implements an autonomous data analysis bot that automates the process of data analysis. It takes a dataset as input (CSV, XLSX, TXT, JSON), performs data cleaning, feature engineering, and then generates insights and visualizations using a language model.

## Functionality

1. **Data Loading:** Accepts datasets in various formats (CSV, XLSX, TXT, JSON).
2. **Data Cleaning:**
   - Handles missing values using imputation (mean or most frequent).
   - Removes duplicate rows.
   - Validates data types in numeric columns.
3. **Feature Selection:** Utilizes a language model (Gemini) to identify the most relevant columns for analysis.
4. **Data Compression:** Compresses the data for efficient storage and transfer using `gzip`.
5. **Insight Generation and Visualization:**
   - Generates a data analysis report summarizing key insights.
   - Creates Python code for data visualization using Pandas, Matplotlib, and Seaborn.

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
    streamlit run app.py
    ```
6.  **Input Data:**
    *  Select input data in CSV format only (one file).

7.  **Output:** The `app.py` file will generate an `insights.py` file containing the data analysis report and visualization code.  You can then run this Python file to see the visualizations:

    ```bash
    python insights.py
    ```

## Project Structure

├── .streamlit/
│   └── config.toml          # Streamlit configuration file for theme customization
├── Datasets/
│   ├── Raw/                 # Folder for raw datasets
│   └── Processed/           # Folder for processed datasets
├── env/                     # Virtual environment folder
├── Main.py                  # Streamlit app for data analysis
├── app.py                 # Additional script for specific analysis
├── Analysis.ipynb           # Jupyter Notebook for primary analysis
├── README1.md               # Project documentation
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables for API keys
├── sketch.png               # Project-related image


## Environment Variables

*   `API_KEY_1`: API key for the Gemini language model 
*   `API_KEY_2`: API key for the Gemini language model 
*   `API_KEY_3`: API key for the Gemini language model 

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


🚀 **Dive into the world of automated data analysis!**
**Fork, star, and clone this repository to explore its potential.**
**Your contributions can help us make data insights smarter, faster, and more accessible. Let's build the future of data analysis together!** 🌟
```
