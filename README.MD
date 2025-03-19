# Autonomous Data Analysis Bot

## Overview

This project implements an autonomous data analysis bot that takes a dataset as input (CSV, XLSX, TXT, JSON), performs data cleaning, feature engineering, and transformation automatically, and then generates insights and visualizations.

## Functionality

The bot performs the following key functions:

1.  **Data Loading:**  Accepts datasets in various formats (CSV, XLSX, TXT, JSON).
2.  **Data Cleaning:**
    *   Handles missing values using imputation (mean or most frequent).
    *   Removes duplicate rows.
    *   Validates data types in numeric columns.
3.  **Feature Selection:**
    *   Utilizes a language model (Gemini) to identify the most relevant columns for analysis.  This step is crucial for focusing on the most important features.
4.  **Data Compression:**
    *   Compresses the selected data using `bz2` for efficient storage.
5.  **Insight Generation and Visualization:**
    *   Uses a language model (Gemini) to generate a data analysis report summarizing key insights.
    *   Creates Python code for data visualization (using Pandas, Matplotlib, and Seaborn) to illustrate these insights.  The code is designed to be saved to a `.py` file and executed.

## Usage

1. **Clone and create Virtual Environment :** use git clone and create a virtual environment before running the project

    ```bash
    git clone https://github.com/atharvrahate296/Data_analysis-Bot
    python -m venv env
    .\env\Scripts\activate
    ```

2.  **Install Dependencies:** Make sure you have the necessary Python packages installed. You can install them using `pip`:

    ```bash
    pip install -r requirements.txt
    ```

3.  **API Keys:**  You will need to obtain API keys for the Gemini language model.  Set these keys as environment variables.  Create a `.env` file in the same directory as your notebook and add the following lines, replacing `"YOUR_API_KEY"` with your actual API keys(Obtain at least three if possible):

    ```

    API_KEY_2="YOUR_API_KEY_2"
    API_KEY_3="YOUR_API_KEY_3"
    ```

4.  **Run the Notebook:**  Open and run the `Analysis.ipynb` notebook.

5.  **Input Data:** Place your dataset file (e.g., `customer_shopping_data.csv`) in the `Datasets/Raw/` directory.  If these folders dont exist create them

6.  **Output:** The notebook will generate an `insights.py` file containing the data analysis report and visualization code.  You can then run this Python file to see the visualizations:

    ```bash
    python insights.py
    ```

## Directory Structure

```
.
├── Analysis.ipynb       # Jupyter Notebook containing the analysis code
├── insights.py         # Output file containing the generated insights and code
├── Datasets
│   ├── Raw                  # Folder for storing raw input datasets
│   │   └── customer_shopping_data.csv
│   └── Processed          # Folder for storing processed datasets
│       └── customer_shopping_data.csv
└── README.md            # This file
```

## Environment Variables

*   `API_KEY_2`: API key for the Gemini language model (used for insight generation).
*   `API_KEY_3`: API key for the Gemini language model (used for feature selection).

## Notes

*   The quality of the generated insights and visualizations depends on the quality and structure of the input data.
*   The language model may require some fine-tuning or adjustments to the prompts to achieve optimal results for different datasets.
*   The `insights.py` file is overwritten each time the notebook is run.  Be sure to save or rename it if you want to keep the results.

## Future Improvements

*   Implement more advanced data cleaning techniques.
*   Add support for more complex feature engineering.
*   Allow users to customize the analysis process through configuration options.
*   Perform analysis on various file formats including .xlsv .json and pdf files.
*   Better accuracy in handling the various difficulties occuring during the analysis on diverse datasets.
*   Improve the quality of the generated insights and visualizations.
*   Implement better error handling and logging.
*   Add unit tests.