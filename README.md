Project: PDF Table Extractor and Classifier

Description:
This project extracts tables from PDF files, classifies them, and processes the required tables into structured JSON format. The final output is a merged JSON file containing all the structured data.

Dependencies:
- Python 3.x
- pdfplumber
- pandas
- numpy
- scikit-learn
- joblib

Installation:
1. Install the required Python packages using pip:

Usage:
1. Place the PDF files in the Data/Pricelist folder.
2. Place the labeled JSON training data in the Data/Training_data folder.
3. Run the script 1.py to perform the following steps:
- Extract tables from PDFs and save them as CSV files in the Data/Output_tables folder.
- Train a RandomForestClassifier model using the labeled JSON data.
- Classify and process the extracted tables, saving the structured JSON files in the Data/Required_output_data folder.
- Merge all the structured JSON files into a single JSON file Data/merged_output.json.

Script Breakdown:
- Step 1: Extract tables from PDFs and save them as CSV files.
- Step 2: Train a model for table classification using labeled JSON data.
- Step 3: Classify and process the extracted tables, saving the structured JSON files.
- Step 4: Merge all the structured JSON files into a single JSON file.

Folder Structure:
- Data/
  - Pricelist/ (Folder containing PDFs)
  - Output_tables/ (Folder to save extracted tables as CSVs)
  - Training_data/ (Folder with labeled JSON data for training)
  - Required_output_data/ (Folder to save structured JSON)
  - merged_output.json (Final merged JSON file)

