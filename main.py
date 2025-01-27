import os
import pdfplumber
import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Paths
folder_path = "Data/Pricelist"  # Folder containing PDFs
output_table_folder = "Data/Output_tables"  # Folder to save extracted tables as CSVs
training_data_folder = "Data/Training_data"  # Folder with labeled JSON data for training
required_output_folder = "Data/Required_output_data"  # Folder to save structured JSON
merged_output_file = "Data/merged_output.json"  # Path for final merged JSON

# Step 1: Extract tables from PDFs
os.makedirs(output_table_folder, exist_ok=True)
table_counter = 1
pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pdf')]

for pdf_file in pdf_files:
    try:
        print(f"Processing {os.path.basename(pdf_file)}...")
        with pdfplumber.open(pdf_file) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables()
                if tables:
                    for i, table in enumerate(tables):
                        df = pd.DataFrame(table)
                        if len(df.columns) > 2:
                            output_csv = os.path.join(output_table_folder, f"table{table_counter}.csv")
                            df.to_csv(output_csv, index=False, header=False)
                            print(f"Saved table {table_counter} to {output_csv}")
                            table_counter += 1
                else:
                    print(f"No tables found on page {page_number} of {os.path.basename(pdf_file)}.")
    except Exception as e:
        print(f"Error processing {pdf_file}: {e}")

# Step 2: Train Model for Table Classification
def load_training_data(json_folder):
    data = []
    labels = []
    for file in os.listdir(json_folder):
        if file.endswith(".json"):
            with open(os.path.join(json_folder, file), "r") as f:
                content = json.load(f)
                for entry in content:
                    combined_text = " ".join(entry.values())
                    data.append(combined_text)
                    labels.append(entry.get("label", "irrelevant"))
    return data, labels

def train_model(data, labels):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data)
    y = labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    joblib.dump(vectorizer, "vectorizer.pkl")
    joblib.dump(model, "table_classifier.pkl")

data, labels = load_training_data(training_data_folder)
train_model(data, labels)

# Step 3: Classify and Process Tables
ROLE_RULES = {
    "ID": ['Cat. NO', 'Order no', 'Diamond Chain Part No', 'Part No.', 'Catalog No.', 'Cat.Nos', 'IDH No.'],
    "title": ['type', 'electrical product names', 'Carding Machine'],
    "description": ['description', 'Type and frame size', 'Product Description', 'in mm', 'in inches'],
    "price": ['M.R.P.', 'LP in INR', 'PRICE IN RS', 'MRP Per Metre', ' Rs.   P.', 'MRP Per Metre Rs.   P.', 'Price', 'MRP*  / Unit', 'MRP', 'price']
}

def classify_and_process_tables(table_folder, output_folder):
    vectorizer = joblib.load("vectorizer.pkl")
    classifier = joblib.load("table_classifier.pkl")
    os.makedirs(output_folder, exist_ok=True)
    csv_files = [os.path.join(table_folder, file) for file in os.listdir(table_folder) if file.endswith(".csv")]
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        column_names = " ".join(list(df.columns))
        content_sample = " ".join(str(df.head(3).values.flatten()))
        num_rows, num_cols = df.shape
        combined_text = column_names + " " + content_sample
        text_features = vectorizer.transform([combined_text]).toarray()
        feature_vector = np.hstack([text_features, [[num_rows, num_cols]]])
        prediction = classifier.predict(feature_vector)
        
        if prediction[0] == "required":
            print(f"Processing required table: {csv_file}")
            column_roles = {}
            for col in df.columns:
                role = "attributes"
                for key, keywords in ROLE_RULES.items():
                    if any(keyword.lower() in col.lower() for keyword in keywords):
                        role = key
                        break
                column_roles[col] = role
            
            structured_data = []
            for _, row in df.iterrows():
                json_entry = {"ID": None, "title": None, "description": None, "price": None, "attributes": {}}
                for col, value in row.items():
                    role = column_roles.get(col, "attributes")
                    if role in json_entry:
                        json_entry[role] = value
                    else:
                        json_entry["attributes"][col] = value
                structured_data.append(json_entry)
            
            output_file = os.path.join(output_folder, os.path.basename(csv_file).replace(".csv", ".json"))
            with open(output_file, "w") as f:
                json.dump(structured_data, f, indent=4)
            print(f"Structured data saved to {output_file}")

classify_and_process_tables(output_table_folder, required_output_folder)

# Step 4: Merge JSON Files
merged_data = []
for json_file in os.listdir(required_output_folder):
    if json_file.endswith(".json"):
        file_path = os.path.join(required_output_folder, json_file)
        with open(file_path, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                merged_data.extend(data)
            else:
                merged_data.append(data)

with open(merged_output_file, "w") as f:
    json.dump(merged_data, f, indent=4)
print(f"All JSON files merged into {merged_output_file}")
