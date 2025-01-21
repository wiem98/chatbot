import pdfplumber
import re
from rapidfuzz import fuzz, process
import logging
import pandas as pd
import unicodedata
from charset_normalizer import detect

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", filename="extractor.log", filemode="a")

class PDFDataExtractor:
    def __init__(self, product_table, fuzzy_threshold=80):
        self.product_table = product_table  # DataFrame with product references and descriptions
        self.fuzzy_threshold = fuzzy_threshold

    def normalize_text(self, text):
        """Normalize text to handle special characters and encoding issues."""
        try:
            # Decode using 'latin1' or 'cp1252' for corrupted characters
            text = text.encode('latin1').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass  # Keep text as is if decoding fails

        # Replace common corrupted characters
        replacements = {
            "�": "é",  # Replace common corrupted characters
            
        }
        for corrupted, fixed in replacements.items():
            text = text.replace(corrupted, fixed)

        # Normalize to NFC to ensure accents are composed
        text = unicodedata.normalize('NFC', text)
        return text


    def detect_pdf_encoding(pdf_path):
        with open(pdf_path, "rb") as pdf_file:
            result = detect(pdf_file.read())
        logging.info(f"Detected PDF encoding: {result['encoding']}")
        return result["encoding"]


    def extract_text_and_tables(self, pdf_path):
        """Extract text and tables from a PDF."""
        data = {"text": "", "tables": []}
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    raw_text = page.extract_text()
                    if raw_text:
                        # Attempt to decode and fix encoding issues
                        try:
                            fixed_text = raw_text.encode('cp1252').decode('utf-8')
                        except (UnicodeEncodeError, UnicodeDecodeError):
                            fixed_text = raw_text  # Use raw text if decoding fails

                        # Normalize the fixed text
                        normalized_text = self.normalize_text(fixed_text)
                        data["text"] += normalized_text + "\n"
                    tables = page.extract_tables()
                    for table in tables:
                        data["tables"].append(table)
            logging.info(f"Text and tables extracted successfully from {pdf_path}.")
            logging.info(f"Raw extracted text: {raw_text}")
            logging.info(f"Normalized text: {normalized_text}")

            return data
        except Exception as e:
            logging.error(f"Failed to extract data from PDF: {e}")
            return data
        
    def match_description(self, input_text):
        """Extract only the designation from the input text and match it with product descriptions."""
        # Predefined descriptions from the product table
        descriptions = self.product_table["designation"].tolist()
        normalized_descriptions = [self.normalize_text(desc.lower().strip()) for desc in descriptions]

        # Extract designation by removing quantity and units
        designation_pattern = r"(?<=\bde\s)(.+)"  # Matches text after 'de' which is typically the designation
        match = re.search(designation_pattern, input_text, flags=re.IGNORECASE)
        if match:
            designation = match.group(1).strip()  # Extract the designation
            normalized_text = self.normalize_text(designation.lower())
        else:
            logging.warning(f"Could not extract designation from: {input_text}")
            return None  # Return None if no designation is found

        # Perform fuzzy matching
        matched = process.extractOne(normalized_text, normalized_descriptions, scorer=fuzz.ratio)
        if matched and matched[1] >= self.fuzzy_threshold:
            # Return the matched description
            return descriptions[normalized_descriptions.index(matched[0])]
        logging.warning(f"No match found for designation: {normalized_text}")
        return designation  # Return designation as fallback

    def extract_fields(self, text):
        """Extract fields such as product lines, addresses, and company names from text."""
        patterns = {
            "product_line": r"^\-\s*(.+)",  # Match lines starting with a dash
            "quantity": r"(\d+|un|une|deux|trois|quatre|cinq|six|sept|huit|neuf|dix)\s*(unit[e\u00e9]s?|pcs|pieces?|qte)?",
            "address": r"(?:Adresse:|Address:)?\s*([A-Za-z0-9,.'\-/\s]+[A-Za-z])",
            "company_name": r"(?:Soci[eé]t[eé]:|Entreprise:)\s*([\w\s,'-]+)"
        }

        def convert_french_number(word):
            french_to_number = {
                "un": 1, "une": 1,
                "deux": 2,
                "trois": 3,
                "quatre": 4,
                "cinq": 5,
                "six": 6,
                "sept": 7,
                "huit": 8,
                "neuf": 9,
                "dix": 10
            }
            return french_to_number.get(word.lower(), word)

        extracted_data = []  # List to store all extracted products
        extracted_info = {"address": None, "company_name": None}
        lines = text.split("\n")

        for line in lines:
            logging.info(f"Processing line: {line}")
            line = self.normalize_text(line)

            # Detect product lines
            product_match = re.search(patterns["product_line"], line)
            if product_match:
                product_text = product_match.group(1).strip()
                logging.info(f"Detected product line: {product_text}")

                # Extract quantity
                quantity_match = re.search(patterns["quantity"], product_text, re.IGNORECASE)
                if quantity_match:
                    raw_quantity = quantity_match.group(1)
                    quantity = convert_french_number(raw_quantity)
                    logging.info(f"Extracted quantity: {quantity}")
                    product_text = re.sub(patterns["quantity"], "", product_text).strip()
                else:
                    quantity = "Unknown"

                # Match description and retrieve reference
                matched_description = self.match_description(product_text)
                if matched_description:
                    try:
                        reference = self.product_table.loc[
                            self.product_table["designation"].str.contains(matched_description, na=False, case=False),
                            "ref"
                        ].iloc[0] if not self.product_table.empty else None
                    except IndexError:
                        reference = None
                        logging.warning(f"No reference found for description: {matched_description}")
                else:
                    reference = None
                    logging.warning(f"Unmatched product line: {product_text}")

                # Add extracted product data
                extracted_data.append({
                    "reference": reference,
                    "description": matched_description if matched_description else product_text,
                    "quantity": quantity
                })

            # Extract address (only once)
            if not extracted_info["address"]:
                address_match = re.search(patterns["address"], line, re.IGNORECASE)
                if address_match:
                    extracted_info["address"] = address_match.group(1).strip()
                    logging.info(f"Address extracted: {extracted_info['address']}")

            # Extract company name (only once)
            if not extracted_info["company_name"]:
                company_name_match = re.search(patterns["company_name"], line, re.IGNORECASE)
                if company_name_match:
                    extracted_info["company_name"] = company_name_match.group(1).strip()
                    logging.info(f"Company name extracted: {extracted_info['company_name']}")

        # Ensure all items are processed and logged
        if not extracted_data:
            logging.warning("No items extracted. Check input text format.")
        else:
            logging.info(f"Extracted {len(extracted_data)} items: {extracted_data}")

        return {"items": extracted_data, **extracted_info}


    def process_table(self, table):
        """Convert table data into structured format."""
        structured_table = [dict(zip(table[0], row)) for row in table[1:]] if table else []
        logging.info(f"Processed table: {structured_table}")
        return structured_table

    def analyze_pdf(self, pdf_path):
        """Analyze the PDF and extract structured data."""
        extracted_data = self.extract_text_and_tables(pdf_path)
        logging.info(f"Final extracted items: {extracted_data}")

        text_fields = self.extract_fields(extracted_data["text"])
        structured_tables = [self.process_table(table) for table in extracted_data["tables"]]

        # Log detailed results
        logging.info(f"Analysis complete for {pdf_path}.")
        logging.info(f"Extracted Items: {text_fields['items']}")
        logging.info(f"Extracted Address: {text_fields['address']}")
        logging.info(f"Extracted Company Name: {text_fields['company_name']}")
        logging.info(f"Extracted Tables: {structured_tables}")

        return {
            "text_fields": text_fields,
            "tables": structured_tables
        }

# Detect the encoding of the CSV file
def detect_encoding(file_path):
    with open(file_path, "rb") as file:
        result = detect(file.read())
        return result["encoding"]

# Load product data from the CSV with detected encoding
csv_path = r"C:\\Users\\LENOVO\\Downloads\\articled.csv"
encoding = detect_encoding(csv_path)
product_table = pd.read_csv(csv_path, encoding=encoding, delimiter=";")

# Initialize the extractor with product data
extractor = PDFDataExtractor(product_table)

# Specify the PDF paths
pdf_paths = [
    r"C:\Users\LENOVO\Downloads\demande_devis_with_company_info1.pdf"
]

# Analyze each PDF
for pdf_path in pdf_paths:
    result = extractor.analyze_pdf(pdf_path)
    logging.info(f"Extracted Text Fields for {pdf_path}:")
    logging.info(result["text_fields"])
    logging.info(f"\nExtracted Tables for {pdf_path}:")
    logging.info(result["tables"])
