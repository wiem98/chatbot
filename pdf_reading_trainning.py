from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import re
import os
import csv
import json
from transformers import pipeline
import spacy

# Path to Tesseract executable (update with your installation path)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load spaCy's NLP model
nlp = spacy.load("en_core_web_sm")  # Use "fr_core_news_sm" if the text is in French

# Load Hugging Face NER pipeline
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

# Function to convert PDF pages to images
def convert_pdf_to_images(pdf_path, output_folder="output_images"):
    poppler_path = r"C:\Users\LENOVO\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin"  # Update this path
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    images = convert_from_path(pdf_path, dpi=200, poppler_path=poppler_path)
    image_paths = []
    for i, image in enumerate(images):
        output_path = os.path.join(output_folder, f"page_{i + 1}.png")
        image.save(output_path, "PNG")
        image_paths.append(output_path)
    return image_paths

# Function to perform OCR on images
def perform_ocr(image_paths):
    extracted_text = []
    for image_path in image_paths:
        text = pytesseract.image_to_string(Image.open(image_path), lang="eng")
        print(f"\n--- Raw Text from Page ---\n{text}\n{'-'*80}")
        extracted_text.append(text)
    return extracted_text

# Function to load a product list from a file
def load_product_list(file_path, file_type="csv", encoding="utf-8"):
    product_list = []
    try:
        if file_type == "csv":
            with open(file_path, mode="r", encoding=encoding) as f:
                reader = csv.DictReader(f)
                product_list = [row for row in reader]
        elif file_type == "json":
            with open(file_path, mode="r", encoding=encoding) as f:
                product_list = json.load(f)
    except UnicodeDecodeError as e:
        print(f"Error reading file {file_path}: {e}")
        print("Try specifying a different encoding, such as 'ISO-8859-1'.")
        raise
    return product_list

# Function to recommend products based on constraints
def recommend_products(extracted_text, product_list):
    recommendations = []
    keywords = {
        "pressure": r"pression maximale de (\d+)\s*bars",
        "durability": r"duree de vie minimale de (\d+)\s*ans",
        "certification": r"certificat[s]? de conformite"
    }

    # Extract requirements from text
    constraints = {}
    for key, pattern in keywords.items():
        match = re.search(pattern, extracted_text, re.IGNORECASE)
        if match:
            constraints[key] = match.group(1)

    # Match products based on constraints
    for product in product_list:
        designation = product.get("designation", "").lower()
        reference = product.get("ref", "").lower()

        if any(word in extracted_text.lower() for word in [designation, reference]):
            recommendations.append(product)

    return recommendations, constraints

# Function to perform Named Entity Recognition (NER)
def extract_entities_with_ner(text):
    entities = ner_pipeline(text)
    extracted_info = {"enterprise_name": None, "address": None, "ntva": None}

    for entity in entities:
        if entity["entity"] == "ORG" and not extracted_info["enterprise_name"]:
            extracted_info["enterprise_name"] = entity["word"]
        elif entity["entity"] == "LOC" and not extracted_info["address"]:
            extracted_info["address"] = entity["word"]
        elif entity["entity"] == "MISC" and "FR" in entity["word"]:
            extracted_info["ntva"] = entity["word"]

    return extracted_info

# Main function
def main():
    pdf_path = r"C:\Users\LENOVO\Downloads\Cahier_de_Charges_Corrected_Final_IngeniusIT.pdf"
    product_file = r"C:\Users\LENOVO\Downloads\articled.csv"
    output_folder = "output_images"

    # Load product list
    product_list = load_product_list(product_file, file_type="csv", encoding="ISO-8859-1")

    # Step 1: Convert PDF to images
    print("Converting PDF to images...")
    image_paths = convert_pdf_to_images(pdf_path, output_folder)
    print(f"PDF converted to {len(image_paths)} images.")

    # Step 2: Perform OCR on images
    print("Performing OCR to extract text...")
    extracted_texts = perform_ocr(image_paths)

    # Step 3: Analyze extracted text and recommend products
    all_recommendations = []
    for page_num, text in enumerate(extracted_texts, start=1):
        print(f"\n--- Page {page_num} ---")
        ner_info = extract_entities_with_ner(text)
        print(f"Enterprise Name: {ner_info['enterprise_name']}")
        print(f"Address: {ner_info['address']}")
        print(f"NTVA: {ner_info['ntva']}")

        # Recommend products based on text
        recommendations, constraints = recommend_products(text, product_list)
        print(f"Constraints Found: {constraints}")
        print("Recommended Products:")
        for product in recommendations:
            print(f"  - {product['designation']} (Reference: {product['ref']})")
        all_recommendations.append({"page": page_num, "constraints": constraints, "recommendations": recommendations})

if __name__ == "__main__":
    main()
