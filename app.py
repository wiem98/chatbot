from flask import Flask, request, jsonify, render_template, session, redirect, url_for, flash
from flask_session import Session
import json
import psycopg2
import re
from jinja2 import Environment, FileSystemLoader
import pdfkit
import joblib
import logging
from sklearn.impute import SimpleImputer
import base64
from datetime import datetime, timedelta  # Fixed import
import os
from datetime import datetime
from io import BytesIO
import matplotlib.pyplot as plt
from responses import get_variable_response
from pdf2image import convert_from_path
from flask_mail import Message
from mail_config import init_mail
import base64
from pdf2image import convert_from_path
from PIL import ImageEnhance


app = Flask(__name__)
app.secret_key = 'wiem3551'
mail = init_mail(app)

app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
Session(app)


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
try:
    conn = psycopg2.connect(
        dbname="chatbot_db",
        user="postgres",
        password="wiem3551",
        host="localhost",
        port="5432"
    )
    cursor = conn.cursor()
except Exception as e:
    raise RuntimeError(f"Database connection failed: {e}")

# Ensure columns are added if they do not already exist
cursor.execute('''
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='informations' AND column_name='quantite') THEN
        ALTER TABLE informations ADD COLUMN quantite INTEGER;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='informations' AND column_name='adresse_entreprise') THEN
        ALTER TABLE informations ADD COLUMN adresse_entreprise TEXT;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='informations' AND column_name='n_tva') THEN
        ALTER TABLE informations ADD COLUMN n_tva VARCHAR(50);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='informations' AND column_name='total_produit') THEN
        ALTER TABLE informations ADD COLUMN total_produit NUMERIC(10, 2);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='informations' AND column_name='status') THEN
        ALTER TABLE informations ADD COLUMN status VARCHAR(50);
    END IF;
END $$;
''')
conn.commit()


def normalize_input(user_input):
    """
    Normalize user input to handle slight variations in natural language using a lexicon.
    """
    # Predefined lexicon mapping phrases to normalized commands
    lexicon = {
        r"(je veux|je souhaite|svp|veuillez|j'ai)": "", 
        r"(le nom est|mon nom est|on est)": "set_name",  
        r"(créer|nouveau|initier)": "create",  
        r"(adresse|l'adresse est)": "set_address",  
    }
    
    # Lowercase and trim input
    user_input = user_input.lower().strip()
    
    # Apply replacements from the lexicon
    for pattern, replacement in lexicon.items():
        user_input = re.sub(pattern, replacement, user_input)
    
    # Remove extra spaces
    user_input = re.sub(r"\s+", " ", user_input).strip()
    return user_input

MODEL_PATH = r'C:\\Users\\LENOVO\\new\\models\\gradient_boosting_model2.pkl'
loaded_model = joblib.load(MODEL_PATH)
print(loaded_model.feature_names_in_)

@app.after_request
def add_headers(response):
    response.headers['Cache-Control'] = 'no-store'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/')
def home():
    session.clear()
    logging.info("Session cleared: Restarting steps.")
    return render_template('index.html')
def log_status_change(client_name, old_status, new_status, reason):
    """
    Logs the status change of a client into the status_history table.
    """
    try:
        cursor.execute(
            """
            INSERT INTO status_history (client_name, old_status, new_status, reason, change_date)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (client_name, old_status, new_status, reason, datetime.now())
        )
        conn.commit()
        logging.info(f"Status change logged for client '{client_name}': {old_status} -> {new_status}")
    except Exception as e:
        logging.error(f"Failed to log status change for client '{client_name}': {e}", exc_info=True)
MODEL_PATH = r'C:\\Users\\LENOVO\\new\\models\\client_status_model_enhanced.pkl'
try:
    status_model = joblib.load(MODEL_PATH)
    logging.info("Status prediction model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load the status prediction model: {e}")
    raise RuntimeError("Model loading failed.")
# Helper functions
def predict_status(features):
    """
    Predicts the client status using the pre-trained RandomForest model while enforcing specific rules:
    - Allow transitions from 'nouveau' to 'normal'
    - Allow transitions from 'normal' to 'VIP'
    - Allow downgrades from 'VIP' to 'normal'
    """
    try:
        # Ensure feature format matches the model input
        feature_vector = [
            features['total_spending'],
            features['total_quantity'],
            features['recency'],
            features['predicted_spending'],
            features['avg_quantity'],
            features['purchase_frequency'],
            features['average_spending'],
            features['churn_indicator'],
            features['total_spending_last_3_months'],
            features['large_purchase'],
            features['frequent_low_selling_purchase']
        ]
        
        # Predict the new status
        prediction = status_model.predict([feature_vector])[0]
        status_mapping = {0: 'nouveau', 1: 'normal', 2: 'VIP'}
        predicted_status = status_mapping[prediction]

        # Infer rules directly from features
        # Example logic: Infer current status based on spending or other features
        if features['total_spending'] < 100:
            current_status = 'nouveau'
        elif features['total_spending'] < 1000:
            current_status = 'normal'
        else:
            current_status = 'VIP'

        # Apply transition rules
        if current_status == 'nouveau' and predicted_status == 'normal':
            return 'normal'  # Upgrade from 'nouveau' to 'normal'
        elif current_status == 'normal' and predicted_status == 'VIP':
            return 'VIP'  # Upgrade from 'normal' to 'VIP'
        elif current_status == 'VIP' and predicted_status == 'normal':
            return 'normal'  # Downgrade from 'VIP' to 'normal'
        else:
            # If the transition doesn't match allowed rules, return the inferred status
            return current_status

    except Exception as e:
        logging.error(f"Status prediction failed: {e}")
        return "unknown"  # Return "unknown" in case of failure

def log_status_change(client_name, old_status, new_status, reason):
    """
    Logs the status change of a client into the status_history table.
    """
    try:
        cursor.execute(
            """
            INSERT INTO status_history (client_name, old_status, new_status, reason, change_date)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (client_name, old_status, new_status, reason, datetime.now())
        )
        conn.commit()
        cursor.execute(
            """
            UPDATE informations_client
            SET status = %s
            WHERE nom_client = %s
            """,
            (new_status, client_name)
        )
        conn.commit()

        logging.info(f"Status change logged for client '{client_name}': {old_status} -> {new_status}")
    except Exception as e:
        logging.error(f"Failed to log status change for client '{client_name}': {e}", exc_info=True)

def add_product(quantity, reference, session, cursor, conn):
    try:
        cursor.execute("SELECT designation, prix_achat FROM new_product WHERE ref ILIKE %s", (reference,))
        product = cursor.fetchone()

        if not product:
            return jsonify({"response": f"Produit introuvable. Référence '{reference}' non trouvée."})

        designation, prix_achat = product
        total_produit = round(int(quantity) * float(prix_achat), 2)

        existing_product = next(
            (prod for prod in session['data']['products'] if prod['reference'] == reference), None
        )

        if existing_product:
            existing_product['quantite'] += int(quantity)
            existing_product['total_produit'] = round(existing_product['quantite'] * existing_product['prix_unitaire'], 2)

            cursor.execute(
                """
                UPDATE informations
                SET quantite = %s, total_produit = %s
                WHERE nom_entreprise = %s AND ref_produit = %s
                """,
                (existing_product['quantite'], existing_product['total_produit'], session['data']['nom_entreprise'], reference)
            )
        else:
            session['data']['products'].append({
                "reference": reference,
                "description": designation,
                "quantite": int(quantity),
                "prix_unitaire": float(prix_achat),
                "total_produit": total_produit
            })

            cursor.execute(
                """
                INSERT INTO informations (
                    nom_entreprise, nom_user, produit, quantite, 
                    total_produit, ref_produit, prix_initial, date
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    session['data']['nom_entreprise'],
                    session['data']['nom_user'],
                    designation,
                    quantity,
                    total_produit,
                    reference,
                    prix_achat,
                    datetime.now()
                )
            )

        conn.commit()
        session.modified = True
        pdf_data, preview_image, pdf_path = generate_quote_base64(
            session['data']['nom_entreprise'], session['data']['products'], 
            archive_folder="C:\\Users\\LENOVO\\new\\archives_devis"
        )
        session['data']['pdf_path'] = pdf_path

        return jsonify({
            "response": f"Produit '{designation}' ajouté avec succès !",
            "preview_image": f"data:image/jpeg;base64,{preview_image}",
            "pdf_data": pdf_data,
            "pdf_filename": "devis_rempli.pdf"
        })

    except Exception as e:
        conn.rollback()
        logging.error(f"Error adding product: {e}", exc_info=True)
        return jsonify({"response": "Erreur lors de l'ajout du produit."})

def update_product(quantity, reference, session, cursor, conn):
    try:
        product_found = False
        for product in session['data']['products']:
            if product['reference'] == reference:
                product_found = True
                product['quantite'] = int(quantity)
                product['total_produit'] = round(int(quantity) * product['prix_unitaire'], 2)

                cursor.execute(
                    """
                    UPDATE informations
                    SET quantite = %s, total_produit = %s
                    WHERE nom_entreprise = %s AND ref_produit = %s
                    """,
                    (quantity, product['total_produit'], session['data']['nom_entreprise'], reference)
                )
                conn.commit()

                pdf_data, preview_image, pdf_path = generate_quote_base64(
                    session['data']['nom_entreprise'], session['data']['products'], 
                    archive_folder="C:\\Users\\LENOVO\\new\\archives_devis"
                )
                session['data']['pdf_path'] = pdf_path

                return jsonify({
                    "response": f"Produit '{product['description']}' modifié avec succès !",
                    "preview_image": f"data:image/jpeg;base64,{preview_image}",
                    "pdf_data": pdf_data,
                    "pdf_filename": "devis_rempli.pdf"
                })

        if not product_found:
            return jsonify({"response": f"Produit '{reference}' non trouvé."})

    except Exception as e:
        conn.rollback()
        logging.error(f"Error updating product: {e}", exc_info=True)
        return jsonify({"response": "Erreur lors de la modification du produit."})

def remove_product(reference, session, cursor, conn):
    try:
        if len(session['data']['products']) <= 1:
            return jsonify({"response": "Un devis ne peut pas être vide. Ajoutez un produit avant d'en supprimer un."})

        session['data']['products'] = [
            product for product in session['data']['products']
            if product['reference'] != reference
        ]
        session.modified = True

        cursor.execute(
            """
            DELETE FROM informations
            WHERE nom_entreprise = %s AND ref_produit = %s
            """,
            (session['data']['nom_entreprise'], reference)
        )
        conn.commit()

        pdf_data, preview_image, pdf_path = generate_quote_base64(
            session['data']['nom_entreprise'], session['data']['products'], 
            archive_folder="C:\\Users\\LENOVO\\new\\archives_devis"
        )
        session['data']['pdf_path'] = pdf_path

        return jsonify({
            "response": f"Produit '{reference}' supprimé avec succès.",
            "preview_image": f"data:image/jpeg;base64,{preview_image}",
            "pdf_data": pdf_data,
            "pdf_filename": "devis_rempli.pdf"
        })

    except Exception as e:
        conn.rollback()
        logging.error(f"Error removing product: {e}", exc_info=True)
        return jsonify({"response": "Erreur lors de la suppression du produit."})
def detect_action(user_input):
    user_input = user_input.lower().strip()  # Normalize input
    match = re.match(r"(ajouter|modifier|supprimer|envoyer)\s*(.*)", user_input)
    if match:
        action, params = match.groups()
        return action, params
    return None, None
@app.route('/predict_price', methods=['POST'])
def predict_price():
    try:
        data = request.json
        features = [
            data['prix_achat'], data['quantite'], data['stock'],
            data['year'], data['month'], data.get('markup', 0)
        ]
        predicted_price = loaded_model.predict([features])[0]
        return jsonify({"predicted_price": round(predicted_price, 2)})
    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        return jsonify({"error": "Failed to predict price."}), 500


def extract_information(user_input):
    """
    Extracts company name, address, VAT number, email, and user name from unordered input.
    """
    # Normalize input (remove extra spaces)
    user_input = user_input.strip()

    # Fix "cest" to "c'est" for better email detection
    user_input = re.sub(r"\bcest\b", "c'est", user_input, flags=re.IGNORECASE)

    print(f"Debugging Input Before Extraction: {user_input}")  # **Add this line to print processed input**

    # Define refined regex patterns
    patterns = {
        # User Name (Capture the actual name, ignoring "de la société")
        "nom_user": r"(?:je suis|moi c'est|mon nom est|je m'appelle|on m'appelle|on me connaît sous le nom de|le gérant|la gérante|le responsable|la responsable|directeur|directrice|responsable des ventes|responsable commercial|chef d'entreprise|PDG|propriétaire|dirigeant|chef de projet|manager|co-fondateur|fondateur|partenaire|consultant|associé|entrepreneur|freelance|auto-entrepreneur|gestionnaire|administrateur|secrétaire général|président|vice-président|vendeur|commerçant|artisan|formateur|représentant|développeur|designer|marketeur|chargé de mission|expert|coach|professeur|avocat|ingénieur|médecin|notaire|courtier|agent immobilier|recruteur|indépendant|travailleur indépendant)\s+([\w-]+(?:\s[\w-]+)?)",

        # Company Name (Ensure "de la société" is properly handled)
        "nom_entreprise": r"(?:de la société|de la societe|de l'entreprise|nom de la société|nom de la societe|de la part de la societe|notre entreprise|notre société|la société s'appelle|entreprise|société|groupe|start-up|marque|cabinet|boutique|commerce|organisme|association|bureau|franchise)\s+([\w-]+)",

        # Address (Ensure it captures after "situé à" or similar phrases)
        "adresse_entreprise": r"(?:\b(situ[ée]?|basé[ée]?|localisé[ée]?|adresse|sise à|dans|notre adresse|l'adresse|se trouve à|installé à|implanté à|nous sommes à|nous sommes situés à|notre siège est à|notre siège social est à|se situe à|domicilié à|établi à)\b)\s+([\w\s,-]+?(?=\s*(notre\s|email|téléphone|$)))",

        # VAT Number (Handles "numéro de TVA")
        "n_tva": r"\b(FR\d{11})\b",

        # Email (Fixes extraction of "set_address e-mail")
        "email": r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)"
    }


    extracted_data = {
        "nom_entreprise": None,
        "adresse_entreprise": None,
        "n_tva": None,
        "email": None,
        "nom_user": None
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            extracted_data[key] = match.group(1).strip()
            print(f"Extracted {key}: {extracted_data[key]}")  # **Debugging each extracted field**

    return extracted_data

def standardize_user_input(user_input):
    """
    Standardizes user input by replacing variations of company-related phrases
    with standardized terms for better regex extraction.
    """
    # Convert input to lowercase for uniform processing
    user_input = user_input.lower()
    greetings = [
        r"\bbonjour\b", r"\bsalut\b", r"\bbonsoir\b", r"\bcoucou\b", r"\byo\b", r"\bhey\b",
        r"\bhello\b", r"\bhi\b", r"\bcher\b", r"\bchère\b", r"\bmonsieur\b", r"\bmadame\b",
        r"\bmesdames et messieurs\b", r"\bme\b", r"\bm\b", r"\bmlle\b", r"\bmme\b", r"\bdistingué\b",
        r"\bcordialement\b", r"\bavec mes salutations\b", r"\bmes respects\b"
    ]

    # Remove greetings from the beginning of the sentence
    user_input = re.sub(r"^(" + "|".join(greetings) + r")[,!\s]*", "", user_input, flags=re.IGNORECASE)

    # Define common word variations to standardize
    replacements = {
        # Standardizing Email Mentions
        r"\bnotre email est\b": "email:",
    r"\bnous contacter à\b": "email:",
    r"\bcontactez-nous sur\b": "email:",
    r"\bnos coordonnées email sont\b": "email:",
    r"\bnotre adresse email est\b": "email:",
    r"\badresse électronique\b": "email:",
    r"\bemail de contact\b": "email:",
    r"\bemail professionnel\b": "email:",
    r"\bemail officiel\b": "email:",
    r"\bnous écrire à\b": "email:",
    r"\bjoignable par email à\b": "email:",
    r"\bnous envoyer un email à\b": "email:",
    r"\bpour nous contacter, utilisez\b": "email:",
    r"\bnous joindre à\b": "email:",
    r"\bemail direct\b": "email:",
    r"\bcontact direct par email\b": "email:",
    r"\badresse email de l'entreprise\b": "email:",
    r"\bemail du service client\b": "email:",
    r"\bemail pour réclamation\b": "email:",
    r"\bemail du support\b": "email:",
    r"\bemail administratif\b": "email:",

    # **Standardizing VAT Mentions (More Synonyms)**
    r"\bnotre tva\b": "numéro de TVA",
    r"\bnotre numéro de tva est\b": "numéro de TVA:",
    r"\bdont notre tva est\b": "numéro de TVA:",
    r"\bdont notre numéro de tva\b": "numéro de TVA:",
    r"\bla valeur de tva est\b": "numéro de TVA:",
    r"\bla valeur du code tva\b": "numéro de TVA:",
    r"\bnous sommes enregistrés sous le code tva\b": "numéro de TVA:",
    r"\bnuméro de taxe\b": "numéro de TVA:",
    r"\bnous sommes enregistrés sous le numéro de TVA\b": "numéro de TVA:",
    r"\bTVA enregistrée sous\b": "numéro de TVA:",
    r"\bTVA de notre entreprise\b": "numéro de TVA:",
    r"\bnous déclarons la TVA sous\b": "numéro de TVA:",
    r"\bTVA facturée sous\b": "numéro de TVA:",
    r"\bTVA appliquée sur\b": "numéro de TVA:",
    r"\bnuméro fiscal de TVA\b": "numéro de TVA:",
    r"\bTVA attribuée à\b": "numéro de TVA:",
    r"\bnuméro d'enregistrement TVA\b": "numéro de TVA:",
    r"\bTVA entreprise\b": "numéro de TVA:",
    r"\bcode TVA\b": "numéro de TVA:",
    r"\bTVA identifiée sous\b": "numéro de TVA:",
    r"\bnous collectons la TVA sous\b": "numéro de TVA:",
    r"\bnotre entreprise est assujettie à la TVA sous\b": "numéro de TVA:",
    r"\bTVA sous le numéro\b": "numéro de TVA:",
    r"\bTVA officielle\b": "numéro de TVA:",
    r"\bTVA en vigueur\b": "numéro de TVA:",
    r"\bTVA enregistrée\b": "numéro de TVA:",
    r"\bTVA active\b": "numéro de TVA:",     
    }

    # Apply replacements
    for pattern, replacement in replacements.items():
        user_input = re.sub(pattern, replacement, user_input, flags=re.IGNORECASE)

    # Ensure "nom utilisateur:" and "nom de la société:" are not overwritten incorrectly
    user_input = re.sub(r"nom utilisateur:\s*de\s*", "nom de la société: ", user_input)

    return user_input.strip()
@app.route('/chat', methods=['POST'])
def chat():
    try:
        if 'step' not in session:
            session['step'] = 0
            session['data'] = {'products': [], 'is_partner': False}
        
        step = session['step']
        data = request.json if request.is_json else None
        user_input = data.get('message', "").strip() if data else ""
        user_input = normalize_input(user_input)


        if step == 0:
            session['step'] = 1
            return jsonify({"response": get_variable_response("welcome")})

        elif step == 1:
            session['step'] = 2
            return jsonify({"response": get_variable_response("ask_company_name")})

        elif step == 2:
            logging.info("Step 2: Parsing user input for mixed information.")
            logging.info(f"Raw user input: {user_input}")

            # 1. Extract the user input for mixed information
            standardized_input = standardize_user_input(user_input)
            logging.info(f"Standardized user input: {standardized_input}")

            # 2. Extract the user input for mixed information
            extracted_data = extract_information(standardized_input)            
            logging.info(f"Extracted data from user input: {extracted_data}")

            # Update session with extracted data
            session['data']['nom_entreprise'] = extracted_data['nom_entreprise'] or session['data'].get('nom_entreprise')
            session['data']['adresse_entreprise'] = extracted_data['adresse_entreprise'] or session['data'].get('adresse_entreprise', "Non spécifié")
            session['data']['n_tva'] = extracted_data['n_tva'] or session['data'].get('n_tva', "Non spécifié")
            session['data']['email'] = extracted_data['email'] or session['data'].get('email', "Non spécifié")
            session['data']['nom_user'] = extracted_data['nom_user'] or session['data'].get('nom_user', "Non spécifié")

            logging.info(f"Updated session data: {session['data']}")

            # 2. Check if the company already exists in the database
            if session['data']['nom_entreprise']:
                logging.info(f"Checking if company '{session['data']['nom_entreprise']}' exists in the database.")
                
                cursor.execute("""
                    SELECT 
                        i.adresse_entreprise, 
                        i.n_tva, 
                        ic.email 
                    FROM informations AS i
                    LEFT JOIN informations_client AS ic 
                    ON i.nom_entreprise = ic.nom_client
                    WHERE i.nom_entreprise = %s
                """, (session['data']['nom_entreprise'],))
                
                company_data = cursor.fetchone()
                logging.info(f"Database result for company '{session['data']['nom_entreprise']}': {company_data}")

                if company_data:
                    # Company exists, update session with DB data
                    session['data']['adresse_entreprise'] = company_data[0] or session['data']['adresse_entreprise']
                    session['data']['n_tva'] = company_data[1] or session['data']['n_tva']
                    session['data']['email'] = company_data[2] or session['data']['email']
                    logging.info("Company found in database. Updated session data with database values.")
                    logging.info(f"Final session data: {session['data']}")

                    # 🚀 **Check if the user name is provided**
                    if not session['data']['nom_user'] or session['data']['nom_user'] == "Non spécifié":
                        logging.info("User name missing. Redirecting to Step 6 to collect user name.")
                        session['step'] = 6
                        return jsonify({"response": get_variable_response("ask_user_name")})
                    
                    # 🚀 **If both company and user name are available, skip to Step 7**
                    logging.info("Company and user name detected. Proceeding directly to Step 7.")
                    session['step'] = 7
                    return jsonify({"response": get_variable_response("company_and_user_detected")})

            # 3. Check if required information is missing
            missing_fields = []
            if not session['data']['adresse_entreprise'] or session['data']['adresse_entreprise'] == "Non spécifié":
                missing_fields.append("adresse de l'entreprise")
            if not session['data']['n_tva'] or session['data']['n_tva'] == "Non spécifié":
                missing_fields.append("numéro de TVA")
            if not session['data']['email'] or session['data']['email'] == "Non spécifié":
                missing_fields.append("adresse email")
            
            if not session['data']['nom_user'] or session['data']['nom_user'] == "Non spécifié":
                missing_fields.append("nom de l'utilisateur")

            logging.info(f"Missing fields: {missing_fields}")

            # 4. Redirect to the correct step based on missing fields
            # 4. Ask for the next missing field (if any)
            if missing_fields:
                next_missing_field = missing_fields[0]  # Pick the first missing field
                
                if next_missing_field == "adresse de l'entreprise":
                    logging.info("Redirecting to Step 3 to collect the address.")
                    session['step'] = 3
                    return jsonify({"response": get_variable_response("ask_company_address")})
                elif next_missing_field == "numéro de TVA":
                    logging.info("Redirecting to Step 4 to collect the VAT number.")
                    session['step'] = 4
                    return jsonify({"response": get_variable_response("ask_vat_number")})
                elif next_missing_field == "adresse email":
                    logging.info("Redirecting to Step 5 to collect the email.")
                    session['step'] = 5
                    return jsonify({"response": get_variable_response("ask_email")})
                elif next_missing_field == "nom de l'utilisateur":
                    logging.info("Redirecting to Step 6 to collect the user's name.")
                    session['step'] = 6
                    return jsonify({"response": get_variable_response("ask_user_name")})

            # If no missing fields, proceed to Step 7 directly
            logging.info("All fields provided. Proceeding to Step 7.")
            session['step'] = 7
            return jsonify({"response": get_variable_response("ask_products")})

        elif step == 3:  # Collecting address
            session['data']['adresse_entreprise'] = user_input

            # Check if VAT, email, or user name are still missing
            missing_fields = []
            if not session['data']['n_tva'] or session['data']['n_tva'] == "Non spécifié":
                missing_fields.append("numéro de TVA")
            if not session['data']['email'] or session['data']['email'] == "Non spécifié":
                missing_fields.append("adresse email")
            if not session['data']['nom_user'] or session['data']['nom_user'] == "Non spécifié":
                missing_fields.append("nom de l'utilisateur")

            if missing_fields:
                next_missing_field = missing_fields[0]  # Go to the next required step
                if next_missing_field == "numéro de TVA":
                    session['step'] = 4
                    return jsonify({"response": get_variable_response("ask_vat_number")})
                elif next_missing_field == "adresse email":
                    session['step'] = 5
                    return jsonify({"response": get_variable_response("ask_email")})
                elif next_missing_field == "nom de l'utilisateur":
                    session['step'] = 6
                    return jsonify({"response": get_variable_response("ask_user_name")})

            # If no fields are missing, proceed directly to Step 7
            session['step'] = 7
            return jsonify({"response": get_variable_response("ask_products")})


        elif step == 4:  # Collecting VAT number
            user_input = user_input.strip().upper()  # Ensure input is cleaned and uppercase
            if not re.match(r"^FR\d{11}$", user_input):
                return jsonify({"response": get_variable_response("invalid_vat")})
            
            session['data']['n_tva'] = user_input

            # Check what is still missing
            missing_fields = []
            if not session['data']['email'] or session['data']['email'] == "Non spécifié":
                missing_fields.append("adresse email")
            if not session['data']['nom_user'] or session['data']['nom_user'] == "Non spécifié":
                missing_fields.append("nom de l'utilisateur")

            if missing_fields:
                next_missing_field = missing_fields[0]  # Go to the next required step
                if next_missing_field == "adresse email":
                    session['step'] = 5
                    return jsonify({"response": get_variable_response("ask_email")})
                elif next_missing_field == "nom de l'utilisateur":
                    session['step'] = 6
                    return jsonify({"response": get_variable_response("ask_user_name")})

            session['step'] = 7
            return jsonify({"response": get_variable_response("ask_products")})


        elif step == 5:  # Collecting email
            if not re.match(r"[^@]+@[^@]+\.[^@]+", user_input):
                return jsonify({"response": get_variable_response("invalid_email")})
            
            session['data']['email'] = user_input

            # Check if user name is still missing
            if not session['data']['nom_user'] or session['data']['nom_user'] == "Non spécifié":
                session['step'] = 6
                return jsonify({"response": get_variable_response("ask_user_name")})

            session['step'] = 7
            return jsonify({"response": get_variable_response("ask_products")})


        elif step == 6:  # Collecting user name
            session['data']['nom_user'] = user_input

            session['step'] = 7
            return jsonify({"response": get_variable_response("ask_products")})

        elif step == 7:
            user_input_lower = user_input.lower().strip()
            if re.search(r"(commander|acheter|demander|besoin de|recherche|chercher)", user_input_lower):
                logging.debug(f"User input received: {user_input}")

                # Extract keywords, allowing optional "un, une, etc."
                keyword_matches = re.findall(r"(?:commander|acheter|demander|besoin de|recherche|chercher)(?:\s+(?:un|une|des|du|le|la|les)?)?\s+([\w-]+)", user_input_lower)

                # Extract extra items after "et" (e.g., "écrou et vis")
                extra_keywords = re.findall(r"(?:et\s+)([\w-]+)", user_input_lower)
                keyword_matches.extend(extra_keywords)

                logging.debug(f"Keywords extracted: {keyword_matches}")

                if keyword_matches:
                    try:
                        product_list = []

                        for keyword in keyword_matches:
                            logging.debug(f"Executing database query with keyword: {keyword}")
                            query = """
                                SELECT ref, designation 
                                FROM new_product 
                                WHERE designation ILIKE %s
                            """
                            cursor.execute(query, (f"{keyword}%",))
                            results = cursor.fetchall()
                            logging.debug(f"Query results for '{keyword}': {results}")

                            for row in results:
                                product_list.append({"ref": row[0], "designation": row[1]})

                        if not product_list:
                            logging.warning(f"No products found for keywords: {keyword_matches}")
                            return jsonify({"response": "Aucun produit correspondant trouvé pour votre recherche."})

                        logging.info(f"Products found: {product_list}")
                        return jsonify({
                            "response": "Voici une liste des produits correspondant à votre recherche :",
                            "product_list": product_list
                        })

                    except Exception as e:
                        logging.error(f"Error during product search: {e}", exc_info=True)
                        return jsonify({"response": "Une erreur est survenue lors de la recherche des produits."})
                else:
                    logging.warning(f"Keyword extraction failed for user input: {user_input}")
                    return jsonify({"response": "Je n'ai pas compris les produits que vous souhaitez commander. Veuillez préciser."})

            matches = re.findall(r"(\d+)\s+([A-Za-z0-9-/]+)", user_input.strip())
            errors = []
            success_count = 0

            if matches:
                for quantity, reference in matches:
                    try:
                        cursor.execute("SELECT designation, prix_achat, stock FROM new_product WHERE ref ILIKE %s", (reference,))
                        product = cursor.fetchone()

                        if not product:
                            logging.warning(f"Produit introuvable: Référence '{reference}' non trouvée.")
                            errors.append(f"Produit introuvable. Référence '{reference}' non trouvée.")
                            continue

                        designation, prix_achat, stock = product
                        quantity = int(quantity)
                        current_date = datetime.now()
                        features = [
                            prix_achat,              
                            quantity,                
                            stock,
                            current_date.year,       
                            current_date.month,     
                            0.1                      
                        ]

                        # Predict the selling price
                        predicted_price = loaded_model.predict([features])[0]
                        total_produit = round(quantity * predicted_price, 2)

                        # Check if the product is already in the session's product list
                        existing_product = next(
                            (prod for prod in session['data']['products'] if prod['reference'] == reference), None
                        )

                        if existing_product:
                            existing_product['quantite'] += int(quantity)
                            existing_product['total_produit'] = round(existing_product['quantite'] * existing_product['prix_unitaire'], 2)
                        else:
                            session['data']['products'].append({
                                "reference": reference,
                                "description": designation,
                                "quantite": int(quantity),
                                "prix_unitaire": round(predicted_price, 2),
                                "total_produit": total_produit
                            })

                        success_count += 1
                        session.modified = True
                        logging.info(f"Produit '{designation}' ajouté ou mis à jour pour référence '{reference}'.")

                    except Exception as e:
                        conn.rollback()
                        logging.error(f"Error processing product '{reference}': {e}", exc_info=True)
                        errors.append(f"Erreur lors du traitement du produit '{reference}'.")

                else:
                    errors.append("Format invalide. Utilisez 'quantité référence'. Exemple : '1 ECIN-14 2 UNPA-1-4-1/4'.")


            if not session['data']['products']:
                return jsonify({"response": get_variable_response("no_products")})

            # Calculate order summary
            items = session['data']['products']
            base_ht = sum(item['total_produit'] for item in items)
            montant_taxes = round(base_ht * 0.2, 2)  
            net_a_payer = round(base_ht + montant_taxes, 2)

            session['data']['base_ht'] = base_ht
            session['data']['montant_taxes'] = montant_taxes
            session['data']['net_a_payer'] = net_a_payer

            try:
                for item in session['data']['products']:
                    cursor.execute(
                        """
                        SELECT COUNT(*) FROM informations
                        WHERE nom_entreprise = %s AND ref_produit = %s AND nom_user = %s
                        """,
                        (session['data']['nom_entreprise'], item['reference'], session['data']['nom_user'])
                    )
                    product_exists = cursor.fetchone()[0]

                    if product_exists:
                        query_update = """
                            UPDATE informations
                            SET quantite = %s, total_produit = %s, prix_ventes = %s
                            WHERE nom_entreprise = %s AND ref_produit = %s AND nom_user = %s
                        """
                        cursor.execute(
                            query_update,
                            (
                                item['quantite'], 
                                item['total_produit'],  
                                item['prix_unitaire'],  
                                session['data']['nom_entreprise'],
                                item['reference'],
                                session['data']['nom_user']
                            )
                        )
                    else:
                        query_insert = """
                            INSERT INTO informations (
                                nom_entreprise, nom_user, produit, quantite,
                                adresse_entreprise, n_tva, total_produit, 
                                total_commande, montant_taxes, net_a_payer,
                                ref_produit, prix_initial, date, prix_ventes, status
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """
                        values = (
                            session['data']['nom_entreprise'],
                            session['data']['nom_user'],
                            item['description'],  
                            item['quantite'],  
                            session['data']['adresse_entreprise'],
                            session['data']['n_tva'],
                            item['total_produit'],  # Quantity * prix_ventes
                            base_ht,  # Total command value 
                            montant_taxes,
                            net_a_payer,
                            item['reference'],  # Product reference
                            item.get('prix_achat', None),  # Initial price 
                            datetime.now(),  # Current date
                            item['prix_unitaire'],  # Selling price
                            "nouveau"  # Default status
                        )
                        cursor.execute(query_insert, values)

                # Handle client status
                client_name = session['data']['nom_entreprise']
                cursor.execute("SELECT status FROM informations_client WHERE nom_client = %s LIMIT 1", (client_name,))
                result = cursor.fetchone()
                old_status = result[0] if result else None

                if old_status is None:
                    new_status = 'nouveau'
                    cursor.execute(
                        """
                        INSERT INTO informations_client (nom_client, adresse, status, email)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (
                            client_name,
                            session['data']['adresse_entreprise'],
                            new_status,
                            session['data']['email']
                        )
                    )
                    log_status_change(client_name, old_status, new_status, "New client added")
                else:
                    features = {
                        'total_spending': base_ht,
                        'total_quantity': sum(item['quantite'] for item in items),
                        'recency': 1,  # Placeholder, calculate actual recency
                        'predicted_spending': 0,  # Placeholder, fetch from another source if available
                        'avg_quantity': base_ht / max(sum(item['quantite'] for item in items), 1),
                        'purchase_frequency': sum(item['quantite'] for item in items),
                        'average_spending': base_ht / max(sum(item['quantite'] for item in items), 1),
                        'churn_indicator': 0,  # Placeholder, calculate actual churn
                        'total_spending_last_3_months': base_ht,  # Placeholder for recent spending
                        'large_purchase': int(base_ht > 5000),
                        'frequent_low_selling_purchase': 0  # Placeholder, calculate if needed
                    }

                    new_status = predict_status(features)

                    if old_status != new_status:
                        log_status_change(client_name, old_status, new_status, "Order finalized")
                        cursor.execute(
                            """
                            UPDATE informations_client
                            SET status = %s
                            WHERE nom_client = %s
                            """,
                            (new_status, client_name)
                        )

                conn.commit()  # Commit all changes
                logging.info("Products and status stored successfully for this session.")

                # Generate the quote and save its path
                pdf_data, preview_image, pdf_path = generate_quote_base64(
                    session['data']['nom_entreprise'], items, archive_folder="C:\\Users\\LENOVO\\new\\archives_devis"
                )
                session['data']['pdf_path'] = pdf_path
                recipient_email = session['data']['email']
                email_sent = send_email(
                    recipient=recipient_email,
                    subject=f"Quote for {session['data']['nom_entreprise']}",
                    pdf_path=pdf_path,
                    client_name=session['data']['nom_entreprise']
                )

                session['step'] = 8
                if email_sent:
                    return jsonify({
                        "response": "Le devis a été généré et envoyé par e-mail avec succès.",
                        "preview_image": f"data:image/jpeg;base64,{preview_image}",
                        "pdf_data": pdf_data,
                        "pdf_filename": "devis_rempli.pdf"
                    })
                else:
                    return jsonify({"response": "The quote was generated, but the email could not be sent."})

            except Exception as e:
                conn.rollback()  # Rollback transaction in case of an error
                logging.error(f"Error saving data: {e}", exc_info=True)
                return jsonify({"response": "Une erreur est survenue lors de l'enregistrement des données."})


        elif step == 8:
            action, params = detect_action(user_input)

            if action == 'ajouter':
                match = re.match(r"(\d+)\s+([A-Za-z0-9-/]+)", params)
                if match:
                    quantity, reference = match.groups()
                    return add_product(quantity, reference, session, cursor, conn)
                else:
                    return jsonify({"response": "Format invalide. Utilisez 'ajouter <quantité> <référence>' pour ajouter un produit."})

            elif action == 'modifier':
                match = re.match(r"(\d+)\s+([A-Za-z0-9-/]+)", params)
                if match:
                    quantity, reference = match.groups()
                    return update_product(quantity, reference, session, cursor, conn)
                else:
                    return jsonify({"response": "Format invalide. Utilisez 'modifier <quantité> <référence>' pour modifier un produit."})

            elif action == 'supprimer':
                match = re.match(r"([A-Za-z0-9-/]+)", params)
                if match:
                    reference = match.group(1)
                    return remove_product(reference, session, cursor, conn)
                else:
                    return jsonify({"response": "Format invalide. Utilisez 'supprimer <référence>' pour supprimer un produit."})
            else:
                return jsonify({"response": "Action non reconnue. Utilisez 'ajouter', 'modifier', 'supprimer', ou 'envoyer'."})

        # Step 9
        elif step == 9:
            action, params = detect_action(user_input)

            if action == 'ajouter':
                match = re.match(r"(\d+)\s+([A-Za-z0-9-/]+)", params)
                if match:
                    quantity, reference = match.groups()
                    response = add_product(quantity, reference, session, cursor, conn)
                else:
                    return jsonify({"response": "Format invalide. Utilisez 'ajouter <quantité> <référence>' pour ajouter un produit."})

            elif action == 'modifier':
                match = re.match(r"(\d+)\s+([A-Za-z0-9-/]+)", params)
                if match:
                    quantity, reference = match.groups()
                    response = update_product(quantity, reference, session, cursor, conn)
                else:
                    return jsonify({"response": "Format invalide. Utilisez 'modifier <quantité> <référence>' pour modifier un produit."})

            elif action == 'supprimer':
                match = re.match(r"([A-Za-z0-9-/]+)", params)
                if match:
                    reference = match.group(1)
                    response = remove_product(reference, session, cursor, conn)
                else:
                    return jsonify({"response": "Format invalide. Utilisez 'supprimer <référence>' pour supprimer un produit."})

            else:
                return jsonify({"response": "Action non reconnue. Utilisez 'ajouter', 'modifier', ou 'supprimer'."})

            # After performing the action, regenerate the quote and send email
            try:
                items = session['data']['products']
                pdf_data, preview_image, pdf_path = generate_quote_base64(
                    session['data']['nom_entreprise'], items, archive_folder="C:\\Users\\LENOVO\\new\\archives_devis"
                )
                session['data']['pdf_path'] = pdf_path

                # Send the updated quote via email
                recipient_email = session['data']['email']
                email_sent = send_email(
                    recipient=recipient_email,
                    subject=f"Updated Quote for {session['data']['nom_entreprise']}",
                    pdf_path=pdf_path,
                    client_name=session['data']['nom_entreprise']
                )

                if email_sent:
                    return jsonify({
                        "response": response.get_json()['response'] + " Le devis mis à jour a été envoyé par e-mail.",
                        "preview_image": f"data:image/jpeg;base64,{preview_image}",
                        "pdf_data": pdf_data,
                        "pdf_filename": "updated_devis.pdf"
                    })
                else:
                    return jsonify({
                        "response": response.get_json()['response'] + " Échec de l'envoi du devis par e-mail.",
                        "preview_image": f"data:image/jpeg;base64,{preview_image}",
                        "pdf_data": pdf_data,
                        "pdf_filename": "updated_devis.pdf"
                    })

            except Exception as e:
                logging.error(f"Error generating updated quote or sending email: {e}", exc_info=True)
                return jsonify({"response": "Une erreur est survenue lors de la mise à jour et de l'envoi du devis."})

    except Exception as e:
        logging.error(f"Error in /chat endpoint: {e}", exc_info=True)
    return jsonify({"response": "Une erreur est survenue. Veuillez réessayer plus tard."})
def send_email(recipient, subject, pdf_path, client_name):
    try:
        # Get the filename from the path
        pdf_filename = os.path.basename(pdf_path)

        # Log the PDF path and filename
        logging.info(f"PDF path: {pdf_path}")
        logging.info(f"PDF filename: {pdf_filename}")

        # Load the email template
        env = Environment(loader=FileSystemLoader("templates"))
        template = env.get_template("email_template.html")
        html_content = template.render(client_name=client_name, pdf_filename=pdf_filename)

        # Log recipient email
        logging.info(f"Recipient email: {recipient}")

        # Read the PDF file
        with open(pdf_path, "rb") as pdf_file:
            pdf_data = pdf_file.read()

        # Log success of reading the PDF
        logging.info(f"PDF file read successfully: {pdf_filename}")

        # Create the email message
        msg = Message(
            subject=subject,
            recipients=[recipient],
            html=html_content,
        )

        msg.attach(pdf_filename, "application/pdf", pdf_data)

        logging.info(f"PDF attached to the email: {pdf_filename}")

        mail.send(msg)

        logging.info(f"Email with PDF attachment ({pdf_filename}) sent to {recipient}.")
        return True

    except Exception as e:
        # Log any errors
        logging.error(f"Failed to send email: {e}")
        return False

def load_counter():
    try:
        with open("quote_counter.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {"date": datetime.now().strftime('%Y-%m-%d'), "counter": 0}
    return data

def save_counter(data):
    with open("quote_counter.json", "w") as f:
        json.dump(data, f)

def get_quote_number():
    data = load_counter()
    current_date = datetime.now().strftime('%Y-%m-%d')

    if data["date"] != current_date:
        data = {"date": current_date, "counter": 1}
    else:
        data["counter"] += 1

    save_counter(data)

    return f'{data["counter"]:04d}'
def generate_quote_base64(client_name, items, archive_folder="C:\\Users\\LENOVO\\new\\archives_devis", output_file=None):
    try:
        env = Environment(loader=FileSystemLoader("templates"))
        template = env.get_template("quote_template.html")
        current_date = datetime.now().strftime('%d/%m/%Y')
        validation_date = (datetime.now() + timedelta(days=15)).strftime('%d/%m/%Y')
        quote_number = get_quote_number()

        html_content = template.render(
            client_name=client_name,
            client_address=session['data']['adresse_entreprise'],
            client_tva=session['data']['n_tva'],
            items=items,
            base_ht=session['data'].get('base_ht', 0),
            montant_taxes=session['data'].get('montant_taxes', 0),
            net_a_payer=session['data'].get('net_a_payer', 0),
            date=current_date,
            validation_date=validation_date,
            quote_number=quote_number

        )

        os.makedirs(archive_folder, exist_ok=True)

        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            sanitized_client_name = re.sub(r"[^\w\-_\.]", "_", client_name)
            output_file = os.path.join(archive_folder, f"devis_{sanitized_client_name}_{timestamp}.pdf")

        #  wkhtmltopdf configuration
        config = pdfkit.configuration(wkhtmltopdf=r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe')
        options = {
            'enable-local-file-access': '',
            'quiet': ''
        }

        pdfkit.from_string(html_content, output_file, options=options, configuration=config)

        images = convert_from_path(output_file, dpi=300, first_page=1, last_page=1)  # Increase DPI to 300 or higher
        image = images[0]

        image = ImageEnhance.Brightness(image).enhance(1.2)
        image = ImageEnhance.Contrast(image).enhance(1.5)
        image = ImageEnhance.Sharpness(image).enhance(2.0)

        preview_image_path = os.path.join(archive_folder, f"preview_{sanitized_client_name}_{timestamp}.jpg")
        image.save(preview_image_path, "JPEG", quality=95, dpi=(300, 300))  # Save with high quality and DPI

        # Encode the preview image to Base64 for response
        with open(preview_image_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode("utf-8")

        # Encode the PDF to Base64 for response
        with open(output_file, "rb") as pdf_file:
            base64_pdf = base64.b64encode(pdf_file.read()).decode("utf-8")

        return base64_pdf, base64_image, output_file  # Return the path of the saved PDF
    except Exception as e:
        logging.error(f"Error generating PDF or image: {e}", exc_info=True)
        raise RuntimeError(f"Failed to generate PDF or image: {e}")

if __name__ == '__main__':
    app.run(debug=False)
