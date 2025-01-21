from flask import Flask, request, jsonify, render_template, session, redirect, url_for, flash
from flask_session import Session
import json
import psycopg2
import re
from jinja2 import Environment, FileSystemLoader
import pdfkit
import pandas as pd
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
from PIL import ImageEnhance, ImageDraw, ImageFont, ImageOps, ImageFilter


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
        r"(je veux|je souhaite|svp|veuillez)": "",  # Remove polite filler phrases
        r"(le nom est|mon nom est)": "set_name",  # Normalize name setting
        r"(créer|nouveau|initier)": "create",  # Normalize creation commands
        r"(adresse|l'adresse est)": "set_address",  # Normalize address commands
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

def predict_status(features):
    """
    Predicts the client status using the pre-trained RandomForest model.
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
        prediction = status_model.predict([feature_vector])[0]
        status_mapping = {0: 'nouveau', 1: 'normal', 2: 'VIP'}
        return status_mapping[prediction]
    except Exception as e:
        logging.error(f"Status prediction failed: {e}")
        return "unknown"
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
            session['data']['nom_entreprise'] = user_input

            # Fetch company details from both tables
            cursor.execute("""
                SELECT 
                    i.adresse_entreprise, 
                    i.n_tva, 
                    ic.email 
                FROM informations AS i
                LEFT JOIN informations_client AS ic 
                ON i.nom_entreprise = ic.nom_client
                WHERE i.nom_entreprise = %s
            """, (user_input,))
            company_data = cursor.fetchone()

            if company_data:
                # Company exists, unpack the data
                session['data']['adresse_entreprise'] = company_data[0] or "Non spécifié"
                session['data']['n_tva'] = company_data[1] or "Non spécifié"
                session['data']['email'] = company_data[2] or "Non spécifié"

                # Skip steps and go directly to asking for the user name
                session['step'] = 6
                return jsonify({"response": "La société existe déjà. Passons directement à votre nom d'utilisateur."})
            else:
                # Company does not exist, proceed to the next step
                session['step'] = 3
                return jsonify({"response": get_variable_response("ask_company_address")})

        elif step == 3:
            session['data']['adresse_entreprise'] = user_input
            session['step'] = 4
            return jsonify({"response": get_variable_response("ask_vat_number")})

        elif step == 4:
            user_input = user_input.strip().upper()  # Ensure input is cleaned and uppercase
            if not re.match(r"^FR\d{11}$", user_input):
                return jsonify({"response": "Le numéro de TVA que vous avez saisi est invalide. Assurez-vous qu'il commence par 'FR' suivi de 11 chiffres. Exemple : 'FR12345678901'."})
            session['data']['n_tva'] = user_input
            session['step'] = 5
            return jsonify({"response": get_variable_response("ask_email")})

        elif step == 5:
            if not re.match(r"[^@]+@[^@]+\.[^@]+", user_input):
                return jsonify({"response": get_variable_response("invalid_email")})
            session['data']['email'] = user_input
            session['step'] = 6
            return jsonify({"response": get_variable_response("ask_user_name")})


        elif step == 6:
            session['data']['nom_user'] = user_input
            session['step'] = 7
            return jsonify({"response": get_variable_response("ask_products")})

        elif step == 7:
            user_input_lower = user_input.lower().strip()
            if re.search(r"(commander|acheter|demander|besoin|recherche)", user_input.lower()):
                # Log the user input
                logging.debug(f"User input received: {user_input}")

                # Extract the keyword (e.g., "ecrou")
                keyword_match = re.search(r"(?:commander|acheter|demander|besoin|recherche) (?:un|une) (\w+)", user_input.lower())
                if keyword_match:
                    keyword = keyword_match.group(1)
                    logging.debug(f"Keyword extracted: {keyword}")  # Log extracted keyword

                    # Query the database for products matching the keyword
                    try:
                        logging.debug(f"Executing database query with keyword: {keyword}")
                        query = """
                            SELECT ref, designation 
                            FROM new_product 
                            WHERE designation ILIKE %s
                        """
                        cursor.execute(query, (f"{keyword}%",))
                        results = cursor.fetchall()
                        logging.debug(f"Database query: {query}, Parameters: {f'{keyword}%'}")
                        logging.debug(f"Database query results: {results}")  # Log query results

                        if not results:
                            logging.warning(f"No products found for keyword: {keyword}")
                            return jsonify({
                                "response": f"Désolé, aucun produit trouvé pour '{keyword}'. "
                                            f"Veuillez essayer un autre mot-clé ou entrez une référence directement."
                            })

                        # Convert results into a structured product list
                        product_list = [{"ref": row[0], "designation": row[1]} for row in results]
                        logging.info(f"Products found for keyword '{keyword}': {product_list}")
                        return jsonify({
                            "response": f"Voici une liste des produits correspondant à '{keyword}':",
                            "product_list": product_list
                        })

                    except Exception as e:
                        logging.error(f"Error during product search for keyword '{keyword}': {e}", exc_info=True)
                        return jsonify({"response": "Une erreur est survenue lors de la recherche des produits."})
                else:
                    logging.warning(f"Keyword extraction failed for user input: {user_input}")
                    return jsonify({"response": "Je n'ai pas compris le produit que vous souhaitez commander. Veuillez préciser."})

            matches = re.findall(r"(\d+)\s+([A-Za-z0-9-/]+)", user_input.strip())
            errors = []
            success_count = 0

            if matches:
                for quantity, reference in matches:
                    try:
                        cursor.execute("SELECT designation, prix_achat FROM new_product WHERE ref ILIKE %s", (reference,))
                        product = cursor.fetchone()

                        if not product:
                            logging.warning(f"Produit introuvable: Référence '{reference}' non trouvée.")
                            errors.append(f"Produit introuvable. Référence '{reference}' non trouvée.")
                            continue

                        designation, prix_achat = product
                        total_produit = round(int(quantity) * float(prix_achat), 2)

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
                                "prix_unitaire": float(prix_achat),
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
            montant_taxes = round(base_ht * 0.2, 2)  # Assuming a 20% tax rate
            net_a_payer = round(base_ht + montant_taxes, 2)

            session['data']['base_ht'] = base_ht
            session['data']['montant_taxes'] = montant_taxes
            session['data']['net_a_payer'] = net_a_payer

            try:
                # Insert or update all products in the database
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
                                item['quantite'],  # Updated quantity
                                item['total_produit'],  # Updated total price
                                item['prix_unitaire'],  # Selling price
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
                            item['description'],  # Product description
                            item['quantite'],  # Quantity of the product
                            session['data']['adresse_entreprise'],
                            session['data']['n_tva'],
                            item['total_produit'],  # Quantity * prix_ventes
                            base_ht,  # Total command value (sum of total_produit)
                            montant_taxes,
                            net_a_payer,
                            item['reference'],  # Product reference
                            item.get('prix_achat', None),  # Initial price (if available)
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

        # Attach the PDF to the email
        msg.attach(pdf_filename, "application/pdf", pdf_data)

        # Log attachment success
        logging.info(f"PDF attached to the email: {pdf_filename}")

        # Send the email
        mail.send(msg)

        # Log email sent success
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

# Save counter to a file
def save_counter(data):
    with open("quote_counter.json", "w") as f:
        json.dump(data, f)

# Increment the counter and reset it if the date changes
def get_quote_number():
    data = load_counter()
    current_date = datetime.now().strftime('%Y-%m-%d')

    if data["date"] != current_date:
        # Reset the counter if it's a new day
        data = {"date": current_date, "counter": 1}
    else:
        # Increment the counter
        data["counter"] += 1

    save_counter(data)

    # Format the number as "0001", "0002", etc.
    return f'{data["counter"]:04d}'
def generate_quote_base64(client_name, items, archive_folder="C:\\Users\\LENOVO\\new\\archives_devis", output_file=None):
    try:
        # Initialize the Jinja2 template rendering
        env = Environment(loader=FileSystemLoader("templates"))
        template = env.get_template("quote_template.html")
        current_date = datetime.now().strftime('%d/%m/%Y')
        validation_date = (datetime.now() + timedelta(days=15)).strftime('%d/%m/%Y')
        quote_number = get_quote_number()

        # Render the HTML content for the PDF
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

        # Ensure the archive folder exists
        os.makedirs(archive_folder, exist_ok=True)

        # Generate a unique filename for the PDF if not provided
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            sanitized_client_name = re.sub(r"[^\w\-_\.]", "_", client_name)
            output_file = os.path.join(archive_folder, f"devis_{sanitized_client_name}_{timestamp}.pdf")

        # Set wkhtmltopdf configuration
        config = pdfkit.configuration(wkhtmltopdf=r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe')
        options = {
            'enable-local-file-access': '',
            'quiet': ''
        }

        # Generate the PDF and save it
        pdfkit.from_string(html_content, output_file, options=options, configuration=config)

        # Convert the first page of the PDF into an image for preview
        images = convert_from_path(output_file, dpi=300, first_page=1, last_page=1)  # Increase DPI to 300 or higher
        image = images[0]

        # Enhance brightness, contrast, and sharpness (optional)
        image = ImageEnhance.Brightness(image).enhance(1.2)
        image = ImageEnhance.Contrast(image).enhance(1.5)
        image = ImageEnhance.Sharpness(image).enhance(2.0)

        # Save the enhanced image at higher resolution
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
