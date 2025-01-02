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

app = Flask(__name__)
app.secret_key = 'wiem3551'
mail = init_mail(app)

app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
Session(app)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

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
def send_email(recipient, subject, pdf_data, client_name):
    try:
        # Render the HTML email template
        env = Environment(loader=FileSystemLoader("templates"))
        template = env.get_template("email_template.html")
        html_content = template.render(client_name=client_name)

        # Ensure the Base64 string is properly padded
        if isinstance(pdf_data, str):
            missing_padding = len(pdf_data) % 4
            if missing_padding != 0:
                pdf_data += "=" * (4 - missing_padding)

        # Decode the Base64-encoded PDF data
        pdf_stream = BytesIO(base64.b64decode(pdf_data))

        # Create the email message
        msg = Message(subject, recipients=[recipient])
        msg.html = html_content  # Set the HTML content

        # Attach the PDF file
        msg.attach("devis.pdf", "application/pdf", pdf_stream.read())

        # Send the email
        mail.send(msg)
        logging.info(f"E-mail successfully sent to {recipient}")
        return True
    except Exception as e:
        logging.error(f"Error sending email: {e}", exc_info=True)
        return False


def normalize_input(user_input):
    """
    Normalize user input to handle slight variations in natural language using a lexicon.
    """
    # Predefined lexicon mapping phrases to normalized commands
    lexicon = {
        r"(je veux|je souhaite|svp|veuillez)": "",  # Remove polite filler phrases
        r"(le nom est|mon nom est)": "set_name",  # Normalize name setting
        r"(créer|nouveau|initier)": "create",  # Normalize creation commands
        r"(commander|acheter|order)": "order",  # Normalize order commands
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
def detect_action(user_input):
    user_input = user_input.lower().strip()  # Normalize input
    match = re.match(r"(ajouter|modifier|supprimer)\s*(.*)", user_input)
    if match:
        action, params = match.groups()
        return action, params
    return None, None

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
            if user_input.lower() == "non":
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
                    for item in items:
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

                    # If client doesn't exist, add them as 'nouveau' and log the status change
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
                        # Predict new status
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

                        # Log status change if there's a difference
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

                    session['step'] = 8
                    return jsonify({
                        "response": (
                            get_variable_response("summary").format(
                                base_ht=base_ht, montant_taxes=montant_taxes, net_a_payer=net_a_payer
                            )
                            + " Répondez par 'modifier' pour apporter des modifications ou 'envoyer' pour l'envoyer par e-mail."
                        ),
                        "preview_image": f"data:image/jpeg;base64,{preview_image}",
                        "pdf_data": pdf_data,
                        "pdf_filename": "devis_rempli.pdf"
                    })

                except Exception as e:
                    conn.rollback()  # Rollback transaction in case of an error
                    logging.error(f"Error saving data: {e}", exc_info=True)
                    return jsonify({"response": "Une erreur est survenue lors de l'enregistrement des données."})

            else:
                # Match and process multiple product entries
                lines = user_input.split('\n')  # Split the input by new lines
                for line in lines:
                    match = re.match(r"(\d+)\s+([A-Za-z0-9-/]+)", line.strip())  # Process each line
                    if match:
                        quantity, reference = match.groups()
                        try:
                            cursor.execute("SELECT designation, prix_achat FROM new_product WHERE ref ILIKE %s", (reference,))
                            product = cursor.fetchone()

                            if not product:
                                logging.warning(f"Produit introuvable: Référence '{reference}' non trouvée.")
                                return jsonify({"response": f"Produit introuvable. Référence '{reference}' non trouvée."})

                            designation, prix_achat = product
                            total_produit = round(int(quantity) * float(prix_achat), 2)

                            # Check if the product is already in the session's product list
                            existing_product = next(
                                (prod for prod in session['data']['products'] if prod['reference'] == reference), None
                            )

                            if existing_product:
                                # Update the existing product's quantity and total in session
                                existing_product['quantite'] += int(quantity)
                                existing_product['total_produit'] = round(existing_product['quantite'] * existing_product['prix_unitaire'], 2)
                            else:
                                # Add the new product to the session's product list
                                session['data']['products'].append({
                                    "reference": reference,
                                    "description": designation,
                                    "quantite": int(quantity),
                                    "prix_unitaire": float(prix_achat),
                                    "total_produit": total_produit
                                })

                            session.modified = True
                            logging.info(f"Produit '{designation}' ajouté ou mis à jour pour référence '{reference}'.")

                        except Exception as e:
                            conn.rollback()  # Rollback transaction in case of an error
                            logging.error(f"Error processing product '{reference}': {e}", exc_info=True)
                            return jsonify({"response": f"Une erreur est survenue lors du traitement du produit '{reference}'."})
                    else:
                        logging.warning(f"Ligne ignorée en raison du format invalide : {line.strip()}")
                        return jsonify({"response": f"Format invalide. Veuillez suivre l'exemple : '3 TUY6169'."})

                return jsonify({"response": "Tous les produits valides ont été ajoutés ou mis à jour. Tapez 'non' pour terminer ou 'oui' pour ajouter d'autres produits."})

        elif step == 8:
            action, params = detect_action(user_input)

            if action == 'modifier':
                session['step'] = 9
                return jsonify({"response": "Quelles modifications souhaitez-vous apporter ? Vous pouvez ajouter, supprimer ou modifier des produits."})

            elif action == 'envoyer':
                pdf_path = session['data'].get('pdf_path')
                if not pdf_path or not os.path.exists(pdf_path):
                    return jsonify({"response": "Aucun devis généré à envoyer. Veuillez réessayer."})
                
                try:
                    items = session['data']['products']
                    base_ht = sum(item['total_produit'] for item in items)
                    montant_taxes = round(base_ht * 0.2, 2)
                    net_a_payer = round(base_ht + montant_taxes, 2)

                    query_update_totals = """
                        UPDATE informations
                        SET total_commande = %s, montant_taxes = %s, net_a_payer = %s
                        WHERE nom_entreprise = %s AND nom_user = %s
                    """
                    cursor.execute(
                        query_update_totals,
                        (base_ht, montant_taxes, net_a_payer, session['data']['nom_entreprise'], session['data']['nom_user'])
                    )

                    conn.commit()
                    recipient_email = session['data'].get('email', 'default_client_email@example.com')
                    with open(pdf_path, "rb") as pdf_file:
                        pdf_data = pdf_file.read()

                    email_sent = send_email(
                        recipient=recipient_email,
                        subject="Votre devis",
                        pdf_data=pdf_data,
                        client_name=session['data']['nom_entreprise']
                    )
                    if email_sent:
                        return jsonify({"response": f"Devis envoyé avec succès à {recipient_email}."})
                    else:
                        return jsonify({"response": "Une erreur est survenue lors de l'envoi de l'e-mail."})

                except Exception as e:
                    conn.rollback()
                    logging.error(f"Error updating database or sending email: {e}", exc_info=True)
                    return jsonify({"response": "Une erreur est survenue lors de la mise à jour des données ou de l'envoi de l'e-mail."})

            else:
                return jsonify({"response": "Action non reconnue. Essayez 'modifier' ou 'envoyer'."})

        # Step 9
        elif step == 9:
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
                return jsonify({"response": "Action non reconnue. Utilisez 'ajouter', 'modifier', ou 'supprimer'."})

    except Exception as e:
        logging.error(f"Error in /chat endpoint: {e}", exc_info=True)
    return jsonify({"response": "Une erreur est survenue. Veuillez réessayer plus tard."})
def generate_quote_base64(client_name, items, archive_folder="C:\\Users\\LENOVO\\new\\archives_devis", output_file=None):
    try:
        # Initialize the Jinja2 template rendering
        env = Environment(loader=FileSystemLoader("templates"))
        template = env.get_template("quote_template.html")
        current_date = datetime.now().strftime('%d/%m/%Y')

        # Render the HTML content for the PDF
        html_content = template.render(
            client_name=client_name,
            client_address=session['data']['adresse_entreprise'],
            client_tva=session['data']['n_tva'],
            items=items,
            base_ht=session['data'].get('base_ht', 0),
            montant_taxes=session['data'].get('montant_taxes', 0),
            net_a_payer=session['data'].get('net_a_payer', 0),
            date=current_date
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
        images = convert_from_path(output_file, dpi=150, first_page=1, last_page=1)
        preview_image_path = os.path.join(archive_folder, f"preview_{sanitized_client_name}_{timestamp}.jpg")
        images[0].save(preview_image_path, "JPEG")

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
