from flask import Flask, request, jsonify, render_template, session, redirect, url_for, flash
from flask_session import Session
import json
import psycopg2
import re
from jinja2 import Environment, FileSystemLoader
import pdfkit
import pandas as pd
import joblib
from openpyxl import load_workbook  # 
import logging
from sklearn.impute import SimpleImputer
import base64
import time
import os
import magic
from datetime import datetime, date
import chardet
import mimetypes
import easyocr  # For OCR from images
import pdfplumber  # For extracting text and tables from PDFs
import shap
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = 'wiem3551'

app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False  # Optional, makes session non-permanent
Session(app)  # Initialize the session extension

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
END $$;
''')
conn.commit()

# Load the ML model
MODEL_PATH = r'C:\Users\LENOVO\new\models\gradient_boosting_model2.pkl'
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
    """
    Clear the session and render the home page.
    """
    session.clear()  # Clear the session to restart the flow
    logging.info("Session cleared: Restarting steps.")
    return render_template('index.html')


@app.route('/predict_price', methods=['POST'])
def predict_price():
    try:
        # Reuse the pre-loaded model
        data = request.json
        input_features = pd.DataFrame([{
            'prix_achat': data['prix_achat'],
            'quantite': data['quantite'],
            'stock': data['stock'],
            'markup': data.get('markup', 0),
            'is_low_stock': data['stock'] < 20,
            'is_high_stock': data['stock'] > 200,
        }])

        predicted_price = loaded_model.predict(input_features)[0]
        return jsonify({"predicted_price": round(predicted_price, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
def analyze_and_set_price(prix_achat, prix_ventes, company_name, ref_produit):
    try:
        # Query stock
        cursor.execute('SELECT stock FROM new_product WHERE ref = %s', (ref_produit,))
        stock_result = cursor.fetchone()
        stock = float(stock_result[0]) if stock_result else 100.0  # Ensure stock is float

        # Historical sales
        cursor.execute('SELECT prix_initial FROM informations WHERE ref_produit = %s', (ref_produit,))
        historical_sales = cursor.fetchall()

        # Compute markup
        markup = 0.0  # Default markup
        if historical_sales:
            historical_df = pd.DataFrame(historical_sales, columns=['prix_initial'])
            historical_df['prix_initial'] = historical_df['prix_initial'].astype(float)  # Convert to float
            historical_df['markup'] = (historical_df['prix_initial'] - float(prix_achat)) / float(prix_achat)
            markup = historical_df['markup'].mean()

        # Prepare input features
        input_features = pd.DataFrame([{
            'prix_achat': float(prix_achat),  # Convert prix_achat to float
            'quantite': 1.0,
            'stock': stock,
            'markup': markup if not pd.isna(markup) else 0.0,  # Handle NaN markup
            'is_low_stock': stock < 20.0,
            'is_high_stock': stock > 200.0,
            'year': 2024,
            'month': 12
        }])

        input_features = input_features.reindex(columns=loaded_model.feature_names_in_, fill_value=0)

        # Impute missing values
        imputer = SimpleImputer(strategy='mean')
        input_features = pd.DataFrame(imputer.fit_transform(input_features), columns=input_features.columns)

        # Predict price
        predicted_price = loaded_model.predict(input_features)[0]

        # SHAP analysis
        try:
            explainer = shap.TreeExplainer(loaded_model)
            shap_values = explainer.shap_values(input_features)

            # Create directory structure for SHAP plots
            shap_dir = r"C:\\Users\\LENOVO\\new\\shap_plot"
            os.makedirs(shap_dir, exist_ok=True)  # Ensure base directory exists

            # Save SHAP summary plot
            summary_plot_path = os.path.join(shap_dir, f"summary_plot_{ref_produit}.png")
            plt.figure()
            shap.summary_plot(shap_values, input_features, show=False)
            plt.savefig(summary_plot_path, bbox_inches='tight')
            plt.close()

            # Save SHAP waterfall plot
            waterfall_plot_path = os.path.join(shap_dir, f"waterfall_plot_{ref_produit}.png")
            plt.figure()
            shap.waterfall_plot(shap_values[0], input_features.iloc[0], show=False)
            plt.savefig(waterfall_plot_path, bbox_inches='tight')
            plt.close()

            logging.info(f"SHAP plots saved successfully: {summary_plot_path}, {waterfall_plot_path}")

        except Exception as shap_error:
            logging.error(f"Error during SHAP analysis: {shap_error}")

        return round(predicted_price, 2)

    except Exception as e:
        logging.error(f"Error in analyze_and_set_price: {e}")
        return round(float(prix_achat) * 1.25, 2)  # Fallback calculation
    
STATUS_MODEL_PATH = r'models\client_status_model_20241219_134757.pkl'
status_model = joblib.load(STATUS_MODEL_PATH)

def predict_client_status(client_data):
    """
    Predicts the client status using the trained model.

    Args:
        client_data (dict): A dictionary containing client features.

    Returns:
        str: Predicted client status ('nouveau', 'normal', 'VIP').
    """
    try:
        input_features = pd.DataFrame([client_data]).reindex(
            columns=status_model.feature_names_in_, fill_value=0
        )
        status_index = status_model.predict(input_features)[0]
        status_mapping = {0: 'nouveau', 1: 'normal', 2: 'VIP'}
        return status_mapping.get(status_index, 'unknown')
    except Exception as e:
        logging.error(f"Error predicting client status: {e}")
        return "unknown"

@app.route('/save_informations', methods=['POST'])
def save_informations():
    try:
        # Extract data from the request
        data = request.json  # Assume data is sent in JSON format

        # Define the required fields
        required_fields = [
            'nom_entreprise', 'nom_user', 'produit', 'quantite',
            'adresse_entreprise', 'n_tva', 'total_produit', 
            'total_commande', 'montant_taxes', 'net_a_payer',
            'ref_produit', 'prix_initial', 'prix_unitaire', 'date', 'prix_ventes'
        ]

        # Check for missing fields
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"status": "error", "message": f"Missing fields: {', '.join(missing_fields)}"}), 400

        # SQL Query to insert data into the `informations` table
        query = """
        INSERT INTO informations (
            nom_entreprise, nom_user, produit, quantite, adresse_entreprise, n_tva, 
            total_produit, total_commande, montant_taxes, net_a_payer, 
            ref_produit, prix_initial, prix_unitaire, date, prix_ventes
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = (
            data['nom_entreprise'], data['nom_user'], data['produit'], data['quantite'],
            data['adresse_entreprise'], data['n_tva'], data['total_produit'], 
            data['total_commande'], data['montant_taxes'], data['net_a_payer'],
            data['ref_produit'], data['prix_initial'], data['prix_unitaire'], data['date'], data['prix_ventes']
        )

        # Execute the query
        cursor.execute(query, values)
        conn.commit()  # Commit the transaction

        # Return success response
        logging.info("Data successfully saved to the 'informations' table.")
        return jsonify({"status": "success", "message": "Data saved successfully!"})

    except Exception as e:
        logging.error(f"Error saving data to the 'informations' table: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)})
    
@app.route('/chat', methods=['POST'])
def chat():
    try:
        start_time = time.time()

        # Step 1: Initialize session
        if 'step' not in session:
            session['step'] = 0
            session['data'] = {'products': [], 'is_partner': False}
        logging.info(f"Session setup completed in {time.time() - start_time:.2f}s")

        # Step 2: Get user input
        step = session['step']
        data = request.json if request.is_json else None
        user_input = data.get('message', "").strip() if data else ""

        logging.info(f"Processing step {step}. User input: {user_input}")

        # Step 3: Process based on the current step
        if step == 0:
            response = "Bonjour et bienvenue ! Pourriez-vous, s'il vous plaît, me fournir le nom de votre entreprise ?"
            session['step'] = 1
            return jsonify({"response": response})

        elif step == 1:
            session['data']['nom_entreprise'] = user_input
            response = "Merci beaucoup. Pourriez-vous également indiquer l'adresse complète de votre entreprise ?"
            session['step'] = 2
            return jsonify({"response": response})

        elif step == 2:
            session['data']['adresse_entreprise'] = user_input
            response = "Merci. Pourriez-vous me communiquer le numéro de TVA de votre entreprise ?"
            session['step'] = 3
            return jsonify({"response": response})

        elif step == 3:
            if not re.match(r"^FR\d{11}$", user_input):
                return jsonify({"response": "Numéro de TVA invalide. Recommencez avec 'FR' suivi de 11 chiffres."})
            session['data']['n_tva'] = user_input
            response = "Merci beaucoup. Puis-je également connaître votre nom complet pour nos enregistrements ?"
            session['step'] = 4
            return jsonify({"response": response})

        elif step == 4:
            session['data']['nom_user'] = user_input
            response = "Merci. Indiquez les quantités et les références ou noms des produits (par exemple : '5 TUY6169 2 Aluminium')."
            session['step'] = 5
            return jsonify({"response": response})

        elif step == 5:  # Manual entry of products
            try:
                # Check if user has finished adding products
                if user_input.lower() == "non":
                    if not session['data']['products']:
                        logging.error("No products found in session data.")
                        return jsonify({"response": "Aucun produit ajouté. Veuillez ajouter des produits avant de continuer."})

                    # Calculate totals
                    items = session['data']['products']
                    base_ht = sum(item['total_produit'] for item in items)
                    montant_taxes = round(base_ht * 0.2, 2)  # Assuming 20% VAT
                    net_a_payer = round(base_ht + montant_taxes, 2)

                    session['data']['base_ht'] = base_ht
                    session['data']['montant_taxes'] = montant_taxes
                    session['data']['net_a_payer'] = net_a_payer

                    logging.info(f"Final session data for products: {session['data']['products']}")

                    # Store session data in the database
                    try:
                        for item in items:
                            # Calculate prix_ventes if not already present
                            prix_ventes = round(item['prix_unitaire'] * 1.2, 2)  # Example: Assuming a 20% markup

                            query = """
                            INSERT INTO informations (
                                nom_entreprise, nom_user, produit, quantite,
                                adresse_entreprise, n_tva, total_produit, 
                                total_commande, montant_taxes, net_a_payer,
                                ref_produit, prix_initial, date, prix_ventes
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """
                            values = (
                                session['data']['nom_entreprise'],
                                session['data']['nom_user'],
                                item['description'],  # Mapping 'description' to 'produit'
                                item['quantite'],
                                session['data']['adresse_entreprise'],
                                session['data']['n_tva'],
                                item['total_produit'],
                                base_ht,
                                montant_taxes,
                                net_a_payer,
                                item['reference'],  # Reference for 'ref_produit'
                                None,  # prix_initial can be added if available
                                datetime.now(),  # Current date for 'date'
                                prix_ventes  # Use calculated prix_ventes
                            )
                            cursor.execute(query, values)

                        conn.commit()
                        logging.info("Session data successfully saved to the 'informations' table.")

                    except Exception as db_error:
                        logging.error(f"Error saving session data to the 'informations' table: {db_error}")
                        return jsonify({"response": "Une erreur est survenue lors de l'enregistrement des données."})

                    # Generate PDF
                    try:
                        pdf_data = generate_quote_base64(session['data']['nom_entreprise'], items)
                    except Exception as pdf_error:
                        logging.error(f"Error generating PDF: {pdf_error}")
                        return jsonify({"response": "Une erreur est survenue lors de la génération du devis."})

                    # Keep session intact until the user confirms
                    response = (f"Merci ! Vos informations ont été enregistrées. "
                                f"Le montant total est {base_ht} EUR, avec {montant_taxes} EUR de taxes. Votre devis est prêt.")
                    return jsonify({"response": response, "pdf_data": pdf_data, "pdf_filename": "devis_rempli.pdf"})

                # Handle manual product input
                match = re.match(r"(\d+)\s+([A-Za-z0-9-/]+)", user_input)
                if match:
                    quantity, reference = match.groups()
                    reference = reference.strip()

                    # Fetch product designation and price from the database
                    logging.info(f"Looking up product with reference: {reference}")
                    cursor.execute("SELECT designation, prix_achat FROM new_product WHERE ref ILIKE %s", (reference,))
                    product = cursor.fetchone()
                    logging.info(f"Query result: {product}")

                    if not product:
                        # Suggest alternatives if no exact match found
                        cursor.execute("SELECT ref FROM new_product WHERE ref ILIKE %s", (f"%{reference}%",))
                        suggestions = cursor.fetchall()
                        if suggestions:
                            return jsonify({"response": f"Produit introuvable. Suggestions : {', '.join(s[0] for s in suggestions)}"})
                        else:
                            return jsonify({"response": f"Produit avec référence '{reference}' introuvable dans la base de données."})

                    designation, prix_achat = product
                    total_produit = round(int(quantity) * float(prix_achat), 2)

                    # Add product details to session data
                    session['data']['products'].append({
                        "reference": reference,
                        "description": designation,  # Ensure description is added
                        "quantite": int(quantity),
                        "prix_unitaire": float(prix_achat),
                        "total_produit": total_produit
                    })

                    # Save session
                    session.modified = True
                    logging.info(f"Updated session products: {session['data']['products']}")

                    response = f"Produit '{designation}' ajouté ! Ajoutez un autre produit ou tapez 'non' pour terminer."
                    return jsonify({"response": response})

                else:
                    response = "Format invalide. Veuillez entrer les produits sous la forme 'quantité référence'."
                    return jsonify({"response": response})

            except Exception as e:
                logging.error(f"Error in step 5: {e}", exc_info=True)
                return jsonify({"response": "Une erreur est survenue. Veuillez réessayer."})

    except Exception as e:
        logging.error(f"Error in /chat endpoint: {e}", exc_info=True)
        return jsonify({"response": "Une erreur est survenue. Veuillez réessayer plus tard."})

def generate_quote_base64(client_name, items, output_file="quote.pdf"):
    try:
        # Debug logging for items and client data
        logging.info(f"Generating PDF for client: {client_name}")
        logging.info(f"Items data: {items}")

        # Ensure all items have the necessary keys
        for item in items:
            item.setdefault('prix_brut', 0.0)
            item.setdefault('coulee', 'N/A')

        # Load the HTML template
        env = Environment(loader=FileSystemLoader("templates"))
        template = env.get_template("quote_template.html")
        current_date = datetime.now().strftime('%d/%m/%Y')

        # Render the template with the provided data
        html_content = template.render(
            client_name=client_name,
            client_address=session['data']['adresse_entreprise'],
            client_tva=session['data']['n_tva'],
            items=items,  # Contains 'produit' (designation)
            base_ht=session['data'].get('base_ht', 0),
            montant_taxes=session['data'].get('montant_taxes', 0),
            net_a_payer=session['data'].get('net_a_payer', 0),
            date=current_date
        )
        with open("debug_quote.html", "w", encoding="utf-8") as f:
                    f.write(html_content)
        logging.info("HTML content saved to debug_quote.html")
        logging.info(f"Items passed to template: {items}")

        logging.info(f"Generated HTML content for PDF:\n{html_content}")

        # Save the rendered HTML to a file for debugging
        with open("debug_quote.html", "w", encoding="utf-8") as f:
            f.write(html_content)
        logging.info("HTML content saved to debug_quote.html")

        # Ensure all paths (e.g., images) are correct
        options = {
            'enable-local-file-access': '',  # Required for local images
            'quiet': ''  # Suppress verbose output
        }

        # Generate the PDF
        pdfkit.from_string(html_content, output_file, options=options)
        logging.info(f"PDF successfully generated at {output_file}")

        # Encode PDF to base64
        with open(output_file, "rb") as pdf_file:
            base64_pdf = base64.b64encode(pdf_file.read()).decode("utf-8")

        return base64_pdf

    except Exception as e:
        logging.error(f"Error generating PDF: {e}", exc_info=True)
        raise RuntimeError(f"Failed to generate PDF: {e}")

if __name__ == '__main__':
    app.run(debug=False)