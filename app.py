from flask import Flask, request, jsonify, render_template, session, redirect, url_for, flash
from flask_session import Session
import json
import psycopg2
import re
from jinja2 import Environment, FileSystemLoader
import pdfkit
import pandas as pd
import joblib
from werkzeug.middleware.profiler import ProfilerMiddleware
import os
import time
import logging
from sklearn.impute import SimpleImputer
import base64
import shap
from datetime import datetime
import chardet

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
MODEL_PATH = r'C:\Users\LENOVO\new\models\gradient_boosting_model.pkl'
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
        stock = stock_result[0] if stock_result else 100
        logging.info(f"Model expects features: {loaded_model.feature_names_in_}")

        # Historical sales
        cursor.execute('SELECT prix_initial FROM informations WHERE ref_produit = %s', (ref_produit,))
        historical_sales = cursor.fetchall()

        # Compute markup if historical data exists
        markup = 0.0  # Default value for markup
        if historical_sales:
            historical_df = pd.DataFrame(historical_sales, columns=['prix_initial'])
            historical_df['prix_initial'] = historical_df['prix_initial'].astype(float)
            historical_df['markup'] = (historical_df['prix_initial'] - float(prix_achat)) / float(prix_achat)
            markup = historical_df['markup'].mean()

        # Construct input features
        input_features = pd.DataFrame([{
            'prix_achat': float(prix_achat),
            'quantite': 1,
            'stock': float(stock),
            'markup': markup if not pd.isna(markup) else 0.0,  # Handle NaN markup
            'is_low_stock': stock < 20,
            'is_high_stock': stock > 200,
            'year': 2024,
            'month': 12
        }])

        # Reindex to ensure input matches model features
        required_columns = loaded_model.feature_names_in_
        input_features = input_features.reindex(columns=required_columns, fill_value=0)
        logging.info(f"Reindexed input features:\n{input_features}")

        # Impute missing values
        imputer = SimpleImputer(strategy='mean')
        input_features = pd.DataFrame(imputer.fit_transform(input_features), columns=input_features.columns)

        # Predict price
        predicted_price = loaded_model.predict(input_features)[0]
        logging.info(f"Predicted price: {predicted_price}")

        # SHAP explanations for interpretability
        try:
            import shap
            explainer = shap.TreeExplainer(loaded_model)
            shap_values = explainer.shap_values(input_features)

            # Log SHAP explanations
            shap_explanation = {
                feature: shap_value for feature, shap_value 
                in zip(input_features.columns, shap_values[0])
            }
            logging.info(f"SHAP explanations: {shap_explanation}")

            # Log top contributors for better analytics
            top_features = sorted(shap_explanation.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            logging.info(f"Top contributing features: {top_features}")

            # Save SHAP visualization (if matplotlib is available)
            try:
                shap.initjs()
                shap.force_plot(
                    explainer.expected_value, shap_values[0], input_features.iloc[0],
                    matplotlib=True, show=False
                ).savefig(f"shap_plot_{ref_produit}.png")
            except Exception as viz_error:
                logging.warning(f"SHAP visualization could not be saved: {viz_error}")

        except Exception as shap_error:
            logging.error(f"SHAP analysis failed: {shap_error}")

        return round(predicted_price, 2)

    except Exception as e:
        conn.rollback()
        logging.error(f"Error in analyze_and_set_price: {e}")
        return round(float(prix_achat) * 1.25, 2)  # Fallback calculation
def detect_columns(df):
    """
    Detects the most likely columns for 'reference' and 'quantity' in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame loaded from the uploaded file.

    Returns:
        tuple: Detected reference column name and quantity column name.
    """
    try:
        ref_candidates = []
        qty_candidates = []

        for col in df.columns:
            # Check for potential "reference" column (alphanumeric content)
            if df[col].apply(lambda x: isinstance(x, str) or isinstance(x, int)).all():
                alnum_ratio = df[col].apply(lambda x: bool(re.match(r'^[A-Za-z0-9-/ ]+$', str(x)))).mean()
                if alnum_ratio > 0.5:  # Adjust threshold as needed
                    ref_candidates.append((col, alnum_ratio))

            # Check for potential "quantity" column (numeric content)
            if pd.api.types.is_numeric_dtype(df[col]) or df[col].apply(lambda x: str(x).isdigit()).mean() > 0.5:
                numeric_ratio = df[col].apply(lambda x: isinstance(x, (int, float)) or str(x).isdigit()).mean()
                if numeric_ratio > 0.5:  # Adjust threshold as needed
                    qty_candidates.append((col, numeric_ratio))

        ref_candidates.sort(key=lambda x: x[1], reverse=True)
        qty_candidates.sort(key=lambda x: x[1], reverse=True)

        ref_col = ref_candidates[0][0] if ref_candidates else None
        qty_col = qty_candidates[0][0] if qty_candidates else None

        if not ref_col or not qty_col:
            raise ValueError("Unable to detect reference or quantity columns automatically.")

        return ref_col, qty_col

    except Exception as e:
        logging.error(f"Error detecting columns: {e}")
        raise RuntimeError(f"Error detecting columns: {e}")

def process_uploaded_file(file):
    """
    Processes an uploaded file to detect columns and extract relevant data.

    Args:
        file: Uploaded file object.

    Returns:
        pd.DataFrame: DataFrame with detected 'reference' and 'quantity' columns.
    """
    try:
        file.seek(0)  # Ensure file pointer is at the beginning
        raw_data = file.read()
        detected = chardet.detect(raw_data)
        encoding = detected['encoding']
        file.seek(0)  # Reset pointer again

        # Load file into a DataFrame
        try:
            # Try reading with headers
            df = pd.read_csv(file, encoding=encoding, sep=None, engine='python')
        except pd.errors.ParserError:
            # Fallback to reading without headers
            logging.warning("No headers detected, reading file without headers.")
            file.seek(0)
            df = pd.read_csv(file, encoding=encoding, sep=None, engine='python', header=None)

        logging.info(f"Raw data from file:\n{df.head()}")

        # Detect reference and quantity columns
        ref_col, qty_col = detect_columns(df)

        # Rename columns for consistency
        df = df[[ref_col, qty_col]].rename(columns={ref_col: "reference", qty_col: "quantity"})
        logging.info(f"Detected reference column: {ref_col}, quantity column: {qty_col}")

        return df

    except Exception as e:
        raise RuntimeError(f"Error processing uploaded file: {e}")

def parse_file(file):
    """
    Parses the uploaded file and retrieves relevant product details.

    Args:
        file: The uploaded file object.

    Returns:
        list: A list of dictionaries representing products with their calculated prices.
    """
    try:
        logging.info("Starting file analysis...")
        file.seek(0)  # Reset file pointer
        raw_data = file.read()
        detected = chardet.detect(raw_data)
        encoding = detected['encoding']
        logging.info(f"Detected file encoding: {encoding}")

        file.seek(0)  # Reset file pointer again

        # Dynamically detect delimiter and handle missing headers
        try:
            df = pd.read_csv(file, encoding=encoding, sep=None, engine='python', header=None)  # No headers
            logging.info(f"File loaded without headers:\n{df.head()}")
            # Assign default column names
            df.columns = ['reference', 'description', 'quantite']
        except Exception as e:
            logging.error("Failed to read file. Please check the format.")
            raise RuntimeError("Erreur lors de la lecture du fichier.") from e

        if df.empty:
            logging.error("Uploaded file is empty.")
            raise ValueError("Le fichier est vide.")

        # Log the loaded data
        logging.info(f"Raw data from file:\n{df}")

        parsed_products = []
        for _, row in df.iterrows():
            # Ensure row content is valid
            product_ref = row['reference'].strip()
            description = row['description']
            try:
                quantity = int(row['quantite'])
            except ValueError:
                logging.error(f"Invalid quantity value in row: {row}")
                raise ValueError("Quantité invalide détectée.")

            if not product_ref:
                logging.error("Missing product reference in a row.")
                raise ValueError("Chaque ligne doit contenir une 'reference'.")

            logging.info(f"Processing product reference: {product_ref}, quantity: {quantity}")

            # Fetch product details from the database
            cursor.execute('''
                SELECT ref, designation, prix_achat, stock
                FROM new_product
                WHERE ref ILIKE %s
            ''', (f"%{product_ref}%",))
            product = cursor.fetchone()

            if not product:
                logging.error(f"Product not found in database for reference: {product_ref}")
                raise ValueError(f"Le produit '{product_ref}' est introuvable dans la base de données.")

            product_reference, product_designation, product_price, stock = product
            logging.info(f"Database product details - ref: {product_reference}, "
                         f"designation: {product_designation}, prix_achat: {product_price}, stock: {stock}")

            # Predict price and calculate total
            predicted_price = analyze_and_set_price(
                prix_achat=float(product_price),
                prix_ventes=None,
                company_name=session['data'].get('nom_entreprise', ''),
                ref_produit=product_reference
            )
            total_price = round(predicted_price * quantity, 2)

            product_data = {
                "reference": product_reference,
                "produit": product_designation,
                "quantite": quantity,
                "total_produit": total_price
            }
            parsed_products.append(product_data)
            logging.info(f"Processed product data: {product_data}")

        logging.info("File analysis completed successfully.")
        return parsed_products

    except Exception as e:
        logging.error(f"Error during file processing: {e}", exc_info=True)
        raise RuntimeError(f"Erreur lors du traitement du fichier : {e}")

@app.route('/chat', methods=['POST'])
def chat():
    try:
        start_time = time.time()

        # Step 1: Initialize session
        if 'step' not in session:
            session['step'] = 0
            session['data'] = {'products': [], 'is_partner': False}
        logging.info(f"Session setup completed in {time.time() - start_time:.2f}s")

        # Step 2: Get user input or file
        step = session['step']
        data = request.json if request.is_json else None
        user_input = data.get('message', "").strip() if data else ""
        uploaded_file = request.files.get('file') if 'file' in request.files else None

        logging.info(f"Processing step {step}. User input: {user_input}, File uploaded: {bool(uploaded_file)}")

        # Step 3: Process based on the current step
        if step == 0:
            response = "Bonjour et bienvenue ! Pourriez-vous, s'il vous plaît, me fournir le nom de votre entreprise ?"
            session['step'] = 1
            return jsonify({"response": response})

        elif step == 1:
            session['data']['nom_entreprise'] = user_input
            cursor.execute("SELECT 1 FROM partenaire WHERE nom = %s", (user_input,))
            session['data']['is_partner'] = bool(cursor.fetchone())
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
            response = ("Merci. Voulez-vous télécharger un fichier contenant les produits ou les entrer manuellement ? "
                        "Répondez par 'fichier' ou 'manuel'.")
            session['step'] = 5
            return jsonify({"response": response})

        elif step == 5:
            if uploaded_file:  # Handle file upload
                try:
                    parsed_data = parse_file(uploaded_file)
                    session['data']['products'].extend(parsed_data)

                    # Process totals and generate PDF
                    items = session['data']['products']
                    base_ht = sum(item['total_produit'] for item in items)
                    montant_taxes = round(base_ht * 0.2, 2)  # Assuming 20% VAT
                    net_a_payer = round(base_ht + montant_taxes, 2)

                    session['data']['base_ht'] = base_ht
                    session['data']['montant_taxes'] = montant_taxes
                    session['data']['net_a_payer'] = net_a_payer
                    logging.info(f"Generating PDF for session data: {session['data']}")

                    pdf_data = generate_quote_base64(session['data']['nom_entreprise'], items)

                    response = (f"Le fichier a été traité avec succès. Le montant total de la commande s'élève à "
                                f"{base_ht} EUR, avec des taxes de {montant_taxes} EUR. Votre devis est prêt.")
                    session.clear()  # Clear the session after processing
                    return jsonify({
                        "response": response,
                        "pdf_data": pdf_data,
                        "pdf_filename": "devis_rempli.pdf"
                    })
                    
                except RuntimeError as e:
                    return jsonify({"response": f"Erreur de fichier : {e}"}), 400

            elif user_input.lower() == "manuel":  # Switch to manual entry
                response = "Indiquez les quantités et les références ou noms des produits (par exemple : '5 TUY6169 2 Aluminium')."
                session['step'] = 6
                logging.info(f"Session products: {session['data']['products']}")
                return jsonify({"response": response})

            else:
                response = "Réponse non reconnue. Veuillez répondre par 'fichier' ou 'manuel'."
                return jsonify({"response": response})

        elif step == 6:  # Manual entry of products
            try:
                if user_input.lower() == "non":
                    # Check if any products were added
                    if not session['data']['products']:
                        logging.error("No products found in session data.")
                        return jsonify({"response": "Aucun produit ajouté. Veuillez ajouter des produits avant de continuer."})

                    # Calculate totals
                    items = session['data']['products']
                    base_ht = sum(item['total_produit'] for item in items)
                    montant_taxes = round(base_ht * 0.2, 2)  # Assuming 20% VAT
                    net_a_payer = round(base_ht + montant_taxes, 2)

                    # Update session with totals
                    session['data']['base_ht'] = base_ht
                    session['data']['montant_taxes'] = montant_taxes
                    session['data']['net_a_payer'] = net_a_payer

                    logging.info(f"Session data before PDF generation: {session['data']}")

                    # Generate the PDF
                    try:
                        pdf_data = generate_quote_base64(session['data']['nom_entreprise'], items)
                    except Exception as pdf_error:
                        logging.error(f"Error generating PDF: {pdf_error}")
                        return jsonify({"response": "Une erreur est survenue lors de la génération du devis. Veuillez réessayer plus tard."})

                    # Clear session after generating PDF
                    session.clear()
                    response = (f"Merci ! Vos informations ont été enregistrées. "
                                f"Le montant total est {base_ht} EUR, avec {montant_taxes} EUR de taxes. Votre devis est prêt.")
                    return jsonify({
                        "response": response,
                        "pdf_data": pdf_data,
                        "pdf_filename": "devis_rempli.pdf"
                    })

                # Process manual product entries
                product_entries = user_input.splitlines()
                for entry in product_entries:
                    match = re.match(r'^(\d+)\s+(.+)$', entry.strip())
                    if not match:
                        return jsonify({"response": f"Format invalide pour la ligne : '{entry}'. Réessayez."})

                    quantity, product_ref_or_name = match.groups()
                    quantity = int(quantity)

                    # Query the database for product details
                    cursor.execute('''
                        SELECT ref, designation, prix_achat, stock
                        FROM new_product
                        WHERE ref ILIKE %s OR designation ILIKE %s
                    ''', (f"%{product_ref_or_name}%", f"%{product_ref_or_name}%"))
                    product = cursor.fetchone()

                    if not product:
                        return jsonify({"response": f"Produit '{product_ref_or_name}' introuvable. Vérifiez la référence ou le nom."})

                    product_reference, product_name, product_price, stock = product

                    # Calculate predicted price and total
                    predicted_price = analyze_and_set_price(
                        prix_achat=float(product_price),
                        prix_ventes=None,
                        company_name=session['data']['nom_entreprise'],
                        ref_produit=product_reference
                    )
                    total_produit = round(predicted_price * quantity, 2)

                    # Append the product to session data
                    product_data = {
                        "reference": product_reference,
                        "produit": product_name,
                        "quantite": quantity,
                        "prix_unitaire_brut": product_price,
                        "prix_unitaire": predicted_price,
                        "total_produit": total_produit
                    }
                    session['data']['products'].append(product_data)

                return jsonify({"response": "Produits ajoutés. Souhaitez-vous en ajouter d'autres ? (oui/non)"})

            except Exception as e:
                logging.error(f"Error in manual entry: {e}", exc_info=True)
                return jsonify({"response": "Une erreur est survenue. Veuillez réessayer plus tard."})

        elif step == 7:  # Confirmation step
            try:
                if not session['data']['products']:
                    return jsonify({"response": "Aucun produit ajouté. Veuillez ajouter des produits avant de continuer."})

                # Compute totals
                items = session['data']['products']
                base_ht = sum(item['total_produit'] for item in items)
                montant_taxes = round(base_ht * 0.2, 2)  # Assuming 20% VAT
                net_a_payer = round(base_ht + montant_taxes, 2)

                session['data']['base_ht'] = base_ht
                session['data']['montant_taxes'] = montant_taxes
                session['data']['net_a_payer'] = net_a_payer

                # Generate PDF
                pdf_data = generate_quote_base64(session['data']['nom_entreprise'], items)

                # Clear session after processing
                response = (f"Merci ! Vos informations ont été enregistrées. "
                            f"Le montant total est {base_ht} EUR, avec {montant_taxes} EUR de taxes.")
                session.clear()
                return jsonify({
                    "response": response,
                    "pdf_data": pdf_data,
                    "pdf_filename": "devis_rempli.pdf"
                })
            except Exception as e:
                logging.error(f"Error generating PDF: {e}")
                return jsonify({"response": "Une erreur est survenue lors de la génération du devis. Veuillez réessayer plus tard."})

    except Exception as e:
            logging.error(f"Error generating PDF: {e}")
    return jsonify({"response": "Une erreur est survenue lors de la génération du devis. Veuillez réessayer plus tard."})
    

def generate_quote_base64(client_name, items, output_file="quote.pdf"):
    try:
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
            items=items,
            base_ht=session['data'].get('base_ht', 0),
            montant_taxes=session['data'].get('montant_taxes', 0),
            net_a_payer=session['data'].get('net_a_payer', 0),
            date=current_date
        )

        # Log HTML content for debugging
        logging.info(f"Generated HTML content for PDF:\n{html_content}")

        # Ensure all paths (e.g., images) are correct
        options = {
            'enable-local-file-access': '',  # Required for local images
            'quiet': ''  # Suppress verbose output
        }

        pdfkit.from_string(html_content, output_file, options=options)

        # Log PDF generation success
        logging.info(f"PDF generated successfully: {output_file}")

        # Encode PDF to base64
        with open(output_file, "rb") as pdf_file:
            base64_pdf = base64.b64encode(pdf_file.read()).decode("utf-8")

        return base64_pdf

    except Exception as e:
        logging.error(f"Error generating PDF: {e}", exc_info=True)
        raise RuntimeError(f"Failed to generate PDF: {e}")

    
if __name__ == '__main__':
    app.run(debug=False)