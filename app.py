from flask import Flask, request, jsonify, render_template, session
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


app = Flask(__name__)
app.secret_key = 'wiem3551'

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
    
@app.route('/analyze_price', methods=['POST'])
def analyze_price():
    data = request.json
    prix_achat = data.get('prix_achat')
    ref_produit = data.get('ref_produit')
    company_name = data.get('company_name', "DefaultCompany")  # Optional company name

    if not prix_achat or not ref_produit:
        return jsonify({"error": "Missing required parameters: prix_achat or ref_produit"}), 400

    try:
        predicted_price, shap_explanation, shap_plot_path = analyze_and_set_price(
            prix_achat=float(prix_achat),
            prix_ventes=None,
            company_name=company_name,
            ref_produit=ref_produit
        )
        return jsonify({
            "predicted_price": predicted_price,
            "shap_explanation": shap_explanation,
            "shap_plot_path": shap_plot_path  # Include the path to the saved SHAP plot, if applicable
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
        data = request.json
        user_input = data.get('message', "").strip() if data else ""
        if not user_input:
            response = "Bonjour et bienvenue ! Pourriez-vous, s'il vous plaît, me fournir le nom de votre entreprise ?"
            session['step'] = 1
            logging.info(f"No user input, sending welcome message in {time.time() - start_time:.2f}s")
            return jsonify({"response": response})

        logging.info(f"Processing step {step}. User input: {user_input}")

        # Step 3: Process based on the current step
        if step == 0:
            response = "Bonjour et bienvenue ! Pourriez-vous, s'il vous plaît, me fournir le nom de votre entreprise ?"
            session['step'] = 1
            logging.info(f"Step 0 completed in {time.time() - start_time:.2f}s")
            return jsonify({"response": response})

        elif step == 1:
            session['data']['nom_entreprise'] = user_input

            # Check if the company is a partner
            query_start = time.time()
            cursor.execute("SELECT 1 FROM partenaire WHERE nom = %s", (user_input,))
            if cursor.fetchone():
                session['data']['is_partner'] = True
            logging.info(f"Partner query executed in {time.time() - query_start:.2f}s")

            response = "Merci beaucoup. Pourriez-vous également indiquer l'adresse complète de votre entreprise ?"
            session['step'] = 2
            logging.info(f"Step 1 completed in {time.time() - start_time:.2f}s")
            return jsonify({"response": response})

        elif step == 2:
            session['data']['adresse_entreprise'] = user_input
            response = "Merci. Pourriez-vous me communiquer le numéro de TVA de votre entreprise ?"
            session['step'] = 3
            logging.info(f"Step 2 completed in {time.time() - start_time:.2f}s")
            return jsonify({"response": response})

        elif step == 3:
            if not re.match(r"^FR\d{11}$", user_input):
                response = ("Le numéro de TVA fourni est invalide. "
                            "Le numéro de TVA pour la France doit commencer par 'FR' suivi de 11 chiffres. "
                            "Veuillez réessayer.")
                logging.info(f"Invalid TVA input in {time.time() - start_time:.2f}s")
                return jsonify({"response": response})

            session['data']['n_tva'] = user_input
            response = "Merci beaucoup. Puis-je également connaître votre nom complet pour nos enregistrements ?"
            session['step'] = 4
            logging.info(f"Step 3 completed in {time.time() - start_time:.2f}s")
            return jsonify({"response": response})

        elif step == 4:
            session['data']['nom_user'] = user_input
            response = ("Merci. Pourriez-vous, s'il vous plaît, spécifier les produits que vous souhaitez acquérir ? "
                        "Indiquez les quantités et les références ou noms des produits (par exemple : '5 TUY6169 2 Aluminium').")
            session['step'] = 5
            logging.info(f"Step 4 completed in {time.time() - start_time:.2f}s")
            return jsonify({"response": response})

        elif step == 5:
            try:
                # Split user input into product entries (one line per product entry)
                product_entries = user_input.splitlines()
                logging.info(f"Parsed product entries: {product_entries}")

                if not product_entries:
                    response = ("Format invalide. Veuillez saisir les produits sous le format "
                                "'quantité référence/nom', un ou plusieurs par ligne (par exemple : '1 ECROU INOX 1/2 1 RAUN-1-2-1/2').")
                    session['step'] = 5
                    return jsonify({"response": response})

                for entry in product_entries:
                    match = re.match(r'^(\d+)\s+(.+)$', entry)
                    if not match:
                        response = f"Format invalide pour la ligne : '{entry}'. Format attendu : 'quantité référence/nom'."
                        logging.info(f"Invalid product line: {entry}")
                        return jsonify({"response": response})

                    # Extract quantity and product reference/name
                    quantity_str, product_ref_or_name = match.groups()
                    try:
                        quantity = int(quantity_str)
                    except ValueError:
                        response = f"Quantité invalide pour le produit '{product_ref_or_name}'. Veuillez entrer un nombre entier valide."
                        return jsonify({"response": response})

                    # Query the database for matching products
                    logging.info(f"Querying product: {product_ref_or_name}")
                    cursor.execute('''
                        SELECT ref, designation, prix_achat, stock
                        FROM new_product
                        WHERE ref ILIKE %s OR designation ILIKE %s
                    ''', (f"%{product_ref_or_name}%", f"%{product_ref_or_name}%"))
                    products = cursor.fetchall()
                    logging.info(f"Product query result: {products}")

                    if not products:
                        response = f"Le produit '{product_ref_or_name}' est introuvable. Veuillez vérifier la référence ou le nom."
                        return jsonify({"response": response})

                    if len(products) > 1:
                        # Handle multiple matches for the product
                        product_options = [{"ref": p[0], "designation": p[1], "prix_achat": float(p[2]), "stock": p[3]} for p in products]
                        session['data']['pending_choices'] = {
                            "product_ref_or_name": product_ref_or_name,
                            "quantity": quantity,
                            "options": product_options
                        }
                        options_text = ", ".join([f"{opt['ref']} ({opt['designation']})" for opt in product_options])
                        response = (f"Le produit '{product_ref_or_name}' correspond à plusieurs références : {options_text}. "
                                    f"Veuillez choisir une référence en saisissant le code exact.")
                        session['step'] = 6  # Move to step 6 for user choice
                        return jsonify({"response": response})

                    # Single product match
                    product = products[0]
                    product_reference, product_name, product_price, stock = product
                    predicted_price = analyze_and_set_price(
                        prix_achat=float(product_price),
                        prix_ventes=None,
                        company_name=session['data']['nom_entreprise'],
                        ref_produit=product_reference
                    )
                    total_produit = round(predicted_price * quantity, 2)
                    session['data']['products'].append({
                        "reference": product_reference,
                        "produit": product_name,
                        "quantite": quantity,
                        "total_produit": total_produit
                    })
                    logging.info(f"Added product to session: {session['data']['products'][-1]}")

                response = "Tous les produits ont été ajoutés. Souhaitez-vous en ajouter d'autres ? (oui/non)"
                session['step'] = 7  # Move to confirmation step
                return jsonify({"response": response})

            except Exception as e:
                logging.error(f"Error in step 5: {e}", exc_info=True)
                return jsonify({"response": "Une erreur interne est survenue. Veuillez réessayer plus tard."}), 500

        elif step == 7:
            try:
                if user_input.lower() == "oui":
                    response = "Indiquez, s'il vous plaît, les produits supplémentaires avec les quantités et références (par exemple : '5 TUY6169 2 Aluminium')."
                    session['step'] = 5
                    logging.info(f"User chose to add more products in {time.time() - start_time:.2f}s")
                    return jsonify({"response": response})

                elif user_input.lower() == "non":
                    try:
                        # Calculate total_commande (Base HT) and create items list for PDF
                        items = []
                        total_commande = 0  # Initialize Base HT
                        for product in session['data']['products']:
                            # Append product details to items list
                            items.append({
                                'reference': product['reference'],
                                'produit': product['produit'],
                                'quantite': product['quantite'],
                                'prix_brut': product.get('prix_brut', 0),  # Default to 0 if not present
                                'coulee': product.get('coulee', ''),       # Default to empty if not present
                                'prix_unitaire': product['total_produit'] / product['quantite'],
                                'total': product['total_produit']
                            })
                            total_commande += product['total_produit']  # Increment Base HT
                        total_commande = round(total_commande, 2)

                        # Calculate Montant TVA and Net à Payer
                        montant_taxes = round((total_commande * 20) / 100, 2)  # Assuming 20% VAT
                        net_a_payer = round(total_commande + montant_taxes, 2)

                        # Add calculated values to session for persistence
                        session['data']['base_ht'] = total_commande
                        session['data']['montant_taxes'] = montant_taxes
                        session['data']['net_a_payer'] = net_a_payer

                        # Insert order details into the database
                        for product in session['data']['products']:
                            cursor.execute('''
                                INSERT INTO informations (
                                    nom_entreprise, adresse_entreprise, n_tva, nom_user, ref_produit, produit, quantite, 
                                    total_produit, total_commande, montant_taxes, net_a_payer
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ''', (
                                session['data']['nom_entreprise'],
                                session['data']['adresse_entreprise'],
                                session['data']['n_tva'],
                                session['data']['nom_user'],
                                product['reference'],
                                product['produit'],
                                product['quantite'],
                                float(product['total_produit']),
                                session['data']['base_ht'],
                                session['data']['montant_taxes'],
                                session['data']['net_a_payer'],
                            ))
                        conn.commit()

                        # Generate PDF
                        pdf_data = generate_quote_base64(session['data']['nom_entreprise'], items)
                        logging.info(f"Base HT: {session['data'].get('base_ht', 0)}, Montant TVA: {session['data'].get('montant_taxes', 0)}, Net à Payer: {session['data'].get('net_a_payer', 0)}")

                        # Clear session after storing everything
                        session.clear()

                        response = (f"Merci infiniment ! Vos informations ont été enregistrées avec succès. "
                                    f"Le montant total de la commande s'élève à {total_commande} EUR, avec des taxes de {montant_taxes} EUR.")
                        logging.info(f"Step 7 completed in {time.time() - start_time:.2f}s")
                        return jsonify({
                            "response": response,
                            "pdf_data": pdf_data,
                            "pdf_filename": "devis_rempli.pdf"
                        })

                    except Exception as e:
                        logging.error(f"Error inserting order details into the database: {e}")
                        conn.rollback()
                        return jsonify({"response": "Erreur lors de l'enregistrement des détails de la commande."}), 500

                else:
                    response = "Réponse non reconnue. Veuillez répondre par 'oui' ou 'non'."
                    logging.info(f"Invalid response in step 7: {user_input}")
                    return jsonify({"response": response})

            except Exception as e:
                logging.error(f"General error in step 7: {e}")
                return jsonify({"response": "Une erreur interne est survenue. Veuillez réessayer plus tard."}), 500
    except Exception as e:
                logging.error(f"General error in chat : {e}")
                return jsonify({"response": "Une erreur interne est survenue. Veuillez réessayer plus tard."}), 500

def generate_quote_base64(client_name, items, output_file="quote.pdf"):
    """
    Generates a quote PDF for the given client and items and returns it as a base64 string.

    Args:
        client_name (str): Name of the client.
        items (list): List of items, where each item is a dictionary with 'reference', 'produit', 'quantite', 'prix_brut', 'coulee', 'prix_unitaire', and 'total'.
        output_file (str): Name of the output PDF file.

    Returns:
        str: Base64-encoded PDF content.
    """
    # Add default values to items for missing keys
    for item in items:
        item.setdefault('prix_brut', 0.0)
        item.setdefault('coulee', '')

    # Load the HTML template
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("quote_template.html")
    
    # Render the template with data
    html_content = template.render(
    client_name=session['data']['nom_entreprise'],
    client_address=session['data']['adresse_entreprise'],
    client_tva=session['data']['n_tva'],
    items=items,
    base_ht=session['data'].get('base_ht', 0),
    montant_taxes=session['data'].get('montant_taxes', 0),
    net_a_payer=session['data'].get('net_a_payer', 0)
    )

    # Generate PDF using pdfkit
    pdfkit.from_string(html_content, output_file, options={
        'enable-local-file-access': None  # Enable access to local files
    })

    # Encode the PDF in base64
    with open(output_file, "rb") as pdf_file:
        base64_pdf = base64.b64encode(pdf_file.read()).decode("utf-8")

    return base64_pdf

    
if __name__ == '__main__':
    app.run(debug=False)