from flask import Flask, request, jsonify, render_template, session
import psycopg2
import re
import fitz  # PyMuPDF
from io import BytesIO
import base64

app = Flask(__name__)
app.secret_key = 'wiem3551'

# Database connection
conn = psycopg2.connect(
    dbname="chatbot_db",
    user="postgres",
    password="wiem3551",
    host="localhost",
    port="5432"
)
cursor = conn.cursor()

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

@app.route('/')
def home():
    return render_template('index.html')

# Define add_wrapped_text_with_label function to handle label and text wrapping
def add_wrapped_text_with_label(page, label, text, start_x, start_y, box_width, line_height):
    page.insert_text((start_x, start_y), f"{label} ", fontname="helv", fontsize=9, color=(0, 0, 0))
    current_x = start_x + 50  # Adjust x-offset after label as needed
    current_y = start_y
    words = text.split(" ")
    current_line = ""

    for word in words:
        test_line = f"{current_line} {word}".strip()
        if len(test_line) * 6 <= box_width:
            current_line = test_line
        else:
            page.insert_text((current_x, current_y), current_line, fontname="helv", fontsize=9, color=(0, 0, 0))
            current_line = word
            current_y += line_height
    if current_line:
        page.insert_text((current_x, current_y), current_line, fontname="helv", fontsize=10, color=(0, 0, 0))

@app.route('/chat', methods=['POST'])
def chat():
    try:
        if 'step' not in session:
            session['step'] = 0
            session['data'] = {'products': [], 'is_partner': False}

        step = session['step']
        data = request.json

        if not data or 'message' not in data or data['message'].strip() == "":
            response = "Bonjour et bienvenue ! Pourriez-vous, s'il vous plaît, me fournir le nom de votre entreprise ?"
            session['step'] = 1
            return jsonify({"response": response})

        user_input = data.get('message').strip()
        response = ""

        if step == 0:
            response = "Bonjour et bienvenue ! Pourriez-vous, s'il vous plaît, me fournir le nom de votre entreprise ?"
            session['step'] = 1
            return jsonify({"response": response})

        elif step == 1:
            session['data']['nom_entreprise'] = user_input

            # Check if the company is a partner
            cursor.execute("SELECT 1 FROM partenaire WHERE nom = %s", (user_input,))
            if cursor.fetchone():
                session['data']['is_partner'] = True

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
                response = ("Le numéro de TVA fourni est invalide. "
                            "Le numéro de TVA pour la France doit commencer par 'FR' suivi de 11 chiffres. "
                            "Veuillez réessayer.")
                return jsonify({"response": response})

            session['data']['n_tva'] = user_input
            response = "Merci beaucoup. Puis-je également connaître votre nom complet pour nos enregistrements ?"
            session['step'] = 4
            return jsonify({"response": response})

        elif step == 4:
            session['data']['nom_user'] = user_input
            response = ("Merci. Pourriez-vous, s'il vous plaît, spécifier les produits que vous souhaitez acquérir ? "
                        "Indiquez les quantités et les références ou noms des produits (par exemple : '5 TUY6169 2 Aluminium').")
            session['step'] = 5
            return jsonify({"response": response})

        elif step == 5:
            product_entries = re.findall(r'(\d+)\s+([A-Za-z0-9]+(?:[\'\s]+[A-Za-z]+)*)', user_input)
            if not product_entries:
                response = ("Format invalide. Veuillez saisir les produits sous le format '5 TUY6169 2 Aluminium'.")
                return jsonify({"response": response})

            for quantity_str, product_ref_or_name in product_entries:
                try:
                    quantity = int(quantity_str)
                except ValueError:
                    response = f"Quantité invalide pour le produit {product_ref_or_name}. Veuillez entrer un nombre."
                    return jsonify({"response": response})

                cursor.execute('''
                    SELECT reference, nom, prix
                    FROM produit
                    WHERE reference = %s OR nom = %s
                ''', (product_ref_or_name, product_ref_or_name))

                product = cursor.fetchone()
                if product:
                    product_reference = product[0]
                    product_name = product[1]
                    product_price = float(product[2])

                    # Apply a 5% discount if the client is a partner
                    if session['data']['is_partner']:
                        product_price *= 0.95

                    total_produit = round(product_price * quantity, 2)

                    session['data']['products'].append({
                        'reference': product_reference,
                        'produit': product_name,
                        'quantite': quantity,
                        'total_produit': total_produit
                    })
                else:
                    response = f"Le produit '{product_ref_or_name}' est introuvable. Veuillez vérifier la référence ou le nom."
                    return jsonify({"response": response})

            response = "Souhaitez-vous ajouter un autre produit ? (Veuillez répondre par 'oui' ou 'non')"
            session['step'] = 7
            return jsonify({"response": response})

        elif step == 7:
            if user_input.lower() == "oui":
                response = "Indiquez, s'il vous plaît, les produits supplémentaires avec les quantités et références (par exemple : '5 TUY6169 2 Aluminium')."
                session['step'] = 5
                return jsonify({"response": response})
            
            elif user_input.lower() == "non":
                total_commande = sum(product['total_produit'] for product in session['data']['products'])
                montant_taxes = round((total_commande * 20) / 100, 2)
                net_a_payer = round(total_commande + montant_taxes, 2)

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
                        total_commande,
                        montant_taxes,
                        net_a_payer
                    ))
                    conn.commit()

                pdf_response = generate_quote_base64()
                pdf_data = pdf_response["pdf_data"]

                session.clear()

                response = (f"Merci infiniment ! Vos informations ont été enregistrées avec succès. "
                            f"Le montant total de la commande s'élève à {total_commande} EUR, avec des taxes de {montant_taxes} EUR.")
            return jsonify({
                    "response": response,
                    "pdf_data": pdf_data,
                    "pdf_filename": "devis_rempli.pdf"  # Added this key to match the previous JSON structure
                })
    except Exception as e:
        print("General error:", e)
        return jsonify({"response": "Une erreur interne est survenue. Veuillez réessayer plus tard."}), 500
@app.route('/generate_quote_base64', methods=['GET'])
def generate_quote_base64():
    try:
        # Open the provided PDF template
        pdf_path = r"C:\Users\LENOVO\Downloads\contour.pdf"
        template_pdf = fitz.open(pdf_path)

        # Retrieve data from the session
        data = session.get('data', {})
        nom_entreprise = data.get('nom_entreprise', "")
        adresse_entreprise = data.get('adresse_entreprise', "")
        n_tva = data.get('n_tva', "")
        total_commande = sum(product['total_produit'] for product in data.get('products', []))
        montant_taxes = round((total_commande * 20) / 100, 2)
        net_a_payer = round(total_commande + montant_taxes, 2)

        # Define pagination variables
        max_rows_per_page = 19   # Adjust this based on layout constraints
        current_row_count = 0

        # Store all pages in a new PDF document
        output_pdf = fitz.open()  # This is a blank new PDF where we'll add pages

        # Start with a copy of the template page for the first page
        first_page = template_pdf[0]
        output_pdf.insert_pdf(template_pdf, from_page=0, to_page=0)
        page = output_pdf[-1]

        # Function to add company information on the top of each page
        def add_company_info(page):
            add_wrapped_text_with_label(page, "Nom :", nom_entreprise, 380, 88, 170, 12)
            add_wrapped_text_with_label(page, "Adresse:", adresse_entreprise, 380, 106, 170, 12)
            add_wrapped_text_with_label(page, "N° TVA:", n_tva, 380, 130, 170, 12)

        # Insert the company details on the first page
        add_company_info(page)

        # Insert products across multiple pages if necessary
        for product in data.get('products', []):
            # Add a new page if needed based on row count
            if current_row_count >= max_rows_per_page:
                # Create a new page and copy the template content layout
                output_pdf.insert_pdf(template_pdf, from_page=0, to_page=0)
                page = output_pdf[-1]  # Get the last page as the current page
                add_company_info(page)  # Reinsert company information at the top of the new page
                current_row_count = 0  # Reset row count for the new page

            # Use the existing template for the first page or duplicated pages
            if len(output_pdf) == 1:
                page = output_pdf[0]
            else:
                page = output_pdf[-1]

            y_position = 275 + (current_row_count * 20)  # Adjust based on row height
            columns_x_positions = {
                "REF": 39,
                "Description": 98,
                "Quantité": 319,
                "Prix unitaire Brut": 386,
                "N° Coulée": 500,
                "Prix unitaire": 493,
                "Total HT": 545
            }

            # Insert product details
            page.insert_text((columns_x_positions["REF"], y_position), product['reference'], fontname="helv", fontsize=10, color=(0, 0, 0))
            page.insert_text((columns_x_positions["Description"], y_position), product['produit'], fontname="helv", fontsize=10, color=(0, 0, 0))
            page.insert_text((columns_x_positions["Quantité"], y_position), str(product['quantite']), fontname="helv", fontsize=10, color=(0, 0, 0))
            page.insert_text((columns_x_positions["Prix unitaire Brut"], y_position), f"{product['total_produit'] / product['quantite']:.2f}", fontname="helv", fontsize=10, color=(0, 0, 0))
            page.insert_text((columns_x_positions["N° Coulée"], y_position), "", fontname="helv", fontsize=10, color=(0, 0, 0))  # Leave blank if not provided
            page.insert_text((columns_x_positions["Prix unitaire"], y_position), f"{product['total_produit'] / product['quantite']:.2f}", fontname="helv", fontsize=10, color=(0, 0, 0))
            page.insert_text((columns_x_positions["Total HT"], y_position), f"{product['total_produit']:,.2f}", fontname="helv", fontsize=10, color=(0, 0, 0))

            # Increment row count
            current_row_count += 1

        # Insert summary fields only on the last page
        summary_positions = {
            "Base HT": (48, 717),
            "Taux taxe": (110, 717),
            "Montant taxe": (154, 717),
            "Total HT": (535, 693),
            "Montant TVA": (535, 716),
            "Net à payer TTC": (530, 736)
        }
        summary_fields = {
            "Base HT": f"{total_commande:,.2f}",
            "Taux taxe": "20 %",
            "Montant taxe": f"{montant_taxes:,.2f}",
            "Total HT": f"{total_commande:,.2f}",
            "Montant TVA": f"{montant_taxes:,.2f}",
            "Net à payer TTC": f"{net_a_payer:,.2f}"
        }

        # Insert summary fields on the last page
        last_page = output_pdf[-1]
        for label, position in summary_positions.items():
            last_page.insert_text(position, summary_fields[label], fontname="helv", fontsize=10, color=(0, 0, 0))

        # Save the output PDF to memory
        output_buffer = BytesIO()
        output_pdf.save(output_buffer)
        output_pdf.close()
        output_buffer.seek(0)

        # Encode PDF to Base64
        pdf_base64 = base64.b64encode(output_buffer.read()).decode('utf-8')
        return {"pdf_data": pdf_base64}

    except Exception as e:
        print(f"Error generating quote: {e}")
        return jsonify({"response": "Une erreur est survenue lors de la génération du devis."}), 500


if __name__ == '__main__':
    app.run(debug=True)
