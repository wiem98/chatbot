import random

def get_variable_response(step):
    responses = {
        "welcome": [
            "Bonjour ! Je suis votre assistant virtuel. Comment puis-je vous aider aujourd'hui ?",
            "Bienvenue ! Je suis là pour répondre à vos besoins. Que puis-je faire pour vous ?",
            "Salut ! Je suis là pour faciliter votre expérience. Que souhaitez-vous faire ?",
            "Bonjour ! Vous pouvez demander un devis, poser des questions ou explorer d'autres options. Comment puis-je vous aider ?"
        ],
        "ask_company_name": [
            "Bonjour et bienvenue ! Pour commencer, pouvez-vous me donner le nom de votre entreprise ? Par exemple : 'Entreprise X'.",
            "Salut ! Quel est le nom de votre entreprise ? Par exemple : 'Société Y'.",
            "Bonjour ! Je vais vous aider à créer votre devis. Comment s'appelle votre entreprise ? Exemple : 'ABC Corp'.",
            "Pour personnaliser votre devis, pouvez-vous me donner le nom exact de votre société ? Exemple : 'XYZ Industries'."
        ],
        "ask_company_address": [
            "Merci beaucoup ! Quelle est l'adresse complète de votre entreprise ? Exemple : '123 Rue de Paris, 75001 Paris'.",
            "Super ! Pouvez-vous me fournir l'adresse exacte de votre entreprise, y compris le code postal ? Exemple : '10 Avenue des Champs, 75008 Paris'.",
            "Merci ! Pour nos archives, pourriez-vous indiquer l'adresse complète de votre société ? Exemple : '456 Rue Principale, Lyon'.",
            "Indiquez l'adresse où vous recevez vos courriers officiels. Par exemple : '789 Boulevard St-Michel, 33000 Bordeaux'."
        ],
        "ask_vat_number": [
            "Merci. Quel est le numéro de TVA de votre entreprise ? Exemple : 'FR12345678901'.",
            "Très bien ! Pouvez-vous me donner le numéro de TVA au format correct ? Exemple : 'FR98765432101'.",
            "Merci. Pour nos enregistrements, pourriez-vous me communiquer le numéro de TVA ? Exemple : 'FR11122233344'.",
            "Veuillez indiquer le numéro de TVA de votre société. Format attendu : 'FR' suivi de 11 chiffres. Exemple : 'FR99988877766'."
        ],
        "invalid_vat": [
            "Le numéro de TVA que vous avez saisi est invalide. Assurez-vous qu'il commence par 'FR' suivi de 11 chiffres. Exemple : 'FR12345678901'.",
            "Numéro de TVA incorrect. Veuillez recommencer avec un format valide : 'FR' suivi de 11 chiffres. Exemple : 'FR98765432101'.",
            "Le format du numéro de TVA semble incorrect. Merci d'utiliser le format attendu. Exemple : 'FR11122233344'.",
            "Numéro invalide. Vérifiez que le numéro commence par 'FR' et contient exactement 11 chiffres. Exemple : 'FR99988877766'."
        ],
        "ask_user_name": [
            "Merci. Pourriez-vous indiquer votre nom complet ? Exemple : 'Jean Dupont'.",
            "Super ! Quel est votre nom complet ? Exemple : 'Marie Curie'.",
            "Merci. Pouvez-vous me donner votre nom complet pour nos archives ? Exemple : 'Paul Durand'.",
            "Pour finaliser vos informations, indiquez votre nom complet. Exemple : 'Elise Martin'."
        ],
        "ask_products": [
            "Merci de l'information. Veuillez indiquer les produits avec leurs quantités en écrivant chaque produit sur une nouvelle ligne. Exemple :\n1 TUY6169\n2 Aluminium",
            "Parfait ! Ajoutez les produits avec les quantités et les références, un produit par ligne. Exemple :\n3 TUY6169",
            "Très bien, indiquez les produits en précisant la quantité et la référence. Chaque produit doit être écrit sur une nouvelle ligne. Exemple :\n2 PRO1234\n4 Steel Pipe",
            "Vous pouvez ajouter plusieurs produits à la fois. Merci d'écrire chaque produit avec sa quantité et sa référence sur une ligne distincte. Exemple :\n10 TUY1234\n5 PLA5678"
        ],

        "invalid_product_format": [
            "Format invalide. Veuillez entrer les produits sous la forme 'quantité référence'. Exemple : '5 TUY6169'.",
            "Désolé, je n'ai pas compris. Essayez de taper : '3 Aluminium'.",
            "Format incorrect. Merci d'utiliser la syntaxe 'quantité référence'. Exemple : '4 UNIO2023'.",
            "Vérifiez le format. Exemple attendu : '2 TUY3039' ou '1 Steel'."
        ],
        "no_products": [
            "Aucun produit ajouté. Veuillez ajouter des produits avant de continuer. Exemple : '3 TUY1234'.",
            "Il semble que vous n'ayez pas ajouté de produits. Ajoutez-en quelques-uns pour générer le devis. Exemple : '5 Aluminium'.",
            "Pas de produits dans la liste. Merci d'en ajouter pour continuer. Exemple : '1 UNIO2023'.",
            "Vous n'avez pas encore saisi de produits. Entrez des références et quantités. Exemple : '4 TUY6169 2 PLA5678'."
        ],
        "ask_email": [
            "Merci ! Pourriez-vous me fournir une adresse e-mail valide où je peux envoyer le devis ? Exemple : 'exemple@email.com'.",
            "Veuillez indiquer une adresse e-mail pour recevoir le devis. Exemple : 'contact@entreprise.fr'.",
            "Pour finaliser les informations, pouvez-vous fournir une adresse e-mail ? Exemple : 'client@domain.com'.",
            "Merci de partager une adresse e-mail pour l'envoi de votre devis. Exemple : 'info@exemple.fr'."
        ],
        "summary": [
            "Voici le résumé de votre devis :\n- Montant total HT : {base_ht} EUR\n- Taxes : {montant_taxes} EUR\n- Total TTC : {net_a_payer} EUR\nTapez 'oui' pour confirmer.",
            "Résumé :\n- Total HT : {base_ht} EUR\n- Taxes (20%) : {montant_taxes} EUR\n- Net à payer : {net_a_payer} EUR\nConfirmez avec 'oui'.",
            "Votre devis est prêt :\n- Montant HT : {base_ht} EUR\n- Taxes (TVA 20%) : {montant_taxes} EUR\n- Total TTC : {net_a_payer} EUR\nTapez 'oui' pour valider.",
            "Le total de votre commande est :\n- Montant HT : {base_ht} EUR\n- Taxes : {montant_taxes} EUR\n- Montant TTC : {net_a_payer} EUR\nConfirmez avec 'oui'."
        ]
    }
    return random.choice(responses.get(step, ["Désolé, je ne comprends pas."]))
