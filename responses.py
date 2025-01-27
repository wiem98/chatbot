import random

def get_variable_response(step):
    responses = {
        "welcome": [
            "Bonjour et bienvenue ! Comment puis-je vous aider aujourd'hui ?",
            "Bienvenue ! Je suis là pour vous accompagner. Que puis-je faire pour vous ?",
            "Salut ! Je suis là pour rendre les choses simples et rapides. Par où voulez-vous commencer ?",
            "Bonjour ! Que puis-je faire pour vous ? N'hésitez pas à me poser vos questions.",
            "Bienvenue à bord ! Je suis ici pour vous assister. Qu'est-ce qui vous amène aujourd'hui ?",
            "Bonjour ! Je suis ravi de vous aider. Dites-moi ce dont vous avez besoin.",
            "Salut ! Comment puis-je rendre votre expérience plus agréable aujourd'hui ?",
            "Bonjour ! Parlez-moi de vos besoins et voyons comment je peux vous aider."
        ],
        "ask_company_name": [
            "Bonjour et bienvenue ! Pour commencer, pourriez-vous m’indiquer le nom de votre entreprise ?",
            "Salut ! J’aurais besoin du nom de votre société pour personnaliser votre devis.",
            "Bonjour ! Pour créer votre devis, j’ai besoin de savoir comment s’appelle votre entreprise.",
            "Merci de me donner le nom exact de votre société pour qu’on puisse continuer.",
            "Quel est le nom de votre entreprise ? Cela nous permettra de bien personnaliser vos documents.",
            "Je vais vous accompagner dans la création du devis. Quel est le nom de votre société ?",
            "Pouvez-vous m’indiquer le nom de votre entreprise, s’il vous plaît ?"
        ],
        
        "ask_company_address": [
            "Merci ! Quelle est l’adresse complète de votre entreprise ?",
            "Super ! Pouvez-vous m’indiquer l’adresse officielle de votre société ?",
            "J’aurais besoin de l’adresse complète de votre entreprise, incluant la ville et le code postal.",
            "Merci ! Où se situe votre entreprise ? N’oubliez pas de préciser l’adresse exacte.",
            "Pour avancer, pourriez-vous me donner l’adresse complète de votre société ?",
            "J’ai besoin de l’adresse officielle pour compléter vos informations. Pouvez-vous me la communiquer ?",
            "Pour enregistrer votre entreprise correctement, pourriez-vous indiquer son adresse complète ?"
        ],

        "ask_vat_number": [
            "Merci. Pouvez-vous me fournir le numéro de TVA de votre entreprise ?",
            "Très bien ! J’aurais besoin du numéro de TVA pour compléter les informations de votre société.",
            "Pour nos enregistrements, pourriez-vous m’indiquer le numéro de TVA de votre entreprise ?",
            "Pour continuer, il me faut le numéro de TVA de votre société. Pouvez-vous me le communiquer ?",
            "J’ai besoin de connaître le numéro de TVA de votre entreprise pour finaliser cette étape.",
            "Merci de m’envoyer votre numéro de TVA. Cela nous permettra d’avancer rapidement.",
            "Pouvez-vous me transmettre le numéro de TVA officiel de votre société ?"
        ],

        "invalid_vat": [
            "Le numéro de TVA que vous avez saisi ne semble pas valide. Pouvez-vous le vérifier et réessayer ?",
            "Je ne reconnais pas ce numéro de TVA. Assurez-vous qu’il respecte le bon format.",
            "Ce numéro de TVA semble incorrect. Pouvez-vous vérifier et me l’envoyer à nouveau ?",
            "Oups ! Le numéro de TVA saisi est invalide. Vérifiez-le et essayez encore une fois.",
            "Il semble que le numéro de TVA ne soit pas valide. Merci de le revérifier avant de le renvoyer.",
            "Le format du numéro de TVA ne correspond pas. Vérifiez-le pour être sûr qu’il soit exact.",
            "Ce numéro de TVA ne semble pas correct. Pouvez-vous le contrôler et le saisir à nouveau ?"
        ],
        "ask_user_name": [
            "Merci ! Pouvez-vous me donner votre nom complet ?",
            "Super ! J’aurais besoin de connaître votre nom complet.",
            "Pour continuer, pourriez-vous m’indiquer votre nom complet, s’il vous plaît ?",
            "Merci ! Comment vous appelez-vous ?",
            "Pouvez-vous me transmettre votre nom complet pour compléter votre profil ?",
            "J’aurais besoin de votre nom complet pour personnaliser les informations.",
            "Pour finaliser cette étape, pouvez-vous me donner votre nom complet ?",
            "Quel est votre nom complet ? Cela m’aidera à bien personnaliser votre devis."
        ],

       "ask_products": [
            "Merci pour l'information. Vous pouvez indiquer les produits avec leurs quantités en écrivant chaque produit sur une nouvelle ligne. Si vous ne connaissez pas la référence, vous pouvez écrire une description.",
            "Parfait ! Ajoutez les produits avec les quantités et les références, un produit par ligne, ou décrivez ce que vous cherchez.",
            "Très bien, indiquez les produits en précisant la quantité et la référence. Vous pouvez aussi écrire une description du produit pour effectuer une recherche.",
            "Vous pouvez ajouter plusieurs produits à la fois. Merci d'écrire chaque produit avec sa quantité et sa référence ou une description, un produit par ligne. "
        ],

       "invalid_product_format": [
            "Le format est incorrect. Veuillez entrer les produits avec la quantité et la référence.",
            "Désolé, je n'ai pas compris votre saisie. Merci de préciser la quantité et la référence.",
            "Je n'ai pas pu interpréter correctement. Assurez-vous d'inclure une quantité et une référence.",
            "Le format attendu inclut une quantité suivie de la référence. Essayez à nouveau.",
            "Vérifiez votre saisie, elle doit inclure une quantité et une référence de produit."
        ],

       "no_products": [
            "Il n'y a aucun produit dans la liste pour l'instant. Ajoutez-en pour continuer.",
            "Votre liste de produits est vide. Merci d'ajouter des articles avant de poursuivre.",
            "Aucun produit détecté. Veuillez entrer des articles avec leurs quantités.",
            "Vous n'avez pas encore ajouté de produits. Indiquez-les pour générer votre devis.",
            "Pour continuer, veuillez renseigner quelques produits avec leurs quantités."
        ],

        "ask_email": [
            "Pouvez-vous me fournir une adresse e-mail valide ?",
            "Merci de m'indiquer une adresse e-mail.",
            "J'ai besoin de votre adresse e-mail. Pourriez-vous la partager ?",
            "Pour continuer, veuillez entrer une adresse e-mail valide.",
            "Merci de saisir une adresse e-mail."
        ],

        "summary": [
            "Voici un récapitulatif de votre devis :\n- Montant HT : {base_ht} EUR\n- Taxes : {montant_taxes} EUR\n- Total TTC : {net_a_payer} EUR\nTapez 'oui' pour confirmer.",
            "Votre devis est prêt :\n- Total hors taxes : {base_ht} EUR\n- Taxes (TVA) : {montant_taxes} EUR\n- Total TTC : {net_a_payer} EUR\nConfirmez avec 'oui'.",
            "Résumé de votre commande :\n- HT : {base_ht} EUR\n- Taxes : {montant_taxes} EUR\n- TTC : {net_a_payer} EUR\nTapez 'oui' si tout est correct.",
            "Le total de votre devis est :\n- Hors taxes : {base_ht} EUR\n- Taxes : {montant_taxes} EUR\n- TTC : {net_a_payer} EUR\nValidez avec 'oui' pour finaliser.",
            "Voici votre devis :\n- Montant HT : {base_ht} EUR\n- Taxes : {montant_taxes} EUR\n- Total TTC : {net_a_payer} EUR\nConfirmez en tapant 'oui'."
        ],
        "invalid_email": [
            "L'adresse e-mail que vous avez saisie ne semble pas valide. Pouvez-vous la vérifier et réessayer ?",
            "Cette adresse e-mail est invalide. Merci d'utiliser un format valide.",
            "Oups ! L'adresse e-mail n'est pas correcte. Veuillez la vérifier et la saisir à nouveau.",
            "L'adresse e-mail saisie ne respecte pas le bon format. Merci de réessayer.",
            "Il semble que votre adresse e-mail soit incorrecte. Assurez-vous de bien la saisir."
        ],
        "company_already_exists": [
            "La société est déjà enregistrée dans nos systèmes. Passons directement à votre nom complet !",
            "Nous avons trouvé les informations de votre société. Continuons avec votre nom  pour finaliser.",
            "Votre entreprise est déjà répertoriée. Parfait, cela nous fait gagner du temps ! Nous allons maintenant passer à votre nom d'utilisateur.",
            "Votre société est bien identifiée dans notre base. Concentrons-nous sur votre nom complet .",
            "Bonne nouvelle ! Votre société figure déjà dans nos enregistrements. Pouvez-vous me fournir votre nom pour continuer ?",
            "Nous avons retrouvé les informations de votre entreprise. Maintenant, personnalisons les détails en ajoutant votre nom.",
            "Super ! Votre société est déjà configurée. Poursuivons avec votre nom pour compléter le dossier."
        ]

    }
    return random.choice(responses.get(step, ["Désolé, je ne comprends pas."]))
