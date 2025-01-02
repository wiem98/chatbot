from flask_mail import Mail

def init_mail(app):
    app.config['MAIL_SERVER'] = 'smtp.mailtrap.io'  # Mailtrap SMTP server
    app.config['MAIL_PORT'] = 2525  
    app.config['MAIL_USE_TLS'] = True  # Enable TLS
    app.config['MAIL_USE_SSL'] = False  # Disable SSL
    app.config['MAIL_USERNAME'] = 'dce59f6db94a77'  # Your Outlook email address
    app.config['MAIL_PASSWORD'] = 'ccc0c065f29eb8'  # Your password or app password
    app.config['MAIL_DEFAULT_SENDER'] = ('Customer Support', 'no-reply@example.com')  # Default sender

    return Mail(app)
