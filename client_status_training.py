import pandas as pd
import logging
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sqlalchemy import create_engine
import joblib
import os
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Database connection
connection_string = "postgresql://postgres:wiem3551@localhost:5432/chatbot_db"
engine = create_engine(connection_string)

try:
    # Step 1: Fetch Data from Tables
    logging.info("Fetching data from the database...")

    informations_query = """
    SELECT 
        nom_entreprise, 
        SUM(total_commande) AS total_spending, 
        SUM(quantite) AS total_quantity, 
        MAX(date) AS last_purchase_date
    FROM informations
    GROUP BY nom_entreprise;
    """

    scenario_query = """
    SELECT 
        nom_client, 
        SUM(totaldachat) AS predicted_spending, 
        AVG(quantite) AS avg_quantity, 
        MAX(date) AS last_scenario_date
    FROM scenario
    GROUP BY nom_client;
    """

    informations = pd.read_sql_query(informations_query, engine)
    scenario = pd.read_sql_query(scenario_query, engine)
    logging.info("Data fetched successfully.")

    # Step 2: Merge and Preprocess Data
    logging.info("Merging and preprocessing data...")
    
    # Merge data from `informations` and `scenario`
    data = pd.merge(informations, scenario, left_on='nom_entreprise', right_on='nom_client', how='left')

    # Calculate recency (days since the last purchase or scenario)
    data['recency'] = (pd.Timestamp.now() - pd.to_datetime(data['last_purchase_date'])).dt.days

    # Replace missing values with 0 for numerical fields from `scenario`
    data['predicted_spending'] = data['predicted_spending'].fillna(0)
    data['avg_quantity'] = data['avg_quantity'].fillna(0)

    # Add additional engineered features
    data['purchase_frequency'] = data['total_quantity'] / data['recency'].clip(lower=1)
    data['average_spending'] = data['total_spending'] / data['total_quantity'].clip(lower=1)
    data['churn_indicator'] = (data['recency'] > 365).astype(int)

    logging.info("Preprocessing completed successfully.")

    # Step 3: Define Features and Labels
    logging.info("Defining features and labels...")

    # Dynamically determine or fetch status
    def determine_status(row):
        if row['total_spending'] < 1000:
            return 'nouveau'
        elif row['total_spending'] < 10000:
            return 'normal'
        else:
            return 'VIP'

    # Create the status column dynamically
    data['status'] = data.apply(determine_status, axis=1)

    # Map statuses to integers
    status_mapping = {'nouveau': 0, 'normal': 1, 'VIP': 2}
    data['status'] = data['status'].map(status_mapping)

    # Define features (input) and labels (target)
    features = data[['total_spending', 'total_quantity', 'recency', 'predicted_spending', 'avg_quantity', 
                     'purchase_frequency', 'average_spending', 'churn_indicator']].copy()
    labels = data['status']

    # Handle missing values by filling with 0
    features.fillna(0, inplace=True)
    logging.info("Missing values handled. No NaNs remain in the features.")

    # Log the shapes of features and labels
    logging.info(f"Features shape: {features.shape}, Labels shape: {labels.shape}")

    # Step 4: Split Data into Training and Testing Sets
    logging.info("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    logging.info(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")

    # Step 5: Train the Model with Hyperparameter Tuning
    logging.info("Training the model with hyperparameter tuning...")
    model = RandomForestClassifier(random_state=42, n_estimators=300, max_depth=15, class_weight='balanced')
    model.fit(X_train, y_train)
    logging.info("Model training completed.")

    # Step 6: Evaluate the Model
    logging.info("Evaluating the model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Model accuracy: {accuracy}")
    logging.info("Classification Report:\n" + classification_report(y_test, y_pred))

    # Cross-validation for robustness
    logging.info("Performing cross-validation...")
    cross_val_scores = cross_val_score(model, features, labels, cv=5, scoring='accuracy')
    logging.info(f"Cross-validation scores: {cross_val_scores}")
    logging.info(f"Mean cross-validation accuracy: {cross_val_scores.mean()}")

    # Step 7: Save the Model for Later Use
    logging.info("Saving the trained model...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_file_path = rf'C:\Users\LENOVO\new\models\client_status_model_{timestamp}.pkl'

    # Ensure the directory exists
    os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
    joblib.dump(model, model_file_path)
    logging.info(f"Model saved at: {model_file_path}")

    # Feature Importance
    logging.info("Plotting feature importance...")
    import matplotlib.pyplot as plt
    feature_importance = pd.Series(model.feature_importances_, index=features.columns)
    feature_importance.sort_values(ascending=False).plot(kind='bar', title='Feature Importance')
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.show()

except Exception as e:
    logging.error(f"An error occurred: {e}")
