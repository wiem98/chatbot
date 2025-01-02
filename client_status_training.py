import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
from sqlalchemy import create_engine
import joblib
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Database connection
connection_string = "postgresql://postgres:wiem3551@localhost:5432/chatbot_db"
engine = create_engine(connection_string)

try:
    # Step 1: Fetch Data from Database
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

    data = pd.merge(informations, scenario, left_on='nom_entreprise', right_on='nom_client', how='left')

    data['recency'] = (pd.Timestamp.now() - pd.to_datetime(data['last_purchase_date'])).dt.days
    data['predicted_spending'] = data['predicted_spending'].fillna(0)
    data['avg_quantity'] = data['avg_quantity'].fillna(0)
    data['purchase_frequency'] = data['total_quantity'] / data['recency'].clip(lower=1)
    data['average_spending'] = data['total_spending'] / data['total_quantity'].clip(lower=1)
    data['churn_indicator'] = (data['recency'] > 365).astype(int)

    # Seasonal Spending
    data['purchase_date'] = pd.to_datetime(data['last_purchase_date'])
    three_months_ago = pd.Timestamp.now() - pd.Timedelta(days=90)
    recent_purchases = data[data['purchase_date'] > three_months_ago]
    recent_purchases_sum = recent_purchases.groupby('nom_client')['total_spending'].sum()
    data['total_spending_last_3_months'] = data['nom_client'].map(recent_purchases_sum).fillna(0)

    data['large_purchase'] = (data['total_spending'] > 5000).astype(int)

    # Low-selling product flag
    product_sales = informations.groupby('nom_entreprise')['total_quantity'].sum().reset_index()
    low_sales_threshold = 10
    low_selling_products = product_sales[product_sales['total_quantity'] < low_sales_threshold]
    data['low_selling_product'] = data['nom_entreprise'].isin(low_selling_products['nom_entreprise']).astype(int)
    data['frequent_low_selling_purchase'] = data.groupby('nom_client')['low_selling_product'].transform('sum')

    # Determine Client Status
    def determine_new_status(row):
        if row['total_spending_last_3_months'] > 5000:
            return 'VIP'
        elif row['large_purchase'] == 1:
            return 'normal'
        elif row['purchase_frequency'] < 1:
            return 'nouveau'
        else:
            return 'nouveau'

    data['new_status'] = data.apply(determine_new_status, axis=1)
    status_mapping = {'nouveau': 0, 'normal': 1, 'VIP': 2}
    data['new_status'] = data['new_status'].map(status_mapping)

    # Define Features and Labels
    features = data[['total_spending', 'total_quantity', 'recency', 'predicted_spending', 'avg_quantity', 
                     'purchase_frequency', 'average_spending', 'churn_indicator', 'total_spending_last_3_months', 
                     'large_purchase', 'frequent_low_selling_purchase']].copy()
    labels = data['new_status']
    features.fillna(0, inplace=True)

    # Step 3: Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Step 4: Oversampling for Class Imbalance
    logging.info("Handling class imbalance using SMOTE...")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Step 5: Model Training and Hyperparameter Tuning
    logging.info("Training the model with GridSearchCV...")
    param_grid = {
        'n_estimators': [100, 300, 500],
        'max_depth': [10, 15, 20],
        'class_weight': ['balanced', 'balanced_subsample'],
    }
    rf_model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    logging.info(f"Best model parameters: {grid_search.best_params_}")

    # Step 6: Evaluation
    logging.info("Evaluating the model...")
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    logging.info(f"Accuracy: {accuracy}, F1-Score: {f1}")
    logging.info("Classification Report:\n" + classification_report(y_test, y_pred))

    # Confusion Matrix
    logging.info("Plotting the confusion matrix...")
    labels_list = [0, 1, 2]  # Ensure all labels are represented
    cm = confusion_matrix(y_test, y_pred, labels=labels_list)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=status_mapping.keys(), yticklabels=status_mapping.keys())
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # Feature Importance
    feature_importance = pd.Series(best_model.feature_importances_, index=features.columns)
    feature_importance.sort_values(ascending=False).plot(kind='bar', title='Feature Importance')
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.show()

    # SHAP Explanation
    logging.info("Explaining the model with SHAP...")
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(features)
    shap.summary_plot(shap_values, features)

    # Step 7: Save the Model
    logging.info("Saving the trained model...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_file_path = rf'C:\Users\LENOVO\new\models\client_status_model_enhanced.pkl'
    os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
    joblib.dump(best_model, model_file_path)
    logging.info(f"Model saved at: {model_file_path}")

except Exception as e:
    logging.error(f"An error occurred: {e}")
