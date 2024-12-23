import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
import joblib
import logging
import os

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Step 1: Create SQLAlchemy engine for PostgreSQL
try:
    engine = create_engine("postgresql+psycopg2://postgres:wiem3551@localhost:5432/chatbot_db")
    logging.info("Database connection successful.")
except Exception as e:
    logging.error(f"Error creating engine: {e}")
    raise

# Step 2: Load CSV data
try:
    csv_file_path = r"C:\\Users\\LENOVO\\Downloads\\salesscenarioadjusted_with_corrected_date.csv"
    csv_data = pd.read_csv(csv_file_path, sep=';', on_bad_lines='skip')
    if len(csv_data.columns) == 1:
        csv_data.columns = csv_data.columns[0].split(';')
    if 'prix_ventes' not in csv_data.columns:
        csv_data['prix_ventes'] = csv_data['prix_achat'] * 1.1
    logging.info("CSV data loaded successfully.")
except Exception as e:
    logging.error(f"Error loading CSV: {e}")
    raise

# Standardize column names
csv_data.rename(columns={"prix_achat": "prix_achat", "prix_ventes": "prix_ventes", "quantite": "quantite", "stock": "stock", "ref": "ref_produit"}, inplace=True)

# Step 3: Load data from database
try:
    with engine.connect() as connection:
        new_product_df = pd.read_sql("SELECT * FROM new_product", con=connection)
        informations_df = pd.read_sql("SELECT * FROM informations", con=connection)
        if 'prix_initial' in informations_df.columns:
            informations_df.rename(columns={'prix_initial': 'prix_achat'}, inplace=True)
        informations_df['prix_ventes'] = informations_df['total_produit'] / informations_df['quantite']
        informations_df['prix_ventes'] = informations_df['prix_ventes'].fillna(0)
        informations_df['date'] = pd.to_datetime(informations_df['date'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
        csv_data['date'] = pd.to_datetime(csv_data['date'], format='%d/%m/%Y', errors='coerce')
        logging.info("Database tables loaded successfully.")
except Exception as e:
    logging.error(f"Error loading database data: {e}")
    raise

# Combine data
csv_data['source'] = 'csv'
informations_df['source'] = 'database'
combined_df = pd.concat([informations_df, csv_data], ignore_index=True)

# Merge with new_product data
try:
    combined_df = pd.merge(
        combined_df,
        new_product_df.rename(columns={'stock': 'product_stock'}),
        left_on='ref_produit',
        right_on='ref',
        how='left'
    )

    combined_df['prix_achat'] = combined_df.get('prix_achat_x', combined_df.get('prix_achat_y', 0))
    combined_df.drop(columns=['prix_achat_x', 'prix_achat_y'], inplace=True, errors='ignore')

    if 'prix_ventes' not in combined_df.columns:
        logging.warning("`prix_ventes` not found. Adding default values.")
        combined_df['prix_ventes'] = combined_df['prix_achat'] * 1.1
    else:
        combined_df['prix_ventes'] = combined_df['prix_ventes'].fillna(combined_df['prix_achat'] * 1.1)

    logging.info("Data merged and cleaned successfully with `new_product` table.")
except Exception as e:
    logging.error(f"Error merging data: {e}")
    raise

# Feature engineering
try:
    combined_df['price_change'] = combined_df['prix_ventes'] - combined_df['prix_achat']
    combined_df['markup'] = combined_df['price_change'] / combined_df['prix_achat']
    combined_df['is_low_stock'] = combined_df['stock'] < 20
    combined_df['is_high_stock'] = combined_df['stock'] > 200
    combined_df['year'] = combined_df['date'].dt.year
    combined_df['month'] = combined_df['date'].dt.month
    logging.info("Feature engineering completed successfully.")
except Exception as e:
    logging.error(f"Error during feature engineering: {e}")
    raise

# Handle missing values in target
try:
    combined_df.dropna(subset=['prix_ventes'], inplace=True)
    logging.info("Dropped rows with NaN values in the target column.")
except Exception as e:
    logging.error(f"Error handling NaN values in target: {e}")
    raise

# Define features and target
required_columns = ['prix_achat', 'quantite', 'stock', 'year', 'month', 'markup']
target_column = 'prix_ventes'

features = combined_df[required_columns]
target = combined_df[target_column]

# Handle missing values in features
imputer = SimpleImputer(strategy='mean')
features = pd.DataFrame(imputer.fit_transform(features), columns=features.columns)

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train model with hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5]
}
try:
    grid_search = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_grid,
        scoring='neg_mean_absolute_error',
        cv=3,
        verbose=2,
        n_jobs=-1,
        error_score='raise'
    )
    logging.info("Starting model training with GridSearchCV...")
    grid_search.fit(X_train, y_train)
    logging.info("Model training completed.")

    # Get the best model
    model = grid_search.best_estimator_

    # Evaluate model
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    logging.info(f"Mean Absolute Error: {mae:.2f}")

    # Save the model
    save_path = r'C:\\Users\\LENOVO\\new\\models\\gradient_boosting_model2.pkl'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    logging.info(f"Model saved successfully to {save_path}.")

except Exception as e:
    logging.error(f"Error during model training or saving: {e}")
    raise
