import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# Step 1: Create SQLAlchemy engine for PostgreSQL
try:
    engine = create_engine("postgresql+psycopg2://postgres:wiem3551@localhost:5432/chatbot_db")
    print(f"Engine created successfully: {type(engine)}")
except Exception as e:
    print(f"Error creating engine: {e}")
    raise

# Step 2: Load data from CSV file
try:
    csv_file_path = r"C:\Users\LENOVO\Downloads\salesscenarioadjusted_with_corrected_date.csv"
    csv_data = pd.read_csv(csv_file_path, sep=';', on_bad_lines='skip')
    print("CSV data loaded successfully.")
    
    # Handle concatenated columns if necessary
    if len(csv_data.columns) == 1:
        csv_data.columns = csv_data.columns[0].split(';')
        print("CSV columns after splitting:", csv_data.columns)
    
    if 'prix_ventes' not in csv_data.columns:
        print("Warning: 'prix_ventes' column missing in CSV. Setting default values.")
        csv_data['prix_ventes'] = csv_data['prix_achat'] * 1.1  # Example markup
except Exception as e:
    print(f"Error loading or processing CSV file: {e}")
    raise

# Standardize column names in CSV
csv_data.rename(
    columns={
        "prix_achat": "prix_achat",
        "prix_ventes": "prix_ventes",
        "quantite": "quantite",
        "stock": "stock",
        "ref": "ref_produit"
    },
    inplace=True
)

# Step 3: Load data from database tables
try:
    with engine.connect() as connection:
        new_product_df = pd.read_sql("SELECT * FROM new_product", con=connection)
        informations_df = pd.read_sql("SELECT * FROM informations", con=connection)
    print("Database tables loaded successfully.")

    # Ensure `prix_achat` exists in `informations_df`
    if 'prix_initial' in informations_df.columns:
        informations_df.rename(columns={'prix_initial': 'prix_achat'}, inplace=True)
    if 'prix_achat' not in informations_df.columns:
        print("Warning: 'prix_achat' column missing in informations_df. Setting default values.")
        informations_df['prix_achat'] = 0  # Replace with a calculated or default value

    # Ensure `prix_ventes` exists in `informations_df`
    if 'prix_ventes' not in informations_df.columns:
        informations_df['prix_ventes'] = informations_df['total_produit'] / informations_df['quantite']
        informations_df['prix_ventes'] = informations_df['prix_ventes'].fillna(0)  # Handle NaN values

    print("`prix_ventes` column successfully created in `informations_df`.")

    # Standardize date formats in `informations_df`
    informations_df['date'] = pd.to_datetime(informations_df['date'], format='%d/%m/%Y %H:%M:%S', errors='coerce')

    # Standardize date formats in `csv_data`
    csv_data['date'] = pd.to_datetime(csv_data['date'], format='%d/%m/%Y', errors='coerce')

    # Remove timezone information if present
    informations_df['date'] = informations_df['date'].dt.tz_localize(None)
    csv_data['date'] = csv_data['date'].dt.tz_localize(None)

except Exception as e:
    print(f"Error loading data or standardizing dates: {e}")
    raise

# Combine data
csv_data['source'] = 'csv'
informations_df['source'] = 'database'
combined_df = pd.concat([informations_df, csv_data], ignore_index=True)

# Debugging combined data columns
print("Columns in combined_df after combining:", combined_df.columns)

# Ensure `prix_achat` exists in combined data
if 'prix_achat' not in combined_df.columns:
    print("Warning: 'prix_achat' column missing in combined data. Setting default values.")
    combined_df['prix_achat'] = 0  # Replace with a calculated or default value

# Merge with `new_product` to enrich data
try:
    # Inspect columns before merging
    print("Columns in combined_df before merge:", combined_df.columns)
    print("Columns in new_product_df:", new_product_df.columns)

    # Merge with `new_product` table
    combined_df = pd.merge(
        combined_df,
        new_product_df.rename(columns={'stock': 'product_stock'}),
        left_on='ref_produit',
        right_on='ref',
        how='left'
    )
    print("Data merged successfully with `new_product` table.")

    # Handle missing `prix_achat` after merge
    if 'prix_achat' not in combined_df.columns:
        print("`prix_achat` column is missing after merge. Reconstructing...")
        if 'prix_achat_x' in combined_df.columns and 'prix_achat_y' in combined_df.columns:
            combined_df['prix_achat'] = combined_df['prix_achat_x'].fillna(combined_df['prix_achat_y'])
            combined_df.drop(columns=['prix_achat_x', 'prix_achat_y'], inplace=True)
        else:
            raise KeyError("Unable to reconstruct `prix_achat` from merged data.")

    # Handle missing `prix_ventes`
    if 'prix_ventes' not in combined_df.columns or combined_df['prix_ventes'].isnull().all():
        print("`prix_ventes` column is missing or entirely NaN after merge. Reconstructing...")
        combined_df['prix_ventes'] = combined_df['prix_achat'] * 1.1  # Example markup

except Exception as e:
    print(f"Error merging data: {e}")
    raise


# Drop duplicates
combined_df.drop_duplicates(inplace=True)

# Feature engineering
try:
    # Handle missing values in categorical columns
    combined_df['status'] = combined_df['status'].fillna('unknown')
    combined_df['nom_user'] = combined_df['nom_user'].fillna('unknown')

    # Create new features
    combined_df['price_change'] = combined_df['prix_ventes'] - combined_df['prix_achat']
    combined_df['markup'] = combined_df['price_change'] / combined_df['prix_achat']
    combined_df['is_low_stock'] = combined_df['stock'] < 20
    combined_df['is_high_stock'] = combined_df['stock'] > 200

    # Handle invalid dates
    combined_df.dropna(subset=['date'], inplace=True)
    combined_df['year'] = combined_df['date'].dt.year
    combined_df['month'] = combined_df['date'].dt.month
    combined_df['day'] = combined_df['date'].dt.day

    # Encode categorical features
    combined_df = pd.get_dummies(combined_df, columns=['status', 'nom_user'], drop_first=True)
    print("Feature engineering completed successfully.")
except Exception as e:
    print(f"Error during feature engineering: {e}")
    raise

# Ensure required features exist
required_columns = ['prix_achat', 'quantite', 'stock', 'year', 'month', 'markup']
for col in required_columns:
    if col not in combined_df.columns:
        combined_df[col] = 0

# Step 6: Split data into training and testing sets
features = combined_df[required_columns]
target = combined_df['prix_ventes']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Handle missing values using imputation
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Verify no missing values
assert not X_train.isnull().any().any(), "X_train contains NaN after imputation!"
assert not X_test.isnull().any().any(), "X_test contains NaN after imputation!"

# Step 7: Train the Gradient Boosting Regressor
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Step 8: Evaluate the model
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error: {mae}")

# Step 9: Save the trained model
joblib.dump(model, r'C:\Users\LENOVO\new\models\gradient_boosting_model.pkl')
print(r"Model saved successfully in C:\Users\LENOVO\new\models\gradient_boosting_model.pkl.")
