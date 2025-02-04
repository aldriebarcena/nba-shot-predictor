import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# Load the dataset
final_df = pd.read_csv('warriors_player_shots_with_defense.csv')

# Step 1: Handle Missing Values
def handle_missing_values(df):
    # Drop rows with missing values in key columns
    df = df.dropna(subset=['SHOT_MADE_FLAG', 'SHOT_ZONE_AREA', 'DEF_RATING'])
    return df

# Step 2: Encode Categorical Variables
def encode_categorical_variables(df):
    # One-hot encode the 'SHOT_ZONE_AREA' column
    encoder = OneHotEncoder(sparse_output=False, drop='first')  # Use sparse_output instead of sparse
    encoded_zones = encoder.fit_transform(df[['SHOT_ZONE_AREA']])
    encoded_zones_df = pd.DataFrame(encoded_zones, columns=encoder.get_feature_names_out(['SHOT_ZONE_AREA']))
    
    # Concatenate encoded columns with the original DataFrame
    df = pd.concat([df, encoded_zones_df], axis=1)
    
    # Drop the original 'SHOT_ZONE_AREA' column
    df = df.drop(columns=['SHOT_ZONE_AREA'])
    
    return df

# Step 3: Normalize Numerical Features
def normalize_numerical_features(df):
    # Define numerical columns to normalize (adjusted to available columns)
    numerical_columns = ['SHOT_DISTANCE', 'DEF_RATING']  # Removed 'CLOSE_DEF_DIST'
    
    # Apply Min-Max scaling
    scaler = MinMaxScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    
    return df

# Step 4: Feature Engineering
def feature_engineering(df):
    # Create a new feature: Shot Difficulty (e.g., inverse of shot distance)
    df['SHOT_DIFFICULTY'] = 1 / (df['SHOT_DISTANCE'] + 1)  # Add 1 to avoid division by zero
    
    # Create a new feature: Defender Impact (e.g., defender rating * shot distance)
    # Since 'CLOSE_DEF_DIST' doesn't exist, use 'DEF_RATING' as an alternative for defender impact
    df['DEFENDER_IMPACT'] = df['DEF_RATING'] * df['SHOT_DISTANCE']  # Adjusted based on available columns
    
    return df

# Step 5: Split Data into Training and Testing Sets
def split_data(df):
    # Define features (X) and target (y)
    X = df.drop(columns=['SHOT_MADE_FLAG'])  # Features
    y = df['SHOT_MADE_FLAG']  # Target (whether the shot was made)
    
    # Split the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

# Main Workflow for Cleaning and Preprocessing
def clean_and_preprocess_data(df):
    # Step 1: Handle Missing Values
    df = handle_missing_values(df)
    
    # Step 2: Encode Categorical Variables
    df = encode_categorical_variables(df)
    
    # Step 3: Normalize Numerical Features
    df = normalize_numerical_features(df)
    
    # Step 4: Feature Engineering
    df = feature_engineering(df)
    
    # Step 5: Split Data into Training and Testing Sets
    X_train, X_test, y_train, y_test = split_data(df)
    
    return X_train, X_test, y_train, y_test

# Execute Cleaning and Preprocessing
if __name__ == "__main__":
    # Load the dataset
    final_df = pd.read_csv('warriors_player_shots_with_defense.csv')
    
    # Clean and preprocess the data
    X_train, X_test, y_train, y_test = clean_and_preprocess_data(final_df)
    
    # Output the results to CSV files
    X_train.to_csv('X_train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)
    
    print("Data has been saved to CSV files.")

