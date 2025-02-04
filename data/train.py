import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

# Load the preprocessed data
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv')
y_test = pd.read_csv('y_test.csv')

# Ensure y_train and y_test are 1D arrays (required for scikit-learn)
y_train = y_train.squeeze()
y_test = y_test.squeeze()

# Step 1: Encode Categorical Variables
def encode_categorical_variables(X_train, X_test):
    # Identify categorical columns
    categorical_columns = X_train.select_dtypes(include=['object']).columns.tolist()

    # If there are no categorical columns, return the original data
    if not categorical_columns:
        return X_train, X_test

    # Initialize OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, drop='first')

    # Fit and transform the training data
    encoded_train = encoder.fit_transform(X_train[categorical_columns])
    encoded_train_df = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(categorical_columns))

    # Transform the test data
    encoded_test = encoder.transform(X_test[categorical_columns])
    encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(categorical_columns))

    # Drop the original categorical columns and concatenate the encoded ones
    X_train = X_train.drop(columns=categorical_columns).reset_index(drop=True)
    X_test = X_test.drop(columns=categorical_columns).reset_index(drop=True)
    X_train = pd.concat([X_train, encoded_train_df], axis=1)
    X_test = pd.concat([X_test, encoded_test_df], axis=1)

    return X_train, X_test

# Step 2: Train a Random Forest Regressor
def train_model(X_train, y_train):
    # Initialize the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    return model

# Step 3: Evaluate the Model
def evaluate_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")

# Step 4: Create a Function to Predict FG%
def predict_fg_percentage(model, player_id, shot_distance, loc_x, loc_y, def_rating, shot_zone_area, shot_difficulty, defender_impact):
    # Create a DataFrame with the input parameters
    input_data = pd.DataFrame({
        'GAME_ID': [0],  # Placeholder, not used in prediction
        'PLAYER_ID_x': [player_id],
        'SHOT_DISTANCE': [shot_distance],
        'LOC_X': [loc_x],
        'LOC_Y': [loc_y],
        'DEF_RATING': [def_rating],
        'SHOT_ZONE_AREA_Center(C)': [1 if shot_zone_area == 'Center(C)' else 0],
        'SHOT_ZONE_AREA_Left Side Center(LC)': [1 if shot_zone_area == 'Left Side Center(LC)' else 0],
        'SHOT_ZONE_AREA_Left Side(L)': [1 if shot_zone_area == 'Left Side(L)' else 0],
        'SHOT_ZONE_AREA_Right Side Center(RC)': [1 if shot_zone_area == 'Right Side Center(RC)' else 0],
        'SHOT_ZONE_AREA_Right Side(R)': [1 if shot_zone_area == 'Right Side(R)' else 0],
        'SHOT_DIFFICULTY': [shot_difficulty],
        'DEFENDER_IMPACT': [defender_impact]
    })

    # Ensure the input data has the same columns as the training data
    input_data = input_data.reindex(columns=X_train.columns, fill_value=0)

    # Predict FG%
    fg_percentage = model.predict(input_data)[0]
    return fg_percentage

# Main Workflow
if __name__ == "__main__":
    # Step 1: Encode Categorical Variables
    X_train, X_test = encode_categorical_variables(X_train, X_test)

    # Step 2: Train the Model
    model = train_model(X_train, y_train)

    # Step 3: Evaluate the Model
    evaluate_model(model, X_test, y_test)

    # Step 4: Example Usage of the Prediction Function
    player_id = 201939  # Example: LeBron James
    shot_distance = 25  # Example: 15 feet
    loc_x = -71  # Example: X-coordinate
    loc_y = 243  # Example: Y-coordinate
    def_rating = 105  # Example: Defender's defensive rating
    shot_zone_area = 'Center(C)'  # Example: Shot zone area
    shot_difficulty = 0.5  # Example: Shot difficulty score
    defender_impact = 0.7  # Example: Defender impact score

    predicted_fg_percentage = predict_fg_percentage(
        model, player_id, shot_distance, loc_x, loc_y, def_rating, shot_zone_area, shot_difficulty, defender_impact
    )
    print(f"Predicted FG%: {predicted_fg_percentage * 100:.2f}%")