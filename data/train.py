import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

class FGPercentageModel:
    def __init__(self, data_path):
        # Load data
        self.df = pd.read_csv(data_path)

        # Define features and target
        self.features = ["SHOT_ZONE_BASIC", "SHOT_ZONE_AREA", "SHOT_ZONE_RANGE", "PLAYER_NAME", "SHOT_DISTANCE"]
        self.target = "SHOT_MADE_FLAG"

        # Create preprocessing pipeline
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), ["SHOT_DISTANCE"]),
                ("cat", OneHotEncoder(handle_unknown="ignore"), ["SHOT_ZONE_BASIC", "SHOT_ZONE_AREA", "SHOT_ZONE_RANGE", "PLAYER_NAME"]),
            ]
        )

        # Define model pipeline
        self.model = Pipeline([
            ("preprocessor", self.preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
        ])

    def train(self):
        # Prepare data
        X = self.df[self.features]
        y = self.df[self.target]

        # Train model
        self.model.fit(X, y)

    def predict(self, player_name, shot_zone_basic, shot_zone_area, shot_zone_range, shot_distance, def_rating=None):
        # Prepare input data
        input_data = pd.DataFrame([[player_name, shot_zone_basic, shot_zone_area, shot_zone_range, shot_distance]], 
                                  columns=self.features)

        # Predict FG%
        fg_percentage = self.model.predict(input_data)[0]

        # If def_rating is provided, adjust FG%, otherwise return the raw FG%
        if def_rating is not None:
            adjusted_percentage = self.adjusted_fg_percentage(fg_percentage, def_rating)
            return round(adjusted_percentage * 100, 2)  # Return as percentage (0-100 scale)
        else:
            return round(fg_percentage * 100, 2)  # Return as percentage (0-100 scale)

    def adjusted_fg_percentage(self, fg_percentage, defender_drting):
        # Normalize DRtg to be in a reasonable range (assuming 90-120 is typical for DRtg)
        defender_multiplier = 1- (120 - defender_drting) / 100
        
        # Adjust the player's FG% based on the defender's rating
        adjusted_fg = fg_percentage * defender_multiplier
        
        # Ensure the adjusted FG% is between 0 and 1
        adjusted_fg = max(0, min(1, adjusted_fg))
        
        return adjusted_fg



# Example usage
fg_model = FGPercentageModel("warriors_player_shots_with_defense.csv")
fg_model.train()
print(fg_model.predict("Stephen Curry", "Above the Break 3", "Right Side Center(RC)", "24+ ft.", 26, 102.9))
