import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load your dataset (assuming it's a CSV file)
df = pd.read_csv("data/warriors_player_shots_with_defense.csv")
# df.columns = df.columns.str.strip()  # Removes leading/trailing spaces from column names
# df["SHOT_ZONE_BASIC"] = df["SHOT_ZONE_BASIC"].astype(str)

# print(df.head())

# # --- Create Derived Features ---
# Compute FG% per shot zone as a reference
shot_zone_fg = df.groupby("SHOT_ZONE_BASIC")["SHOT_MADE_FLAG"].mean().rename("SHOT_ZONE_FG")
df = df.merge(shot_zone_fg, on="SHOT_ZONE_BASIC", how="left")

# --- One-Hot Encode Categorical Variables ---
categorical_features = ["SHOT_ZONE_BASIC", "SHOT_ZONE_AREA", "SHOT_ZONE_RANGE"]
one_hot_encoder = OneHotEncoder(sparse_output=False, drop="first")
encoded_features = one_hot_encoder.fit_transform(df[categorical_features])
encoded_df = pd.DataFrame(encoded_features, columns=one_hot_encoder.get_feature_names_out())

df = pd.concat([df, encoded_df], axis=1)
df.drop(columns=categorical_features, inplace=True)

# --- Normalize Numerical Features ---
numerical_features = ["SHOT_DISTANCE", "LOC_X", "LOC_Y"]
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Interaction term: shot distance & shot attempt
df["SHOT_DIST_ATTEMPTED"] = df["SHOT_DISTANCE"] * df["SHOT_ATTEMPTED_FLAG"]

# Save the processed dataset
df.to_csv("processed_data.csv", index=False)
