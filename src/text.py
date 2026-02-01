def add_features(df):
    df = df.copy()

    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 30, 40, 120],
        labels=[0, 1, 2]
    ).astype(int)

    return df

import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


MODEL_FILE = "over_30_model.joblib"


def train_and_save_model():
    # Load data
    df = pd.read_csv("ml_ready_people.csv")

    # Add new feature: age_group
    df = add_features(df)

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("city", OneHotEncoder(), ["city"]),
            ("numeric", "passthrough", ["age", "age_group"])
        ]
    )

    model = Pipeline(steps=[
        ("prep", preprocessor),
        ("clf", LogisticRegression())
    ])

    # Train
    model.fit(X, y)

    # Save model
    joblib.dump(model, MODEL_FILE)
    print(f"Model saved to {MODEL_FILE} (with age_group)")


def load_and_predict():
    # Load saved model
    model = joblib.load(MODEL_FILE)

    # New data
   new_people = add_features(new_people)

    predictions = model.predict(new_people)
    probabilities = model.predict_proba(new_people)

    for i, person in new_people.iterrows():
    	prob_over_30 = probabilities[i][1]

    print(
        f"Person {i + 1}: "
        f"age={person['age']}, city={person['city']} → "
        f"{prob_over_30:.2f} probability of being over 30"
    	)
    THRESHOLD = 0.6
    decisions = (probabilities[:, 1] >= THRESHOLD).astype(int)

    print("\nDecisions with threshold =", THRESHOLD)
    print(decisions)
	

if __name__ == "__main__":
    train_and_save_model()
    load_and_predict()