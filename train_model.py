import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

DATA_PATH = "bank (1).csv"
MODEL_PATH = "model.joblib"


def load_and_clean_data():
    df = pd.read_csv(DATA_PATH)

    df["balance"] = pd.to_numeric(df["balance"], errors="coerce")
    df["balance"] = df["balance"].fillna(df["balance"].median())

    df["marital"] = df["marital"].replace(["EVLI", "singel"], df["marital"].mode()[0])
    df["age"] = df["age"].fillna(df["age"].median())
    df["education"] = df["education"].fillna(df["education"].mode()[0])
    df["housing"] = df["housing"].fillna(df["housing"].mode()[0])

    df["job"] = df["job"].replace("unknown", df["job"].mode()[0])
    df["education"] = df["education"].replace("unknown", df["education"].mode()[0])
    df["poutcome"] = df["poutcome"].replace("unknown", "Ilk_Kez_Araniyor")

    return df


def main():
    df = load_and_clean_data()

    y = df["deposit"].replace({"yes": 1, "no": 0})
    X = df.drop(columns=["deposit"])
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {acc:.3f}")

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
