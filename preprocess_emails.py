import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(directory):
    emails, labels = [], []
    for root, _, files in os.walk(directory):
        for file in files:
            with open(os.path.join(root, file), 'r', errors='ignore') as f:
                content = f.read()
                emails.append(content)
                labels.append(1 if "spam" in root.lower() else 0)
    return pd.DataFrame({"email": emails, "label": labels})

def split_data(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df['email'], df['label'], test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = load_data("data/enron")
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Training Samples: {len(X_train)} | Test Samples: {len(X_test)}")
