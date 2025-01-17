from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle
import preprocess_emails

def train_model(X_train, y_train):
    vectorizer = CountVectorizer(stop_words='english', max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    with open('spam_model.pkl', 'wb') as f:
        pickle.dump((vectorizer, model), f)
    print("Model trained and saved.")

if __name__ == "__main__":
    df = preprocess_emails.load_data("data/enron")
    X_train, X_test, y_train, y_test = preprocess_emails.split_data(df)
    train_model(X_train, y_train)

    # Test the model
    with open('spam_model.pkl', 'rb') as f:
        vectorizer, model = pickle.load(f)
    X_test_vec = vectorizer.transform(X_test)
    predictions = model.predict(X_test_vec)
    print(f"Accuracy: {accuracy_score(y_test, predictions)}")
