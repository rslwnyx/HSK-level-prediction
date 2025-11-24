import pandas as pd
import numpy as np
import joblib
import scipy.sparse as sp
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from src.utils import clean_chinese_text, load_hsk_data, extract_hsk_features, hsk_tokenizer

col_names = ["composition_id", "nationality", "gender", "exam_date",
              "composition_title", "speaking_test_score", "composition_score",
              "listening_score", "reading_score", "general_score", "total_score", 
              "certificate", "sample_sentence"]
def main():

    cache_dir = 'cache'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    cache_X = os.path.join(cache_dir, 'X_final_weighted.pkl')
    cache_y = os.path.join(cache_dir, 'y.pkl')
    cache_tfidf = os.path.join(cache_dir, 'tfidf_fitted.pkl')



    if os.path.exists(cache_X) and os.path.exists(cache_y) and os.path.exists(cache_tfidf):
        X_final = joblib.load(cache_X)
        y_hsk_level = joblib.load(cache_y)
        tfidf = joblib.load(cache_tfidf)
    
        word_dict, grammar_patterns = load_hsk_data(r"data\hsk_data.csv")
        
    else:
        print("Cache not found. Starting heavy calculation (Regex & Tokenization)...")
        
        # Load Raw Data
        try:
            df = pd.read_csv(r"data\hsk_composition.csv")
            df.columns = col_names
            word_dict, grammar_patterns = load_hsk_data(r"data\hsk_data.csv")
        except FileNotFoundError:
            print("Error: Data files not found.")
            return

    

        df["clean_text"] = df["sample_sentence"].apply(clean_chinese_text)
        df = df.dropna(subset=["total_score", "clean_text"])


        X_stats = df['clean_text'].apply(
            lambda x: extract_hsk_features(x, word_dict, grammar_patterns)
        ).tolist()
        X_stats = np.array(X_stats)


        tfidf = TfidfVectorizer(tokenizer=hsk_tokenizer, max_features=3000, token_pattern=None)
        X_tfidf = tfidf.fit_transform(df['clean_text'])


        X_final = sp.hstack((X_tfidf, X_stats))
        final_thresholds = np.array([0.0, 150.0, 220.0, 290.0, 360.0, 430.0, 500.0])
        y_hsk_level = np.digitize(df["total_score"].values, bins=final_thresholds[1:-1])

        joblib.dump(X_final, cache_X, compress=3)
        joblib.dump(y_hsk_level, cache_y, compress=3)
        joblib.dump(tfidf, cache_tfidf, compress=3)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_final, y_hsk_level, test_size=0.2, random_state=42)

    models = {"Logistic Regression": LogisticRegression(max_iter=1000),
              "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42),
              "XGBoost": XGBClassifier(n_estimators = 60,max_depth = 6,   learning_rate = 0.1, n_jobs = -1, random_state = 42)
    }

    best_model = None
    best_acc = 0
    best_name = ""

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')

        print(f"\n{name} \nResults: Accuracy = {accuracy:.2f}\nMacro F1 = {f1:.2f}\n")
        
        if accuracy > best_acc:
            best_acc = accuracy
            best_model = model
            best_name = name
    
    print(f"Best Model: {best_name}")
    

    #Artifact saving
    joblib.dump(best_model, r"models/hsk_regressor.pkl", compress=3)
    joblib.dump(tfidf, r"models/tfidf_vectorizer.pkl", compress=3)
    joblib.dump((word_dict, grammar_patterns), r"models/hsk_assets.pkl", compress=3)


if __name__ == "__main__":
    main()