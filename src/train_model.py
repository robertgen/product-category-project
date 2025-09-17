import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# 1. Load dataset
df = pd.read_csv("product-category-project/data/products.csv")

df.columns = df.columns.str.lower().str.replace(" ", "_").str.strip("_")
df = df.dropna()
# 2. Drop rows with missing product_title or category_label
df = df.dropna(subset=["product_title", "category_label"])

# 3. Normalize category_label
df['category_label'] = df['category_label'].astype(str).str.lower().str.strip()

# 4. Features and label
X = df[["product_title"]]  
y = df["category_label"]

# 5. Preprocessor: TF-IDF doar pe product_title
preprocessor = ColumnTransformer(
    transformers=[
        ("title_tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1,2)), "product_title")
    ]
)

# 6. Pipeline cu RandomForestClassifier
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
])

# 7. Train model on entire dataset
pipeline.fit(X, y)

# calea către folderul model
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, "../model")
os.makedirs(model_dir, exist_ok=True)

# calea completă a fișierului
model_path = os.path.join(model_dir, "sentiment_model.pkl")

joblib.dump(pipeline, model_path)

print(f"Model antrenat și salvat la: {model_path}")

print("Model trained and saved as '../model/sentiment_model.pkl'")