from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import urllib.request
import tarfile
import os

# Step 1: Download IMDB dataset (large but widely used)
dataset_url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
dataset_path = "/content/aclImdb"

if not os.path.exists(dataset_path):
    print("Downloading dataset... This may take some minutes.")
    urllib.request.urlretrieve(dataset_url, "aclImdb_v1.tar.gz")
    print("Extracting dataset...")
    tar = tarfile.open("aclImdb_v1.tar.gz", "r:gz")
    tar.extractall()
    tar.close()
    print("Done!")

# Step 2: Load data from folders
print("Loading dataset...")
data = load_files(dataset_path + '/train/', categories=['pos', 'neg'])
X, y = data.data, data.target

# Step 3: Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 5: Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Step 6: Evaluate model
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred, target_names=['negative', 'positive']))

# Step 7: Real-time input
print("\nModel ready for live testing!")
while True:
    review = input("\nEnter your product review (or 'exit' to quit): ")
    if review.lower() == 'exit':
        break
    review_vec = vectorizer.transform([review.encode('utf-8')])
    pred = model.predict(review_vec)[0]
    sentiment = "POSITIVE" if pred == 1 else "NEGATIVE"
    print("💬 Sentiment:", sentiment)
