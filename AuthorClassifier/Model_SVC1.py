import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import datetime
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import preprocessing

def preprocess_data(books):
    for author, text in books.items():
        print("Preprocessing:", author, "at:", datetime.datetime.now().strftime("%H:%M:%S"))
        # Tokenize & Normalize
        tokens = word_tokenize(text.lower())  # Convert to lowercase during tokenization
        # Remove Stopwords
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word not in stop_words]
        # Feature Extraction (TF-IDF)
        #text_strings = [tokens + ' ' for tokens in filtered_tokens]
        # TF-IDF vectorization
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(filtered_tokens)
        books[author] = tfidf_matrix

# Download NLTK resources for stopwords (run this line once)
#nltk.download('stopwords')
#nltk.download('punkt')

#with open("/Users/wel51x/Documents/" + "books.json", "w") as f:
#    json.dump(books, f)

sentences = pd.read_csv("sentences.csv")
# Assuming tfidf_matrix is your feature matrix and authors is your list of corresponding author labels
# Extracting X and y from the 'books' dictionary
X = list(sentences.Sentence)  # List of TF-IDF matrices
le = preprocessing.LabelEncoder()
X = le.fit_transform(X)
X = X.reshape(-1, 1)
y = list(sentences.Source)  # List of authors (labels)
y = le.fit_transform(y)

# Split the data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,  random_state=42)

print("Finished split at:", datetime.datetime.now().strftime("%H:%M:%S"))

# Create and train the classifier
classifier = SVC()
classifier.fit(X_train, y_train)

print("Finished fit at:", datetime.datetime.now().strftime("%H:%M:%S"))

# Evaluate the classifier
accuracy = classifier.score(X_test, y_test)
print("Accuracy:", accuracy)

print("Finished at:", datetime.datetime.now().strftime("%H:%M:%S"))
