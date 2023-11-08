import os
import sys
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import datetime
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

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
nltk.download('stopwords')
nltk.download('punkt')

# Get a list of all the files and directories in the current directory
files = os.listdir()

texts_dir = "texts/"

books = {}
#contents = []

# Find directories in "texts/"
for root, dirs, files in os.walk(texts_dir):
    # for each directory
    for directory in dirs:
        print(directory)
        # get the files in each directory
        contents = ""
        for _, _, files in os.walk(texts_dir + directory):
            print(files)
            for file in files:
                print("Processing:", file)
                if (file == ".DS_Store"):
                    continue
                # Open the file in read mode
                #with codecs.open(texts_dir + directory + '/' + file, "r", unicode_escape=False) as f:
                with open(texts_dir + directory + '/' + file, "r") as f:
                    # Read the contents of the file
                    contents += (f.read().replace('\n', ' ')
                                    .replace('\u2014', " - ")) +' '
            books.update({directory : contents})

#print(sys.getsizeof(books))

#with open("/Users/wel51x/Documents/" + "books.json", "w") as f:
#    json.dump(books, f)

preprocess_data(books)

# Assuming tfidf_matrix is your feature matrix and authors is your list of corresponding author labels
# Extracting X and y from the 'books' dictionary
X = list(books.values())  # List of TF-IDF matrices
y = list(books.keys())  # List of authors (labels)

# Split the data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = {}, {}, {}, {}
for author, tfidf_matrix in books.items():
    print("Splitting:", author, "at:", datetime.datetime.now().strftime("%H:%M:%S"))
    # blows up
    X_train[author], X_test[author], y_train[author], y_test[author] = train_test_split(tfidf_matrix,
                                                                                        [author] * len(tfidf_matrix),
                                                                                        test_size=0.2,
                                                                                        random_state=42)

print("Finished split at:", datetime.datetime.now().strftime("%H:%M:%S"))

# Create and train the classifier
#classifier = SVC()
#classifier.fit(X_train, y_train)

#print("Finished fit at:", datetime.datetime.now().strftime("%H:%M:%S"))

# Evaluate the classifier
#accuracy = classifier.score(X_test, y_test)
#print("Accuracy:", accuracy)

models = {}
for author in books.keys():
    model = SVC()
    print("Fitting:", author, "at:", datetime.datetime.now().strftime("%H:%M:%S"))
    model.fit(X_train[author], y_train[author])
    accuracy = model.score(X_test[author], y_test[author])
    models[author] = {'model': model, 'accuracy': accuracy}

    print(f"Author: {author}, Accuracy: {accuracy}")

print("Finished at:", datetime.datetime.now().strftime("%H:%M:%S"))
