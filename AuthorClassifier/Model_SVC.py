import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
#                     color="white" if cm[i, j] > thresh else "black")
                     color = "white" if i != j else "black")

#                     color="white")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color = "white" if i != j else "black")
#                     color="white" if cm[i, j] > thresh else "black")
#                    color = "white")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

# Read the CSV file into a DataFrame
sentences = pd.read_csv('grand_sentences.csv',encoding='latin-1')

# Split the data into features (X) and labels (y)
X = sentences['Sentence']
#y = sentences['Source']
y = sentences['Author']
X = X.replace(np.nan, '')
#y.replace(np.nan, '')
# Check class distribution
#print(sentences['Source'].value_counts())
print(sentences['Author'].value_counts())

target_names = sentences['Author'].unique()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the sentences into TF-IDF vectors
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Calculate class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

# Train an SVC model
svc_model = SVC(kernel='linear', C=1.0, class_weight=class_weight_dict)
#svc_model = SVC(kernel='linear', C=1.0)
svc_model.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = svc_model.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'SVC (C=1.0) Accuracy: {accuracy * 100:.2f}%')

from sklearn import metrics

#create confusion matrix
c_matrix = metrics.confusion_matrix(y_test, y_pred)
print(c_matrix)

plot_confusion_matrix(c_matrix, target_names, title='Author Confusion Matrix', cmap="plasma")

exit()
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1, 1, 10, 100]}
grid_search = GridSearchCV(SVC(kernel='linear', class_weight=class_weight_dict), param_grid, cv=5)
grid_search.fit(X_train_tfidf, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("best_params:", best_params)
print("best_model:", best_model)

# Train an SVC model
svc_model = SVC(kernel='linear', C=10, class_weight=class_weight_dict)
svc_model.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = svc_model.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'SVC (C=10) Accuracy: {accuracy * 100:.2f}%')
