import nltk
nltk.download('punkt')

from nltk.tokenize import sent_tokenize

def get_sentences_from_file(file_path):
    with open(file_path, 'r') as file:
        paragraph = file.read()
    sentences = sent_tokenize(paragraph)
    return sentences

# Example usage:
file_path = 'paragraphs.txt'  # Replace with the path to your text file
sentences = get_sentences_from_file(file_path)

for idx, sentence in enumerate(sentences, 1):
    print(sentence)