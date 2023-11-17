import os
import pandas as pd

def break_into_sentences(text):
    sentences = text.replace('\n', ' ').split('. ')
    sentences = [sentence + '.' for sentence in sentences]
    return sentences

authors = ["Jane Austen", "D H Lawrence", "Ernest Hemingway", "William Faulkner",
           "Edith Wharton", "George Orwell", "Jack Kerouac", "F. Scott Fitzgerald"]

main_directory = 'texts'  # Replace 'texts' with your main directory name

data = {'Author': [], 'Sentence': []}

for author in authors:
    author_directory = os.path.join(main_directory, author)
    for subdir, _, files in os.walk(author_directory):
        for file in files:
            file_path = os.path.join(subdir, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read(75000)
                    text = text.replace('\n', '')
                    sentences = break_into_sentences(text)
                    last_name = author.split()[-1]  # Extracting the last name of the author
                    for sentence in sentences:
                        data['Author'].append(last_name)
                        data['Sentence'].append(sentence)
            except UnicodeDecodeError:
                pass

df = pd.DataFrame(data)
print(df.head())  # Displaying the first few rows of the DataFrame
print(df.tail())  # Displaying the last few rows of the DataFrame
df.to_csv("grand_sentences.csv")