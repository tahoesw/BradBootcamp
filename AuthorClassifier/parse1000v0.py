authors = ["Jane Austen", "D H Lawrence", "Ernest Hemingway", "William Faulkner",
           "Edith Wharton", "George Orwell", "Jack Kerouac", "F. Scott Fitzgerald", ]

def break_into_sentences(text):
    sentences = text.replace('\n', ' ').split('. ')
    # Add the period back to each sentence
    sentences = [sentence + '. ' for sentence in sentences]
    return sentences

file_path = "texts/D H Lawrence/D H Lawrence+Lady Chatterley's Lover.txt"  # Replace 'your_file.txt' with the file path

# Read the first 10,000 characters from the file
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read(10000)

# Replace newline characters with blanks
text = text.replace('\n', '')

# Break the text into sentences
sentences = break_into_sentences(text)

# Print each sentence
for sentence in sentences:
    print(sentence)