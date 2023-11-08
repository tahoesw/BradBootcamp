import requests

def download_book(book_id, file_path):
    url = f'http://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt'
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(response.text)
        print(f"Book {book_id} downloaded successfully and saved to {file_path}")
    else:
        print(f"Failed to download book {book_id}")

Texts = [
["Jane Austen", "Pride and Prejudice", 1342, "Sense and Sensibility", 161],
["Jane Austen", "Emma", 158, "Mansfield Park", 141],
["John Dos Passos", "Three Soldiers", 1322, "Manhattan Transfer", 7268],
["John Dos Passos", "The 42nd Parallel", 1323, "One Man's Initiation - 1917", 3776],
["John Steinbeck", "The Grapes of Wrath", 1060, "Of Mice and Men", 829],
["John Steinbeck", "East of Eden", 1156, "Travels with Charley in Search of America", 28800],
["Ernest Hemingway", "The Old Man and the Sea", 3325, "A Farewell to Arms", 144],
["Ernest Hemingway", "For Whom the Bell Tolls", 4800, "The Sun Also Rises", 145],
["William Faulkner", "The Sound and the Fury", 9948, "As I Lay Dying", 3030],
["William Faulkner", "Light in August", 1639, "Absalom, Absalom!", 24572],
["Sinclair Lewis", "Main Street", 143, "Babbitt", 1156],
["Sinclair Lewis", "Elmer Gantry", 952, "Arrowsmith", 1076],
["Willa Cather", "My Ántonia", 242, "O Pioneers!", 24],
["Willa Cather", "Death Comes for the Archbishop", 2426, "The Song of the Lark", 3526]]

# Example usage
if __name__ == "__main__":
    for item in Texts:
        print(item)
        file_path = "texts/" + item[0] + '+' + item[1] + ".txt"
        print(file_path, item[2])
        download_book(item[2], file_path)
        file_path = "texts/" + item[0] + '+' + item[3] + ".txt"
        print(file_path, item[4])
        download_book(item[4], file_path)
#        break
