import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import os

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Tokenization
    words = word_tokenize(text.lower())

    # Remove punctuation
    words = [word for word in words if word not in string.punctuation]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    return ' '.join(words)

# Define topics and document numbers
topics = ['Fashion', 'Disease', 'Sports']
num_documents_per_topic = 15

# Directory containing the original documents
documents_directory = "C:\\Users\\DELL\\Desktop\\GTProject\\Document"

# Directory to save preprocessed documents
preprocessed_directory = "C:\\Users\\DELL\\Desktop\\GTProject\\Preprocessed"

# Loop through each topic and document number
for topic in topics:
    for doc_num in range(1, num_documents_per_topic + 1):
        input_file_path = os.path.join(documents_directory, topic, f'File{doc_num}.txt')
        output_file_path = os.path.join(preprocessed_directory, topic, f'File{doc_num}.txt')

        # Check if the input file exists
        if os.path.exists(input_file_path):
            # Read the original file with 'utf-8' encoding
            with open(input_file_path, 'r', encoding='utf-8') as file:
                text = file.read()

            # Preprocess the text
            preprocessed_text = preprocess_text(text)

            # Create directories if they don't exist
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

            # Save preprocessed text to a new file
            with open(output_file_path, 'w', encoding='utf-8') as file:
                file.write(preprocessed_text)

            print(f"Preprocessed and saved {output_file_path}")
        else:
            print(f"File not found: {input_file_path}")
