from utils.data_preprocessing import load_data, get_embedding_matrix, load_glove_embeddings
from models.cnn_model import create_cnn_model
from models.lstm_model import create_lstm_model
from models.lstm_cnn_model import create_lstm_cnn_model
from utils.train import train_model
from utils.evaluate import evaluate_model

BASE_PATH = 'aclImdb'  # Base path to your dataset
GLOVE_FILE_PATH = 'aclImdb/glove.6B.100d.txt'  # Path to your GloVe embeddings file
MAX_WORDS = 8000
MAX_LEN = 500
EMBEDDING_DIM = 100

# Load data and embeddings
X_train, X_test, y_train, y_test, word_index = load_data(BASE_PATH, MAX_WORDS, MAX_LEN)
embeddings_index = load_glove_embeddings(GLOVE_FILE_PATH, EMBEDDING_DIM)
embedding_matrix = get_embedding_matrix(word_index, embeddings_index, MAX_WORDS, EMBEDDING_DIM)

# Select model type: CNN, LSTM, LSTM+CNN
print("Which model? (Pick number according to choice)\n1. CNN\n2. LSTM\n3. LSTM+CNN")
while(1):
    model_type = int(input("Enter your choice: "))
    if model_type in [1, 2, 3]:
        break

if model_type == 1:
    model = create_cnn_model(MAX_LEN, MAX_WORDS, EMBEDDING_DIM, embedding_matrix)
        
elif model_type == 2:
    model = create_lstm_model(MAX_LEN, MAX_WORDS, EMBEDDING_DIM, embedding_matrix)
    
elif model_type == 3:
    model = create_lstm_cnn_model(MAX_LEN, MAX_WORDS, EMBEDDING_DIM, embedding_matrix)

# Train and evaluate the model
history = train_model(model, X_train, y_train, X_test, y_test)
accuracy, cm, report = evaluate_model(model, X_test, y_test)

print(f'Accuracy: {accuracy}\n')
print(f'Confusion Matrix:\n{cm}\n')
print(f'Classification Report:\n{report}')
