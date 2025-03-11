import streamlit as st
import torch
import nltk
import pickle
import json
import random
import torch.nn as nn
from nltk.stem import WordNetLemmatizer
nltk.data.path.append("https://fantastic-memory-7gvx5x597g7hppqq.github.dev/?folder=%2Fhome%2Fvscode%2Fnltk_data")
nltk.download('punkt_tab')
# Download NLTK resources if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model (must match the architecture of your trained model)
class ChatModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(ChatModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Load the model and data
@st.cache_resource
def load_model_and_data():
    # Load the training data
    with open('training_data.pkl', 'rb') as f:
        data = pickle.load(f)
        words = data['words']
        classes = data['classes']
    
    # Load intents
    with open('intents.json') as file:
        intents = json.load(file)
    
    # Determine input size from words length
    input_size = len(words)
    
    # Define model with the same architecture used during training
    hidden_size1 = 128
    hidden_size2 = 64
    output_size = len(classes)
    
    model = ChatModel(input_size, hidden_size1, hidden_size2, output_size)
    model.load_state_dict(torch.load('chatbot_model.pth', map_location=device))
    model.to(device)
    model.eval()
    
    return model, words, classes, intents

def clean_up_sentence(sentence):
    # Tokenize and lemmatize the sentence
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    # Generate a bag of words array
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return torch.tensor(bag, dtype=torch.float32).unsqueeze(0).to(device)

def predict_class(sentence, model, words, classes):
    # Predict the class of the sentence
    bow = bag_of_words(sentence, words)
    with torch.no_grad():
        res = model(bow)
    ERROR_THRESHOLD = 0.25
    res = res.cpu().numpy()[0]
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    # Get the response for the predicted intent
    if len(intents_list) == 0:
        # Find the fallback intent (usually the last one or one marked as fallback)
        for intent in intents_json['intents']:
            if intent.get('tag') == 'fallback' or intent.get('tag') == 'noanswer':
                return random.choice(intent['responses'])
        # If no fallback intent is found, return a default message
        return "I'm not sure I understand. Could you please rephrase?"
    
    tag = intents_list[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])

# Streamlit UI
def main():
    st.title("AI Chatbot")
    st.subheader("Ask me anything!")
    
    # Load model and data
    model, words, classes, intents = load_model_and_data()
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Input for user message
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Get bot response
        intents_list = predict_class(user_input, model, words, classes)
        bot_response = get_response(intents_list, intents)
        
        # Add bot response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
        
        # Display bot response
        with st.chat_message("assistant"):
            st.write(bot_response)
            
            # Optionally show the detected intent for debugging
            if st.sidebar.checkbox("Show detected intent", value=False):
                if intents_list:
                    st.info(f"Detected intent: {intents_list[0]['intent']} (Confidence: {float(intents_list[0]['probability']):.2f})")
                else:
                    st.info("No intent detected")

    # Sidebar options
    with st.sidebar:
        st.title("Options")
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.experimental_rerun()

if __name__ == "__main__":
    main()