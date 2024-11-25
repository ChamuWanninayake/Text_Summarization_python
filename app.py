from flask import Flask, render_template, request, jsonify, redirect, url_for
import pickle
import gensim
from gensim import corpora
from transformers import pipeline
import traceback
from helper import preprocessing, vectorizer, get_prediction
from werkzeug.utils import secure_filename
from transformers import BartForConditionalGeneration, BartTokenizer
import os

import spacy
import nltk
from rake_nltk import Rake
from collections import Counter
import logging

# Initialize Flask application
app = Flask(__name__)


#Topic Modelling
# Load the LDA model and dictionary
with open('static/model/lda_small_DB.pickle', 'rb') as f:
    lda_model = pickle.load(f)

with open('static/model/dictionary_small_DB.pickle', 'rb') as f:
    id2word = pickle.load(f)

# Initialize the text generation pipeline
text_generator = pipeline("text-generation", model="gpt2")


@app.route('/')
def index():
    return render_template('index.html')


#Text summarization
# Set the upload folder and allowed file types
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the fine-tuned BART model and tokenizer
model_path = "C:\\Users\\Chamuditha\\Desktop\\sayumi\\Sayumi\\Bart-Samsum\\my_model"
tokenizer = BartTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path)

# Check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


#Keyword Extraction
# Configure logging
logging.basicConfig(level=logging.DEBUG)  # Set logging level to DEBUG

# Load the Spacy model and download necessary NLTK data
nlp = spacy.load("en_core_web_sm")
nltk.download('punkt')

# Define the models
rake = Rake()

# Function for RAKE keyword extraction
def extract_keywords_rake(text):
    rake.extract_keywords_from_text(text)
    return rake.get_ranked_phrases_with_scores()

# Function for SpaCy keyword extraction
def extract_keywords_spacy(text):
    doc = nlp(text)
    keywords = [chunk.text for chunk in doc.noun_chunks if chunk.root.pos_ in ['NOUN', 'PROPN']]
    return Counter(keywords)

# Combined Keyword Extraction
def combined_keyword_extraction(text):
    rake_keywords = extract_keywords_rake(text)
    spacy_keywords = extract_keywords_spacy(text)
    
    combined_keywords = Counter()

    # Add RAKE keywords to the counter (with their original scores)
    for score, phrase in rake_keywords:
        combined_keywords[phrase] += score

    # Add SpaCy keywords to the counter (with adjusted weight)
    for word, count in spacy_keywords.items():
        combined_keywords[word] += count * 0.5  # Give less weight to SpaCy keywords

    # Extract the top 10 keywords, but only return the phrases (ignore scores)
    top_keywords = [keyword for keyword, _ in combined_keywords.most_common(10)]
    
    return top_keywords





# Route for summarizing text from a .txt file
@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the file is part of the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    # If no file is selected
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # If the file is allowed
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Read the contents of the .txt file
        with open(file_path, 'r', encoding='utf-8') as f:
            text_input = f.read()

        # Tokenize and summarize the text
        inputs = tokenizer([text_input], max_length=2048, return_tensors='pt', truncation=True)
        summary_ids = model.generate(
            inputs['input_ids'], 
            max_length=300,  
            min_length=50,   
            length_penalty=2.0, 
            num_beams=4, 
            early_stopping=True
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Delete the file after processing to save space
        os.remove(file_path)

        return jsonify({'summary': summary})
    else:
        return jsonify({'error': 'File type not allowed. Only .txt files are accepted.'}), 400


# Route for summarizing text from input form
@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    text_input = data.get('text_input', '')

    if not text_input.strip():  # Check if the input is empty or contains only spaces
        return jsonify({'error': 'Please enter some text to summarize.'}), 400

    # Tokenize and summarize the input text
    inputs = tokenizer([text_input], max_length=2048, return_tensors='pt', truncation=True)
    summary_ids = model.generate(
        inputs['input_ids'], 
        max_length=300,  
        min_length=50,   
        length_penalty=2.0, 
        num_beams=4, 
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return jsonify({'summary': summary})

@app.route('/summarize_txt')
def summarize_txt():
    return render_template('summarize_txt.html')




#Route for Sentiment Analysis
@app.route('/sentiment', methods=['POST'])
def analyze_sentiment():
    text = request.json.get('text_input', '')
    
    preprocessed_txt = preprocessing(text)
    
    vectorized_txt = vectorizer(preprocessed_txt)
   
    prediction = get_prediction(vectorized_txt)
   
    if prediction == 'negative':
       
        sentiment_image = 'negative_image.png'
    else:
       
        sentiment_image = 'positive_image.jpg'

   
    return jsonify({
        'review': text,
        'sentiment': prediction,
        'sentiment_image': sentiment_image
        
    })




# Route for Keyword Extraction
@app.route('/extract_keywords', methods=['POST'])
def extract_keywords():
    data = request.json
    text = data.get('text_input', '')
    
    # Log the input text
    app.logger.debug(f"Input text: {text}")
    
    # Use the combined keyword extraction function
    keywords = combined_keyword_extraction(text)
    
    # Log the output keywords
    app.logger.debug(f"Extracted keywords: {keywords}")
    
    return jsonify(keywords)




#Route for Topic Modeling
@app.route('/topic-modeling', methods=['POST'])
def topic_modeling():
    data = request.json
    text_input = data.get('text_input', '')
    
    # Preprocess the text input
    processed_text = preprocess(text_input)
    
    try:
        print(f"Input text: {text_input}")
        print(f"Processed text: {processed_text}")

        # Get the bag of words vector
        bow_vector = id2word.doc2bow(processed_text)
        print(f"Bag of Words vector: {bow_vector}")

        # Get topic distribution
        topics = lda_model.get_document_topics(bow_vector)

        # Find the topic with the highest probability
        if topics:
            top_topic_id, top_topic_prob = max(topics, key=lambda x: x[1])  # Get topic with highest probability
            print(f"Top topic: {top_topic_id}, Probability: {top_topic_prob}")

            # Get top 5 words of the highest probability topic
            top_topic_words = [word for word, _ in lda_model.show_topic(top_topic_id, topn=5)]  # Top 5 words of the topic
            print(f"Top topic words: {top_topic_words}")

            # Create a prompt for the text generation model
            prompt = f"The following topic is about: {', '.join(top_topic_words)}. "
            print(f"GPT-2 prompt: {prompt}")

            # Generate a sentence using GPT-2
            generated_sentence = prompt
            #generated_sentence = text_generator(prompt, max_length=50, num_return_sequences=1)[0]['generated_text']
            print(f"Generated sentence: {generated_sentence}")

            return jsonify({
                'top_topic': {
                    'topic_id': top_topic_id,
                    'probability': float(top_topic_prob),
                    'top_words': top_topic_words
                },
                'generated_sentence': generated_sentence
            })
        else:
            return jsonify({'error': 'No topics found for the given text'}), 500
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")  # Print error for debugging
        print(traceback.format_exc())  # This will print a full stack trace
        return jsonify({'error': str(e)}), 500  # Return error message and status code


def preprocess(text):
    # Tokenize and remove punctuation, convert to lowercase
    tokens = gensim.utils.simple_preprocess(text, deacc=True)
    print(f"Preprocessed tokens: {tokens}")
    return tokens


if __name__ == "__main__":
     # Create the uploads folder if it doesn't exist
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
