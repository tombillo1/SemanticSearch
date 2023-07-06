from flask import Flask, render_template, request
import openai
import pandas as pd
import nltk
nltk.download('punkt')
from config import API_KEY
from nltk.tokenize import sent_tokenize
from openai.embeddings_utils import get_embedding
import tempfile
import os
import PyPDF2
import faiss
import numpy as np

def extract_text(file):
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        file.save(temp.name)
        file_path = temp.name
    try:
        if file.filename.lower().endswith('.pdf'):
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
        else:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
    finally:
        os.remove(file_path)

    return text

openai.api_key = API_KEY

app = Flask(__name__, static_url_path='/static', static_folder='static')

def document_to_dataframe(document):
    sentences = sent_tokenize(document)
    sentences = [sentence.replace('\n', '').replace('\t', '') for sentence in sentences]
    df = pd.DataFrame(sentences, columns=['Sentence'])
    return df

def compute_embeddings(df):
    embeddings = []
    for sentence in df['Sentence']:
        embedding = get_embedding(sentence, engine='text-embedding-ada-002')
        embeddings.append(embedding)
    df['embedding'] = embeddings

def load_embeddings():
    df = pd.read_pickle('embeddings.pkl')
    return df

def save_embeddings(df):
    df.to_pickle('embeddings.pkl')

@app.route('/')
def index():
    results = ["Result 1", "Result 2", "Result 3"]
    return render_template("results.html", results=results)

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        query = request.form['query']
        files = request.files.getlist('files')
        df = pd.DataFrame(columns=['Sentence'])

        for file in files:
            text = extract_text(file)
            file_df = document_to_dataframe(text)
            df = pd.concat([df, file_df], ignore_index=True)

        # Check if embeddings are already computed and stored
        if os.path.exists('embeddings.pkl'):
            df = load_embeddings()
        else:
            compute_embeddings(df)
            save_embeddings(df)

        df['embedding'] = df['embedding'].apply(lambda x: np.array(x))  # Convert embeddings to NumPy arrays

        d = df['embedding'][0].shape[0]  # Get the dimension of the embeddings
        index = faiss.IndexFlatL2(d)

        embeddings = np.vstack(df['embedding'].to_numpy())
        index.add(embeddings)

        query_embedding = get_embedding(query, engine='text-embedding-ada-002')
        query_embedding = np.array(query_embedding).reshape(1, -1)

        _, indices = index.search(query_embedding, k=5)
        top_3_items = df.iloc[indices[0]]
        results = top_3_items['Sentence'].tolist()

        return render_template('results.html', query=query, results=results)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
