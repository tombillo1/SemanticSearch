from flask import Flask, render_template, request
import openai
import pandas as pd
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from openai.embeddings_utils import get_embedding
from config import API_KEY
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

        filename = file.filename
        return {"text": text, "filename": filename}
    finally:
        os.remove(file_path)

openai.api_key = API_KEY

app = Flask(__name__, static_url_path='/static', static_folder='static')

def document_to_dataframe(document):
    sentences = sent_tokenize(document)
    sentences = [sentence.replace('\n', '').replace('\t', '') for sentence in sentences]
    df = pd.DataFrame(sentences, columns=['Sentence'])
    return df

def compute_embeddings(df, filenames):
    for filename in filenames:
        file_df = df.loc[df['Filename'] == filename]
        embeddings = []
        for sentence in file_df['Sentence']:
            embedding = get_embedding(sentence, engine='text-embedding-ada-002')
            embeddings.append(embedding)
        file_df['embedding'] = embeddings
        save_embeddings(file_df, f"{filename}_embeddings.pkl")

def load_embeddings(filename):
    df = pd.read_pickle(filename)
    return df

def save_embeddings(df, filename):
    df.to_pickle(filename)

@app.route('/')
def index():
    results = ["Result 1", "Result 2", "Result 3"]
    return render_template("results.html", results=results)

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        query = request.form['query']
        files = request.files.getlist('files')
        df = pd.DataFrame(columns=['Sentence', 'Filename'])

        for file in files:
            file_text = extract_text(file)
            text = file_text["text"]
            filename = file_text["filename"]
            file_df = document_to_dataframe(text)
            file_df['Filename'] = filename
            df = pd.concat([df, file_df], ignore_index=True)

        filenames = df['Filename'].unique()

        for filename in filenames:
            file_df = df.loc[df['Filename'] == filename]
            if os.path.exists(f"{filename}_embeddings.pkl"):
                file_df = load_embeddings(f"{filename}_embeddings.pkl")
            else:
                compute_embeddings(file_df, [filename])

        # Combine all embeddings
        df = pd.concat([load_embeddings(f"{filename}_embeddings.pkl") for filename in filenames])

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
