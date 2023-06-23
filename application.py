from flask import Flask, render_template, request
import openai
import pandas as pd
import nltk
nltk.download('punkt')
from config import API_KEY
from nltk.tokenize import sent_tokenize
from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity
import tempfile
import os
import PyPDF2

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
        df

        for file in files:
            text = extract_text(file)
            file_df = document_to_dataframe(text)
            df = pd.concat([df, file_df], ignore_index=True)

        df['embedding'] = df['Sentence'].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))

        earnings_search_vector = get_embedding(query, engine="text-embedding-ada-002")

        df['similarities'] = df['embedding'].apply(lambda x: cosine_similarity(x, earnings_search_vector))
        df = df.sort_values('similarities', ascending=False)
        top_3_items = df.head(5)
        results = top_3_items['Sentence'].tolist()
        results = [str(element) for element in results]

        return render_template('results.html', query=query, results=results)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
