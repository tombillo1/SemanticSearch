from flask import Flask, render_template, request
import openai
import pandas as pd
import nltk
nltk.download('punkt')
from config import API_KEY
from nltk.tokenize import sent_tokenize
from openai.embeddings_utils import get_embedding
import faiss
import numpy as np
from config import API_KEY
import requests

# Set your Confluence page ID and API key here
CONFLUENCE_PAGE_ID = "YOUR_CONFLUENCE_PAGE_ID"
CONFLUENCE_API_KEY = "YOUR_CONFLUENCE_API_KEY"

openai.api_key = API_KEY

app = Flask(__name__, static_url_path='/static', static_folder='static')

def extract_text(page_id, api_key):
    # Confluence API endpoint to fetch the content of a specific page
    api_url = f"https://your-confluence-domain/wiki/rest/api/content/{page_id}?expand=body.storage,children.page"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Make a GET request to fetch the page content
    response = requests.get(api_url, headers=headers)

    if response.status_code == 200:
        # Extract the content from the response JSON
        content = response.json().get("body", {}).get("storage", {}).get("value", "")

        # Recursively fetch content from sub-pages
        children = response.json().get("children", {}).get("page", [])
        for child in children:
            child_id = child.get("id")
            child_content = extract_text(child_id, api_key)
            content += " " + child_content

        return content
    else:
        return None

def document_to_dataframe(document):
    sentences = sent_tokenize(document)
    sentences = [sentence.replace('\n', '').replace('\t', '') for sentence in sentences]
    df = pd.DataFrame(sentences, columns=['Sentence'])
    return df

def initialize_faiss_index(df):
    d = df['embedding'][0].shape[1]  # Get the dimension of the embeddings
    index = faiss.IndexFlatL2(d)

    embeddings = np.vstack(df['embedding'].to_numpy())
    index.add(embeddings)

    # Save the index to a file
    faiss.write_index(index, "faiss_index.index")

def load_faiss_index():
    # Load the index from the file
    index = faiss.read_index("faiss_index.index")
    return index

@app.route('/')
def index():
    results = ["Result 1", "Result 2", "Result 3"]
    return render_template("results.html", results=results)

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        query = request.form['query']

        # Fetch the content from Confluence using the stored page ID and API key
        content = extract_text(CONFLUENCE_PAGE_ID, CONFLUENCE_API_KEY)

        if content:
            df = document_to_dataframe(content)

            df['embedding'] = df['Sentence'].apply(lambda x: np.array(get_embedding(x, engine='text-embedding-ada-002')))
            df['embedding'] = df['embedding'].apply(lambda x: x.reshape(1, -1))

            # Initialize and save the Faiss index during application startup
            initialize_faiss_index(df)

            # Load the Faiss index from the file
            index = load_faiss_index()

            query_embedding = get_embedding(query, engine='text-embedding-ada-002')
            query_embedding = np.array(query_embedding).reshape(1, -1)

            _, indices = index.search(query_embedding, k=5) # Returns 5 closest results
            top_items = df.iloc[indices[0]]
            results = top_items['Sentence'].tolist()
        else:
            results = []

        return render_template('results.html', query=query, results=results)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
