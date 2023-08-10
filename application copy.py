# python -m spacy download en_core_web_md
from flask import Flask, render_template, request
import pandas as pd
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import spacy
import faiss
import numpy as np

# Load the pre-trained spaCy model with GloVe embeddings
nlp = spacy.load('en_core_web_md')

app = Flask(__name__, static_url_path='/static', static_folder='static')

def extract_text(page_id, api_key):
    # Confluence API endpoint to fetch the content of a specific page
    api_url = f"https://tempthing.atlassian.net/wiki/rest/api/content/{page_id}?expand=body.storage"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Make a GET request to fetch the page content
    response = requests.get(api_url, headers=headers)

    if response.status_code == 200:
        # Extract the content from the response JSON
        content = response.json().get("body", {}).get("storage", {}).get("value", "")
        return content
    else:
        print("Error fetching content from Confluence API. Status code:", response.status_code)
        return None

def document_to_dataframe(document):
    sentences = sent_tokenize(document)
    sentences = [sentence.replace('\n', '').replace('\t', '') for sentence in sentences]
    df = pd.DataFrame(sentences, columns=['Sentence'])
    return df

def sentence_to_embedding(sentence):
    doc = nlp(sentence)
    return doc.vector

def initialize_faiss_index(df):
    d = df['embedding'][0].shape[0]  # Get the dimension of the embeddings
    index = faiss.IndexFlatL2(d)

    embeddings = np.vstack(df['embedding'].to_numpy())
    index.add(embeddings)

    # Save the index to a file
    faiss.write_index(index, "faiss_index.index")

def load_faiss_index():
    # Load the index from the file
    index = faiss.read_index("faiss_index.index")
    return index

# Extract the Confluence page ID from the given URL
def extract_page_id(confluence_page_url):
    page_id = confluence_page_url.split("/")[-1]
    return page_id

@app.route('/')
def index():
    results = ["Result 1", "Result 2", "Result 3"]
    return render_template("results.html", results=results)

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        query = request.form['query']

        # Extract the Confluence page ID from the provided URL
        confluence_page_url = "https://tempthing.atlassian.net/wiki/spaces/~71202061d5761c38f44d29a548bf3719b442a6/pages/458753/Test"
        confluence_page_id = extract_page_id(confluence_page_url)

        # Fetch the content from Confluence using the extracted page ID and API key
        content = extract_text(confluence_page_id, CONFLUENCE_API_KEY)

        if content:
            df = document_to_dataframe(content)

            df['embedding'] = df['Sentence'].apply(sentence_to_embedding)

            # Initialize and save the Faiss index during application startup
            initialize_faiss_index(df)

            # Load the Faiss index from the file
            index = load_faiss_index()

            query_embedding = sentence_to_embedding(query)
            _, indices = index.search(np.array([query_embedding]), k=5)  # Returns 5 closest results
            top_items = df.iloc[indices[0]]
            results = top_items['Sentence'].tolist()
        else:
            results = []

        return render_template('results.html', query=query, results=results)
    return render_template('index.html')

if __name__ == '__main__':
    # Set your Confluence API key here
    CONFLUENCE_API_KEY = "ATATT3xFfGF07lbt3Zk6_6p7B2MZFLgJZsRO5_0DlVOV6ecn2xy2Cv4Wg-8PfiQOwP6ouAcKL-_aW5vQtfEGJDq8VXKCPP-hEESmCGxutKkL8F_eFuUvBuhYevND8OTB0-eDr5eo1eve0Ho3bCYgYlJUMCi4g28pYCJbF9jlhg7bDLIfqCttdvU=6D0AC283"
    app.run(debug=True)
