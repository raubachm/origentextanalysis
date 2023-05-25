import os
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from flask import Flask, render_template, request

app = Flask(__name__)

def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()

def import_sentences(filename):
    sentences = read_file(filename).split(',')
    sentences = [sentence for sentence in sentences if len(sentence.split()) >= 4]
    return sentences

def split_document_into_sentences(document):
    sentences = re.split(r'(?<=[.!?])\s', document)
    sentences = [sentence for sentence in sentences if len(sentence.split()) >= 4]
    return sentences

def calculate_similarity(embeddings1, embeddings2):
    similarity_matrix = cosine_similarity(embeddings1, embeddings2)
    return similarity_matrix

embeddings_path = {
    'all-MiniLM-L6-v2': 'embeddings_original_all-MiniLM-L6v2.npy',
    'RoBERTa-base': 'embeddings_original_roberta-base.npy'
}

original_text = read_file('DePrincipii_AgainstCelsus_CommentaryonJohn_CommentaryonMatthew')
sentences_original = split_document_into_sentences(original_text)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        model_name = request.form['model']  # Get the selected model from the form

        if model_name == 'all-MiniLM-L6-v2':
            model_path = 'sentence-transformers/all-MiniLM-L6-v2'
        elif model_name == 'RoBERTa-base':
            model_path = 'sentence-transformers/roberta-base-nli-mean-tokens'
        else:
            return 'Invalid model selection.'

        model = SentenceTransformer(model_path)  # Load the selected model

        embeddings_original = np.load(embeddings_path[model_name])  # Load the corresponding pretrained embeddings

        new_text_path = request.form['new_text']
        if new_text_path:
            new_text = read_file(new_text_path)
            sentences_new = split_document_into_sentences(new_text)
            embeddings_new = model.encode(sentences_new)

            similarity_matrix = calculate_similarity(embeddings_original, embeddings_new)

            similarity_score = np.mean(similarity_matrix)
            indices = np.unravel_index(np.argsort(similarity_matrix, axis=None), similarity_matrix.shape)
            top_indices_original = indices[0][-10:]
            top_indices_new = indices[1][-10:]

            result = []
            for i, j in zip(top_indices_original, top_indices_new):
                result.append({
                    'original': sentences_original[i],
                    'new': sentences_new[j],
                    'similarity': similarity_matrix[i][j]
                })

            return render_template('result.html', similarity_score=similarity_score, result=result)
        else:
            return 'Please provide a valid file path.'

    return render_template('index.html')

if __name__ == '__main__':
    app.run()