import pandas as pd
import spacy
import numpy as np
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_text(text, nlp):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return tokens

def tokenize_and_preprocess(df, nlp):
    return df['Review'].apply(lambda x: preprocess_text(x, nlp)).tolist()

def build_word2vec_model(sentences, vector_size=100, window=5, min_count=5, workers=3, epochs=3):
    return Word2Vec(sentences=sentences, vector_size=vector_size, window=window, min_count=min_count, workers=workers, epochs=epochs)

def compare_similar_words(model, words):
    for word in words:
        similar_words = model.wv.most_similar(word)
        print(f"Words similar to '{word}': {similar_words}")

def visualize_word_embeddings(model, words):
    word_vectors = np.array([model.wv[word] for word in words])

    tsne_model = TSNE(n_components=2, random_state=42,perplexity=min(5, len(word_vectors)-1))
    word_vectors_2d = tsne_model.fit_transform(word_vectors)

    plt.figure(figsize=(10, 8))
    for i, word in enumerate(words):
        plt.scatter(word_vectors_2d[i, 0], word_vectors_2d[i, 1], label=word)

    plt.legend()
    plt.show()


def start():
    file_path = 'lab5_files\\tripadvisor_hotel_reviews.csv'
    df = load_data(file_path)
    nlp = spacy.load("en_core_web_sm")
    sentences = tokenize_and_preprocess(df, nlp)
    word2vec_model = build_word2vec_model(sentences)
    words_to_compare = ['dirt', 'room', 'light', 'window']
    compare_similar_words(word2vec_model, words_to_compare)
    visualize_word_embeddings(word2vec_model, words_to_compare)

if __name__ == "__main__":
    start()