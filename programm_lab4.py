import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import spacy
import matplotlib.pyplot as plt

def load_data(file_path):
    data = pd.read_csv(file_path)
    data = data.dropna(subset=['overview'])
    return data

def preprocess_text(text, nlp):
    doc = nlp(text)
    lemmatized_text = ' '.join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])
    return lemmatized_text.lower()

def preprocess_data(data):
    nlp = spacy.load("en_core_web_sm")
    data['processed_overview'] = data['overview'].apply(lambda x: preprocess_text(x, nlp))
    return data

def vectorize_data(data):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(data['processed_overview'])
    return vectorizer, tfidf_matrix

def visualize_data(tfidf_matrix):
    tsne = TSNE(n_components=2, random_state=42, init='random')
    tsne_result = tsne.fit_transform(tfidf_matrix)

    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.5)
    plt.show()

def find_similar_movies(movie_index, matrix, top_n=5):
    cosine_similarities = cosine_similarity(matrix[movie_index], matrix).flatten()
    related_movies = cosine_similarities.argsort()[:-top_n-1:-1]
    return related_movies

def vectorize_new_movie(description, vectorizer,nlp):
    processed_description = preprocess_text(description, nlp)
    new_movie_vector = vectorizer.transform([processed_description])
    
    return new_movie_vector
def find_similar_movies_to_new(new_movie_vector, matrix, top_n=5):
    cosine_similarities = cosine_similarity(new_movie_vector, matrix).flatten()
    related_movies = cosine_similarities.argsort()[:-top_n-1:-1]
    return related_movies

def start(file_path='lab4_files\\tmdb_5000_movies.csv'):
    data = load_data(file_path)
    data = preprocess_data(data)
    vectorizer, tfidf_matrix = vectorize_data(data)
    visualize_data(tfidf_matrix)
    movie_index = 0
    similar_movies = find_similar_movies(movie_index, tfidf_matrix)
    nlp = spacy.load("en_core_web_sm")
    print(f"Фильм: {data['original_title'][movie_index]}")
    print("\nПохожие фильмы:")
    for i in similar_movies[1:]:
        print(data['original_title'][i])
        
    new_movie_description = """
    It's a very clever and interesting premise, it came up as a horror recommendation, 
    I would class it more in the suspenseful thriller bracket. It showcases mankind 
    at its very worst, it basically sets out a message that you're alright at the top, 
    and as you come down the social order, things just get worse and worse, with those 
    above you doing slightly better.

    Socially and ideologically this is a major triumph, it really does get you thinking, 
    particularly in these days where has had such a terrible impact worldwide.

    I thought the acting was superb, normally I dislike dubbed films, but in this instance 
    I didn't even notice it, I enjoyed it that much.

    Genuinely terrific acting. I would have liked to have seen a bit of those running 
    the regime, but I guess they wanted ambiguity.

    Quality, 9/10.
    """
    new_movie_vector = vectorize_new_movie(new_movie_description, vectorizer,nlp)
    similar_movies_new = find_similar_movies_to_new(new_movie_vector, tfidf_matrix, top_n=5)
    print("\nПохожие на платформу фильмы:")
    for i in similar_movies_new:
        print(data['original_title'][i])

# Запуск программы
if __name__ == '__main__':
    start()