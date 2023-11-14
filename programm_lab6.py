import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
nlp = spacy.load("en_core_web_sm")
def load_data(file_path):
    # Загрузка данных
    data = pd.read_csv(file_path)
    return data

def preprocess_text(text):
    # Предобработка текстов с использованием spacy
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

def prepare_data(data):
    # Подготовка данных
    data['Processed_Text'] = data['Comment'].apply(preprocess_text)
    X_train, X_test, y_train, y_test = train_test_split(data['Processed_Text'], data['Emotion'], test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def vectorize_tfidf(X_train, X_test):
    # Векторизация текстов с использованием TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf

def vectorize_doc2vec(X_train):
    # Векторизация текстов с использованием Doc2Vec
    documents = [TaggedDocument(words=text.split(), tags=[str(i)]) for i, text in enumerate(X_train)]
    doc2vec_model = Doc2Vec(vector_size=100, window=5, min_count=1, workers=4, epochs=20)
    doc2vec_model.build_vocab(documents)
    doc2vec_model.train(documents, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)
    X_train_doc2vec = [doc2vec_model.infer_vector(text.split()) for text in X_train]
    return X_train_doc2vec

def train_model(X_train, y_train, vectorization_method='tfidf'):
    # Обучение модели классификации
    if vectorization_method == 'tfidf':
        model = LogisticRegression()
        model.fit(X_train, y_train)
        return model
    elif vectorization_method == 'doc2vec':
        model = LogisticRegression()
        model.fit(X_train, y_train)
        return model
    else:
        raise ValueError("Invalid vectorization method. Choose 'tfidf' or 'doc2vec'.")

def evaluate_model(model, X_test, y_test):
    # Оценка точности модели
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

def visualize_vectors(X_test, y_test, vectorization_method='tfidf'):
    # Визуализация векторов текстов с использованием t-SNE
    tsne = TSNE(n_components=2, random_state=42,init="random")
    if vectorization_method == 'tfidf':
        X_tsne = tsne.fit_transform(X_test)
    elif vectorization_method == 'doc2vec':
        X_tsne = tsne.fit_transform(X_test)
    else:
        raise ValueError("Invalid vectorization method. Choose 'tfidf' or 'doc2vec'.")
    unique_labels = y_test.unique()
    label_to_num = {label: i for i, label in enumerate(unique_labels)}
    y_test_numeric = y_test.map(label_to_num)
    # Визуализация
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1],c=y_test_numeric, cmap="viridis")
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.savefig('figure.png')
    plt.show()

def start(file_path):
    # Основная функция для запуска всех шагов
    data = pd.read_csv(file_path, encoding='latin1')
    print(data.head())
    X_train, X_test, y_train, y_test = prepare_data(data)
    # Выберите один из методов векторизации: 'tfidf' или 'doc2vec'
    vectorization_method = 'tfidf'

    if vectorization_method == 'tfidf':
        X_train_vectorized, X_test_vectorized = vectorize_tfidf(X_train, X_test)
    else:
        X_train_vectorized = vectorize_doc2vec(X_train)
        X_test_vectorized = vectorize_doc2vec(X_test)

    model = train_model(X_train_vectorized, y_train, vectorization_method)
    evaluate_model(model, X_test_vectorized, y_test)
    visualize_vectors(X_test_vectorized, y_test, vectorization_method)

# Замените "your_dataset.csv" на имя вашего файла с данными
start("Emotion_classify_Data.csv")