import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from skmultilearn.ensemble import RakelD
import nltk
from imblearn.over_sampling import SMOTE
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from string import punctuation
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.adapt import MLkNN
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
import warnings
warnings.filterwarnings('always')
def load_data(file_path):
    return pd.read_csv(file_path)

def plot_genre_distribution(data):
    genres_count = data['genres'].apply(lambda x: len(eval(x)))
    plt.hist(genres_count, bins=max(genres_count), edgecolor='black')
    plt.title('Distribution of Number of Genres per Movie')
    plt.xlabel('Number of Genres')
    plt.ylabel('Number of Movies')
    plt.savefig('table_genres.png')

def plot_movies_by_genre(data):
    genres_list = data['genres'].apply(lambda x: [genre['name'] for genre in eval(x)])
    genres_count = genres_list.apply(lambda x: len(x))
    plt.hist(genres_count, bins=max(genres_count), edgecolor='black')
    plt.title('Distribution of Movies by Number of Genres')
    plt.xlabel('Number of Genres')
    plt.ylabel('Number of Movies')
    plt.savefig('table_mevies_by_genres.png')
def plot_word_clouds(data):
    unique_genres = set()
    for genres_list in data['genres'].apply(lambda x: eval(x)):
        unique_genres.update([genre['name'] for genre in genres_list])   
    for genre_name in unique_genres:
        print(genre_name)
        genre_descriptions = data[data['genres'].apply(lambda x: genre_name in x)]['overview']
        print(genre_descriptions)
        genre_descriptions = genre_descriptions.astype(str)
        all_descriptions = ' '.join(genre_descriptions)

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_descriptions)


        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud for {genre_name} Genre')
        plt.savefig(f'{genre_name} cloud')


def preprocess(text, lemmatizer):
    sentences = sent_tokenize(text)

    sentences = [sentence.lower() for sentence in sentences]

    sentences = [
        "".join(c for c in sentence if c not in punctuation) for sentence in sentences
    ]

    sentences = [word_tokenize(sentence) for sentence in sentences]

    stop_words = stopwords.words("english")
    sentences = [
        [word for word in sentence if word not in stop_words] for sentence in sentences
    ]
    sentences = [
        [word for word in sentence if not word.startswith("http")]
        for sentence in sentences
    ]
    sentences = [
        [word for word in sentence if len(word) >= 3] for sentence in sentences
    ]
    sentences = [[lemmatizer.lemmatize(word) for word in sentence] for sentence in sentences]
    words = " ".join([" ".join(sentence) for sentence in sentences])
    return words

# Multilabel классификация - Подход 1
def train_multilabel_classifier(X_train, Y_train, X_test, Y_test, unique_genres):
    model_list = []
    
    for genre in unique_genres:
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', OneVsRestClassifier(LogisticRegression(solver='lbfgs', max_iter=1000, class_weight='balanced')))
        ])
        pipeline.fit(X_train, Y_train.apply(lambda x: genre in x))
        Y_pred = pipeline.predict(X_test)
        print(Y_pred)
        print(classification_report(Y_test.apply(lambda x: genre in x), Y_pred))
        print('Accuracy: ', accuracy_score(Y_test.apply(lambda x: genre in x), Y_pred))
        print('Precision: ', precision_score(Y_test.apply(lambda x: genre in x), Y_pred))
        print('Recall: ', recall_score(Y_test.apply(lambda x: genre in x), Y_pred))
        print('F1: ', f1_score(Y_test.apply(lambda x: genre in x), Y_pred))
        print(f'-'*50)
        
        model_list.append(pipeline)
def train_multilabel_classifier_2(x_train, y_train, X_test, Y_test):
    mlb = MultiLabelBinarizer()
    accuracy_values = []
    precision_values = []
    recall_values = []
    f1_values = []
    sum_values = []
    max_k = 50
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    y_train = np.array(mlb.fit_transform(y_train))
    for k in range(1, max_k):
            print('k = ', k)
            classifier = MLkNN(k=10)
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('clf',classifier)
            ])
            pipeline.fit(x_train, y_train)
            y_pred = pipeline.predict(X_test)
            print('Accuracy: ', accuracy_score(y_pred, mlb.transform(Y_test)))
            print('Precision: ', precision_score(y_pred, mlb.transform(Y_test), average='micro'))
            print('Recall: ', recall_score(y_pred, mlb.transform(Y_test), average='micro'))
            print('F1: ', f1_score(y_pred, mlb.transform(Y_test), average='micro'))
            sum = accuracy_score(y_pred, mlb.transform(Y_test)) + precision_score(y_pred, mlb.transform(Y_test), average='micro') + recall_score(y_pred, mlb.transform(Y_test), average='micro') + f1_score(y_pred, mlb.transform(Y_test), average='micro')
            print('Sum: ', sum)
            accuracy_values.append(accuracy_score(y_pred, mlb.transform(Y_test)))
            precision_values.append(precision_score(y_pred, mlb.transform(Y_test), average='micro'))
            recall_values.append(recall_score(y_pred, mlb.transform(Y_test), average='micro'))
            f1_values.append(f1_score(y_pred, mlb.transform(Y_test), average='micro'))
            sum_values.append(sum)
            print()

def train_rakeld_classifier(X_train, y_train, X_test, y_test, genres):
    from sklearn.naive_bayes import GaussianNB  
    mlb = MultiLabelBinarizer()
    x_train = np.array(X_train)
    y_train_bin = np.array(mlb.fit_transform(y_train))
    pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', RakelD(base_classifier=GaussianNB(), labelset_size=20))
        ])
    pipeline.fit(x_train, y_train_bin)
    print('code goes here')
    y_pred = pipeline.predict(X_test)

    print('Accuracy:', accuracy_score(mlb.transform(y_test), y_pred))
    print('Precision:', precision_score(mlb.transform(y_test), y_pred, average='weighted'))
    print('Recall:', recall_score(mlb.transform(y_test), y_pred, average='weighted'))
    print('F1 Score:', f1_score(mlb.transform(y_test), y_pred, average='weighted'))


if __name__ == "__main__":
    data = load_data("lab4_files/tmdb_5000_movies.csv")
    data = data.dropna(subset=['overview', 'genres'])
    plot_movies_by_genre(data)
    plot_word_clouds(data)
    genres = data['genres'].apply(lambda x: [i['name'] for i in eval(x)])
    data['genres'] = genres
    genres = genres.explode()
    genres = genres.value_counts()
    genres = genres[genres > 100]
    genres = genres.index.tolist()
    lemmatizer = WordNetLemmatizer()
    data['overview_processed'] = data['overview'].apply(lambda x: preprocess(x, lemmatizer))
    X_train, X_test, y_train, y_test = train_test_split(data['overview_processed'], data['genres'], test_size=0.2, random_state=42)

    train_multilabel_classifier(X_train, y_train,X_test, y_test,genres)
    train_rakeld_classifier(X_train, y_train,X_test, y_test,genres)

    new_text = "Your new movie overview text"