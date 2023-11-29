from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
from gensim import corpora
import numpy as np
import matplotlib.pyplot as plt
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_texts(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        texts = file.readlines()
    return texts

def preprocess(texts):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    processed_texts = []
    
    for text in texts:
        words = word_tokenize(text.lower())
        processed_words = [lemmatizer.lemmatize(word) for word in words if word.isalnum() and word not in stop_words]
        processed_texts.append(processed_words)

    return processed_texts

def save_dictionary(dictionary, file_path):
    dictionary.save(file_path)

def save_corpus(corpus, file_path):
    corpora.MmCorpus.serialize(file_path, corpus)

def train_lda_model(corpus, dictionary, num_topics, alpha):
    model = LdaModel(corpus, id2word=dictionary, num_topics=num_topics, alpha=alpha)
    return model

def evaluate_model_coherence(model, texts, dictionary, coherence_measure='c_v'):
    coherence_model = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence=coherence_measure)
    return coherence_model.get_coherence()

def find_optimal_num_topics(dictionary,corpus,processed_texts,start, end, step, alpha, coherence_measure):
    num_topics_range = range(start, end, step)
    coherence_scores = []

    for num_topics in num_topics_range:
        model = LdaModel(corpus, id2word=dictionary, num_topics=num_topics, alpha=alpha)
        coherence = evaluate_model_coherence(model, processed_texts, dictionary, coherence_measure)
        coherence_scores.append(coherence)

    plt.plot(num_topics_range, coherence_scores, marker='o')
    plt.ylabel(f'Coherence Метрика ({coherence_measure})')
    plt.savefig('graph_pab9.png')
    plt.show()

    optimal_num_topics = num_topics_range[np.argmax(coherence_scores)]
    return optimal_num_topics

def visualize_topics(model, corpus, dictionary, output_path='lda_visualization_lab9.html'):
    vis_data = gensimvis.prepare(model, corpus, dictionary)
    pyLDAvis.save_html(vis_data, output_path)

def start():
    texts = load_texts('ap/ap.txt')
    processed_texts = preprocess(texts)
    dictionary = Dictionary(processed_texts)
    save_dictionary(dictionary, 'ap/vocab.txt')
    corpus = [dictionary.doc2bow(text) for text in processed_texts]
    save_corpus(corpus, 'ap/ap.dat')
    alpha_value = 'asymmetric' 
    optimal_topics_c_v = find_optimal_num_topics(dictionary,corpus,processed_texts,start=5, end=50, step=5, alpha=alpha_value, coherence_measure='c_v')
    print(f'Лучшее колличество тем для метрики c_v: {optimal_topics_c_v}')
    optimal_topics_c_uci = find_optimal_num_topics(dictionary,corpus,processed_texts,start=5, end=50, step=5, alpha=alpha_value, coherence_measure='c_uci')
    print(f'Лучшее колличество тем для метрики c_uci: {optimal_topics_c_uci}')
    lda_model = train_lda_model(corpus, dictionary, num_topics=optimal_topics_c_uci, alpha=alpha_value)
    coherence_score = evaluate_model_coherence(lda_model, processed_texts, dictionary)
    print(f'Coherence метрика: {coherence_score}')

    visualize_topics(lda_model, corpus, dictionary)

if __name__ == "__main__":
    start()