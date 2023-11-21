import pymorphy2
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')
import re

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-zA-Zа-яА-Я\- ]', '', text)
    tokens = nltk.word_tokenize(text, language='russian')
    return tokens
    
def analyze_text(text):
    morph = pymorphy2.MorphAnalyzer()
    words =preprocess_text(text)
    word_types_count = {}
    
    for word in words:
        parsed_word = morph.parse(word)[0]
        word_type = parsed_word.tag.POS
        if word_type is not None:
            if word_type in word_types_count:
                word_types_count[word_type] += 1
            else:
                word_types_count[word_type] = 1
    return word_types_count

def plot_word_types_distribution(word_types_count, title):
    labels = list(word_types_count.keys())
    values = list(word_types_count.values())
    
    plt.bar(labels, values,align='center')
    plt.title(title)
    plt.xlabel("Часть речи")
    plt.ylabel("Количество слов")
    plt.xticks(rotation=30, ha='right')
    plt.savefig(f'lab7_{title}')
    
def start():
    with open("sciencetext.txt", "r", encoding="utf-8") as file:
        scientific_text = file.read()
    with open("trashbook.txt", "r", encoding="utf-8") as file:
        fictional_text = file.read()
    scientific_analysis = analyze_text(scientific_text)
    fictional_analysis = analyze_text(fictional_text)
    print(scientific_analysis)
    print(fictional_analysis)
    
    plot_word_types_distribution(scientific_analysis,'анализ научной статьи')
    plot_word_types_distribution(fictional_analysis,'анализ части плохой книги')


if __name__=='__main__':
    start()