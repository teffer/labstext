import nltk
nltk.download('stopwords')
nltk.download('WordNetLemmatizer')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_df():
    df = pd.read_csv('youtoxic_english_1000.csv',sep=',')
    selected_columns = ['CommentId', 'VideoId', 'Text', 'IsHatespeech', 'IsReligiousHate', 'IsProvocative']
    df_selected = df[selected_columns]
    df_selected =df_selected[(df_selected[['IsHatespeech','IsReligiousHate','IsProvocative']]==True).any(axis=1)]
    print(df_selected)
    return df_selected


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Zа-яА-Я\- ]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords]
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    print(tokens)
    print(lemmatized_tokens)
    return tokens, lemmatized_tokens

def counter_and_visualiser(df):
    categories = ['IsHatespeech', 'IsReligiousHate', 'IsProvocative']

    for category in categories:
        category_df = df[df[category] == True]
        lemmatized_tokens = [token for tokens in category_df['LemmatizedTokens'] for token in tokens]
        non_lemmatized_tokens = [token for tokens in category_df['Tokens'] for token in tokens]
        lemmatized_word_counts = Counter(lemmatized_tokens)
        non_lemmatized_word_counts = Counter(non_lemmatized_tokens)
        sorted_lemmatized_word_counts = sorted(lemmatized_word_counts.items(), key=lambda x: x[1], reverse=True)
        sorted_non_lemmatized_word_counts = sorted(non_lemmatized_word_counts.items(), key=lambda x: x[1], reverse=True)
        print(f"Category: {category}")
        print("Lemmatized Word Frequency:")
        for word, count in sorted_lemmatized_word_counts:
            print(f"{word}: {count}")
        print("\nNon-Lemmatized Word Frequency:")
        for word, count in sorted_non_lemmatized_word_counts:
            print(f"{word}: {count}")
            
            
        plt.figure(figsize=(10, 5))
        plt.bar(range(20), [count for word, count in sorted_lemmatized_word_counts[:20]], tick_label=[word for word, count in sorted_lemmatized_word_counts[:20]])
        plt.title(f'Lemmatized Word Frequency ({category})')
        plt.tight_layout()
        plt.savefig(f'Lemmatized Word Frequency ({category}).png')
        plt.show()
        plt.figure(figsize=(10, 5))
        plt.barh(range(20), [count for word, count in sorted_non_lemmatized_word_counts[:20]], tick_label=[word for word, count in sorted_non_lemmatized_word_counts[:20]])
        plt.title(f'Non-Lemmatized Word Frequency ({category})')
        plt.tight_layout()
        plt.savefig(f'Non-Lemmatized Word Frequency ({category}).png')
        plt.show()
        lemmatized_wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(lemmatized_word_counts)
        plt.figure(figsize=(10, 5))
        plt.imshow(lemmatized_wordcloud, interpolation='bilinear')
        plt.title(f'Lemmatized Word Cloud ({category})')
        plt.savefig(f'Lemmatized Word Cloud ({category}).png')
        plt.show()
        non_lemmatized_wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(non_lemmatized_word_counts)
        plt.figure(figsize=(10, 5))
        plt.imshow(non_lemmatized_wordcloud, interpolation='bilinear')
        plt.title(f'Non-Lemmatized Word Cloud ({category})')
        plt.savefig(f'Non-Lemmatized Word Cloud ({category}).png')
        plt.show()
if __name__ == '__main__':
    df = preprocess_df()
    df['Tokens'], df['LemmatizedTokens'] = zip(*df['Text'].apply(preprocess_text))
    counter_and_visualiser(df)
    print(df)