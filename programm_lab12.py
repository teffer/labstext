import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN

def load_imdb_data(max_words=10000, maxlen=200):
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)
    dictionary = imdb.get_word_index()
    
    x_train = pad_sequences(x_train, maxlen=maxlen, padding='post')
    x_test = pad_sequences(x_test, maxlen=maxlen, padding='post')
    
    return x_train, y_train, x_test, y_test

def build_simple_nn_model(maxlen=200):
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=(maxlen,)))
    model.add(Dense(128, activation='relu', input_shape=(maxlen,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def train_and_plot(model, x_train, y_train,type, epochs=10, validation_data=None):
    history = model.fit(x_train, y_train, epochs=epochs, validation_data=validation_data)
    
    plt.plot(history.history['accuracy'], label=f'Доля верных ответов на тренировочной обучающей выборке {type}')
    plt.plot(history.history['val_accuracy'], label=f'Доля верных ответов на тренировочной проверяющей выборке{type}')
    plt.xlabel('Эпоха обучения')
    plt.ylabel('Доля правильных ответов')
    plt.legend()
    plt.savefig(f'lab12.png')

    return history

def evaluate_model(model, x_test, y_test):
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"Точность на тестовой выборке: {test_accuracy * 100:.2f}%")


if __name__ == '__main__':
    max_words = 10000
    maxlen = 200
    x_train, y_train, x_test, y_test = load_imdb_data(max_words, maxlen)

    model_nn = build_simple_nn_model(maxlen)
    history_nn = train_and_plot(model_nn, x_train, y_train,'before', epochs=5, validation_data=(x_test, y_test))
    evaluate_model(model_nn, x_test, y_test)

    embedding_dim = 2
    rnn_neurons = 32

    model_rnn = Sequential()
    model_rnn.add(Embedding(max_words, embedding_dim, input_length=maxlen))
    model_rnn.add(SimpleRNN(rnn_neurons))
    model_rnn.add(Dense(1, activation='sigmoid'))

    model_rnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history_rnn = train_and_plot(model_rnn, x_train, y_train,'before', epochs=5, validation_data=(x_test, y_test))
    evaluate_model(model_rnn, x_test, y_test)