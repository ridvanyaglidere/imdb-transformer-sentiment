# load data and prepcrossing

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from IPython import embed
from docutils.nodes import attention
from keras import layers, models
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from prompt_toolkit.shortcuts import input_dialog
from tensorflow.python.keras.backend import epsilon

# veri setini yükleme
max_features = 10000  # en çok kullanılan 10000 kelime
max_len = 100  # her yorumun max uzunluğu

# imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# yorum uzunluklarını max=100
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

# veri inceleme
word_index = imdb.get_word_index()

# kelime dizinini geri döndürmek için ters çevirme
reverse_word_index = {index + 3: word for word, index in word_index.items()}
reverse_word_index[0] = "<PAD>"
reverse_word_index[1] = "<START>"
reverse_word_index[2] = "<UNK>"
reverse_word_index[3] = "<UNUSED>"

def decode_review(encoded_review):
    return " ".join([reverse_word_index.get(i, "?") for i in encoded_review])

random_indicies = np.random.choice(len(x_train), size=3, replace=False)

for i in random_indicies:
    print(f"{decode_review(x_train[i])}")
    print(f"etiket:{y_train[i]}")
    print()

# transformers katmanı tanımlaması
class TransformerBlock(layers.Layer):
    def __init__(self, embed_size, heads, dropout_rate=0.3):
        super(TransformerBlock, self).__init__()

        self.attention = layers.MultiHeadAttention(num_heads=heads, key_dim=embed_size)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

        self.feed_forward = models.Sequential([
            layers.Dense(embed_size * heads, activation="relu"),
            layers.Dense(embed_size)
        ])

        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, x, training):
        attention = self.attention(x, x)

        x = self.norm1(x + self.dropout1(attention, training=training))

        feed_forward = self.feed_forward(x)

        return self.norm2(x + self.dropout2(feed_forward, training=training))


class TransformerModel(models.Model):
    def __init__(self, num_layers, embed_size, heads, input_dim, output_dim, dropout_rate=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = layers.Embedding(input_dim, output_dim=embed_size)

        self.transformers_blok = [
            TransformerBlock(embed_size, heads, dropout_rate) for _ in range(num_layers)
        ]

        self.global_avg_pooling = layers.GlobalAveragePooling1D()
        self.dropout = layers.Dropout(dropout_rate)
        self.fc = layers.Dense(output_dim, activation="sigmoid")

    def call(self, x, training):
        x = self.embedding(x)

        for transformer in self.transformers_blok:
            x = transformer(x, training=training)   # ← ← ← BURASI ZORUNLU DÜZELTME

        x = self.global_avg_pooling(x)
        x = self.dropout(x, training=training)

        return self.fc(x)


num_layers = 4
embed_size = 64
num_heads = 4
input_dim = max_features
output_dim = 1
dropout_rate = 0.1

model = TransformerModel(num_layers, embed_size, num_heads, input_dim, output_dim, dropout_rate)
model.build(input_shape=(None, max_len))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit(x_train, y_train, epochs=2, batch_size=256, validation_data=(x_test, y_test))

plt.figure()
plt.subplot(1,2,1)
plt.plot(history.history["loss"] , label="Training Loss")
plt.plot(history.history["val_loss"],label="Validation Loss")
plt.title("Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["accuracy"],label="Training Accuracy")
plt.plot(history.history["val_accuracy"],label="Validation Accuracy")
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

def predict_sentiment(model , text ,word_index ,maxlen):
    encod_text=[word_index.get(word ,0) for word in text.lower().split()]
    paded_text=pad_sequences([encod_text], maxlen=maxlen)
    prediction=model.predict(paded_text)
    return  prediction[0][0]

word_index=imdb.get_word_index()
user_input=input("bir film yorumu yazınız lütfen : ")
sentiment_score=predict_sentiment(model,user_input,word_index,max_len)
print(sentiment_score)

if sentiment_score>0.5:
    print(f"Tahmin Sonucu % {int(round(sentiment_score *100,0))} olasılığı ile olumlu skor : {sentiment_score}")
else:
    print(f"Tahmin Sonucu %{ 100 -int(round(sentiment_score *100,0))} olasılığı ile olumsuz skor : {sentiment_score}")
