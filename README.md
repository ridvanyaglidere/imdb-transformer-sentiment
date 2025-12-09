# Transformer-based Sentiment Analysis on IMDB Dataset

## Overview

This repository contains a **Transformer-based model** for sentiment analysis of movie reviews using the IMDB dataset. The model classifies reviews as either **positive** or **negative**. It is implemented from scratch using **TensorFlow** and **Keras**, showcasing how Transformer architectures can be applied to natural language processing tasks.

The goal of this project is to provide a practical example of using **Multi-Head Attention**, **Layer Normalization**, **Feed Forward Networks**, and **Residual Connections** for text classification.

---

## Features

* Custom Transformer block with:

  * Multi-Head Self-Attention
  * Layer Normalization
  * Feed Forward Network
  * Residual Connections
  * Dropout Regularization
* Multi-layer Transformer architecture
* Tokenization and padding for IMDB movie reviews
* Real-time prediction for new movie reviews
* Training and validation visualization (loss & accuracy)
* Easy-to-use sentiment prediction interface

---

## Dataset

The model uses the **IMDB movie review dataset** with:

* **Vocabulary size**: 10,000 most frequent words
* **Max review length**: 100 tokens

The dataset is automatically loaded via `keras.datasets.imdb`.

---

## Model Architecture

1. **Embedding Layer**: Converts word indices into dense vectors.
2. **Transformer Blocks** (stacked):

   * Multi-Head Self-Attention
   * Layer Normalization
   * Feed Forward Network
   * Residual Connections
   * Dropout
3. **Global Average Pooling**: Aggregates sequence information.
4. **Fully Connected Output Layer**: Sigmoid activation for binary classification.

---

## Installation

Make sure Python 3.8+ is installed. Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/imdb-transformer-sentiment.git
cd imdb-transformer-sentiment
pip install -r requirements.txt
```

**Dependencies:**

* tensorflow
* keras
* numpy
* matplotlib
* prompt_toolkit

---

## Usage

### Training the Model

```python
from model import TransformerModel
from data import load_imdb_data

x_train, y_train, x_test, y_test = load_imdb_data()
model = TransformerModel(num_layers=4, embed_size=64, heads=4, input_dim=10000, output_dim=1)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history = model.fit(x_train, y_train, epochs=2, batch_size=256, validation_data=(x_test, y_test))
```

### Predicting Sentiment

```python
from predict import predict_sentiment

review = "This movie was amazing!"
score = predict_sentiment(model, review, word_index, max_len=100)

if score > 0.5:
    print(f"Positive sentiment ({score:.2f})")
else:
    print(f"Negative sentiment ({score:.2f})")
```

---

## Results

* After just a few epochs, the model achieves high training and validation accuracy.
* Real-time sentiment prediction works effectively on short reviews.

---

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests to improve the model, add pre-trained embeddings, or extend it to other NLP datasets.

---

## License

This project is licensed under the **MIT License**.

---

