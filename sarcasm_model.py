# sarcasm_model.py
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
import numpy as np
from keras.preprocessing.sequence import pad_sequences  
import json

# Dummy dataset
sentences = [
    "I'm so happy to be stuck in traffic for two hours",
    "What a wonderful day to forget my umbrella",
    "I love when my internet stops working during meetings",
    "The food was absolutely amazing, just kidding it was terrible",
    "I won the lottery! Just kidding, I lost my wallet",
    "Looking forward to Monday... said no one ever",
    "It's raining again, how lovely",
    "The product was great and worked as expected",
    "Thank you for your help, it was really appreciated",
    "This is the best movie I’ve seen all year"
]

labels = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]

# Parameters
vocab_size = 1000
embedding_dim = 16
max_length = 32
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"

# Tokenize
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
labels = np.array(labels)

# Train-test split
split = int(len(padded) * 0.8)
train_sentences = padded[:split]
train_labels = labels[:split]
test_sentences = padded[split:]
test_labels = labels[split:]

# Model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train model
model.fit(train_sentences, train_labels, epochs=10, validation_data=(test_sentences, test_labels))

# Save tokenizer as JSON
tokenizer_json = tokenizer.to_json()
with open("tokenizer.json", "w", encoding='utf-8') as f:
    f.write(tokenizer_json)

# Convert and save TFLite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
import tensorflow as tf

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter._experimental_lower_tensor_list_ops = False
tflite_model = converter.convert()
with open("sarcasm_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model and tokenizer saved successfully as TFLite and JSON.")
