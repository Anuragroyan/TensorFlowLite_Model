# train_spam_model.py
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences

# Dummy dataset
labels = [
        'ham', 'spam', 'ham', 'spam', 'ham',
        'spam', 'ham', 'spam', 'ham', 'spam',
        'ham', 'spam', 'ham', 'spam', 'ham',
        'spam', 'ham', 'spam', 'ham', 'spam',
        'ham', 'spam', 'ham', 'spam', 'ham',
        'spam', 'ham', 'spam', 'ham', 'spam',
        'ham', 'spam', 'ham', 'spam', 'ham',
        'spam', 'ham', 'spam', 'ham', 'spam',
        'ham', 'spam', 'ham', 'spam', 'ham'
    ]
texts = [
        'Hey, how are you doing?',
        'Win a FREE iPhone now!!!',
        'Are we still on for lunch?',
        'Congratulations, you have won a prize!',
        'Let’s catch up tomorrow.',
        'Claim your $1000 gift card now!',
        'Can you send me the notes?',
        'Limited time offer, act now!',
        'I\'ll be there in 10 minutes.',
        'Exclusive deal just for you!',
        'Happy birthday!',
        'Get cash fast, no credit check!',
        'What time is the meeting?',
        'You have been selected for a survey.',
        'Dinner at 7?',
        'Earn money from home easily!',
        'Did you complete the assignment?',
        'Urgent! Your account is compromised.',
        'Thanks for the update!',
        'Click here to win a trip!',
        'Let\'s meet at the coffee shop.',
        'You are pre-approved for a loan!',
        'Please find the attachment.',
        'Lowest insurance rates guaranteed!',
        'Join the Zoom call now.',
        'Congratulations! You’ve won!',
        'Where should we go today?',
        'Unlock your special bonus now!',
        'See you at the party!',
        'Important notice: Final warning!',
        'Call me when you\'re free.',
        'You’re a lucky winner!',
        'How’s the new job?',
        'Get rich quick with this scheme!',
        'Let’s plan a movie night.',
        'Your account will be deactivated!',
        'Lunch at our usual place?',
        'Hurry up! Limited slots left!',
        'Good luck with your exam!',
        'Don\'t miss this limited-time deal!',
        'That sounds perfect!',
        'Act now to claim your reward!',
        'Catch you later!',
        'Earn $5000 from your phone!',
        'Meeting postponed to tomorrow.'  
    ] 

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Tokenize texts
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=10)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, labels_encoded, test_size=0.2)

# Model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=1000, output_dim=16, input_length=10),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save tokenizer
import pickle
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('spam_model.tflite', 'wb') as f:
    f.write(tflite_model)
