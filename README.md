# Password Strength Prediction

## Project Objective

The objective of this project is to predict the strength of passwords using machine learning models. This is achieved by training models on a dataset containing passwords and their corresponding strength labels. The models can then classify new passwords as weak, medium, or strong.

## Dataset Description

The dataset used in this project consists of passwords and their respective strength labels, which are categorized into three classes: Weak, Medium, and Strong. The data is stored in an SQLite file named `password_data.sqlite` and contains two columns:
- `password`: The actual password string.
- `strength`: The strength label of the password (0 for Weak, 1 for Medium, 2 for Strong).

## Models Used

1. **Neural Network**: A deep learning model built using TensorFlow and Keras, with multiple dense layers and dropout layers to prevent overfitting.
2. **RandomForestClassifier**: A machine learning model using an ensemble of decision trees to improve classification performance.

## Notebook Description

### Mounting Google Drive
```python
from google.colab import drive

drive.mount('/content/drive')
```

### Loading Data from SQLite File
```python
import sqlite3
import pandas as pd

sqlite_file_path = '/content/drive/MyDrive/password_data.sqlite'
conn = sqlite3.connect(sqlite_file_path)
query = "SELECT password, strength FROM Users"
data = pd.read_sql_query(query, conn)
conn.close()
```
Loads the password data into a pandas DataFrame.

### Data Preprocessing and Feature Extraction
```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

X = data['password']
y = data['strength']

vectorizer = TfidfVectorizer(analyzer='char', max_features=100)
X_tfidf = vectorizer.fit_transform(X).toarray()

X_train_val, X_test, y_train_val, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)
```
Transforms passwords into TF-IDF features and splits the data into training, validation, and test sets.

### Calculating Class Weights
```python
from sklearn.utils import class_weight
import numpy as np

class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
```
Computes class weights to handle class imbalance.

### Building and Training the Neural Network
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_val, y_val), class_weight=class_weights_dict, callbacks=[early_stopping])
```
Trains the neural network model with early stopping to prevent overfitting. The training and validation loss and accuracy are printed for each epoch.

**Output**: 
```
Epoch 1/50
938/938 [==============================] - 5s 4ms/step - loss: 0.8614 - accuracy: 0.5306 - val_loss: 0.7110 - val_accuracy: 0.6511
Epoch 2/50
938/938 [==============================] - 8s 9ms/step - loss: 0.6014 - accuracy: 0.6228 - val_loss: 0.6811 - val_accuracy: 0.6554
Epoch 3/50
938/938 [==============================] - 5s 6ms/step - loss: 0.5338 - accuracy: 0.6569 - val_loss: 0.6273 - val_accuracy: 0.6779
Epoch 4/50
938/938 [==============================] - 5s 6ms/step - loss: 0.4950 - accuracy: 0.6780 - val_loss: 0.5886 - val_accuracy: 0.7027
Epoch 5/50
938/938 [==============================] - 4s 4ms/step - loss: 0.4705 - accuracy: 0.6922 - val_loss: 0.5631 - val_accuracy: 0.7211
Epoch 6/50
938/938 [==============================] - 4s 4ms/step - loss: 0.4504 - accuracy: 0.7037 - val_loss: 0.5433 - val_accuracy: 0.7255
Epoch 7/50
938/938 [==============================] - 5s 5ms/step - loss: 0.4393 - accuracy: 0.7120 - val_loss: 0.5116 - val_accuracy: 0.7376
Epoch 8/50
938/938 [==============================] - 5s 5ms/step - loss: 0.4254 - accuracy: 0.7185 - val_loss: 0.5496 - val_accuracy: 0.7175
Epoch 9/50
938/938 [==============================] - 4s 4ms/step - loss: 0.4145 - accuracy: 0.7255 - val_loss: 0.5337 - val_accuracy: 0.7357
Epoch 10/50
938/938 [==============================] - 4s 4ms/step - loss: 0.4068 - accuracy: 0.7283 - val_loss: 0.5413 - val_accuracy: 0.7354
Epoch 11/50
938/938 [==============================] - 6s 6ms/step - loss: 0.4047 - accuracy: 0.7324 - val_loss: 0.5197 - val_accuracy: 0.7427
Epoch 12/50
938/938 [==============================] - 4s 4ms/step - loss: 0.3928 - accuracy: 0.7391 - val_loss: 0.4932 - val_accuracy: 0.7641
Epoch 13/50
938/938 [==============================] - 4s 4ms/step - loss: 0.3921 - accuracy: 0.7409 - val_loss: 0.4890 - val_accuracy: 0.7586
Epoch 14/50
938/938 [==============================] - 5s 5ms/step - loss: 0.3819 - accuracy: 0.7506 - val_loss: 0.5116 - val_accuracy: 0.7479
Epoch 15/50
938/938 [==============================] - 4s 5ms/step - loss: 0.3762 - accuracy: 0.7511 - val_loss: 0.5038 - val_accuracy: 0.7503
Epoch 16/50
938/938 [==============================] - 4s 4ms/step - loss: 0.3729 - accuracy: 0.7534 - val_loss: 0.4959 - val_accuracy: 0.7618
Epoch 17/50
938/938 [==============================] - 4s 5ms/step - loss: 0.3656 - accuracy: 0.7598 - val_loss: 0.4865 - val_accuracy: 0.7678
Epoch 18/50
938/938 [==============================] - 5s 5ms/step - loss: 0.3609 - accuracy: 0.7579 - val_loss: 0.4696 - val_accuracy: 0.7703
Epoch 19/50
938/938 [==============================] - 4s 4ms/step - loss: 0.3575 - accuracy: 0.7641 - val_loss: 0.4403 - val_accuracy: 0.7875
Epoch 20/50
938/938 [==============================] - 4s 4ms/step - loss: 0.3525 - accuracy: 0.7619 - val_loss: 0.4681 - val_accuracy: 0.7699
Epoch 21/50
938/938 [==============================] - 6s 6ms/step - loss: 0.3486 - accuracy: 0.7672 - val_loss: 0.4593 - val_accuracy: 0.7807
Epoch 22/50
938/938 [==============================] - 4s 4ms/step - loss: 0.3452 - accuracy: 0.7695 - val_loss: 0.4488 - val_accuracy: 0.7897
Epoch 23/50
938/938 [==============================] - 4s 4ms/step - loss: 0.3407 - accuracy: 0.7714 - val_loss: 0.4822 - val_accuracy: 0.7680
Epoch 24/50
938/938 [==============================] - 5s 5ms/step - loss: 0.3408 - accuracy: 0.7713 - val_loss: 0.4425 - val_accuracy: 0.7902
```
### Evaluating the Neural Network Model
```python
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy:.2f}')
```
Evaluates the neural network model on the test set and prints the test accuracy. 

**output:**
```
Test Accuracy: 0.79
```
### Making Predictions with the Neural Network Model
```python
y_test_pred = model.predict(X_test)
y_test_pred_classes = y_test_pred.argmax(axis=1)
```

**Output:**
```
625/625 [==============================] - 1s 1ms/step
```
```python
print(classification_report(y_test, y_test_pred_classes, target_names=['Weak', 'Medium', 'Strong']))
```
Prints the classification report for the test set, showing precision, recall, and F1-score for each class.

**Output**: 
```
              precision    recall  f1-score   support

        Weak       0.46      0.98      0.63      2700
      Medium       0.97      0.74      0.84     14852
      Strong       0.71      0.90      0.79      2448

    accuracy                           0.79     20000
   macro avg       0.72      0.87      0.75     20000
weighted avg       0.87      0.79      0.80     20000
```
### Predicting Password Strength with the Neural Network Model
```python
def predict_password_strength(password):
    password_transformed = vectorizer.transform([password]).toarray()
    strength_pred = model.predict(password_transformed)
    strength_class = strength_pred.argmax(axis=1)[0]
    return ['Weak', 'Medium', 'Strong'][strength_class]

new_password = "hhhhhhhhhhhccGGG_@FSJSK52424hhhhhhhhhhhhhhhhhhhhhhhhhhhhh"
predicted_strength = predict_password_strength(new_password)
print(f'The predicted strength of the password "{new_password}" is: {predicted_strength}')
```
Predicts the strength of a new password and prints the result. 

**Output**: 
```
The predicted strength of the password "hhhhhhhhhhhccGGG_@FSJSK52424hhhhhhhhhhhhhhhhhhhhhhhhhhhhh" is: Weak.
```

### Training the RandomForestClassifier
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

rf_model = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
```
Trains the RandomForestClassifier model.

### Evaluating the RandomForestClassifier Model
```python
y_test_pred_rf = rf_model.predict(X_test)
print(classification_report(y_test, y_test_pred_rf, target_names=['Weak', 'Medium', 'Strong']))
```
Prints the classification report for the test set, showing precision, recall, and F1-score for each class.

**Output**:
```
              precision    recall  f1-score   support

        Weak       0.94      0.62      0.75      2700
      Medium       0.92      0.99      0.95     14852
      Strong       0.96      0.86      0.91      2448

    accuracy                           0.92     20000
   macro avg       0.94      0.83      0.87     20000
weighted avg       0.93      0.92      0.92     20000

```

### Predicting Password Strength with the RandomForestClassifier Model
```python
def predict_password_strength_rf(password):
    password_transformed = vectorizer.transform([password]).toarray()
    strength_pred = rf_model.predict(password_transformed)[0]
    return ['Weak', 'Medium', 'Strong'][strength_pred]

new_password = "charif"
predicted_strength_rf = predict_password_strength_rf(new_password)
print(f'The predicted strength of the password "{new_password}" is: {predicted_strength_rf}')
```
Predicts the strength of a new password and prints the result.

**Output**: 
```
The predicted strength of the password "charif" is: Weak
```
## Conclusion

In this project, two different models were trained to predict password strength: a neural network and a RandomForestClassifier. Both models were evaluated on a test set, and their performances were compared. The neural network achieved a test accuracy of 79%, while the RandomForestClassifier achieved a higher accuracy of 92%. 

The notebook also includes functions to predict the strength of new passwords using the trained models, demonstrating the practical application of these models in real-world scenarios.

Feel free to clone this repository and experiment with the models and data. If you have any questions or suggestions, please open an issue or contact me directly.
