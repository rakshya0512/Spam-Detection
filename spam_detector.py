# spam_detector.py
"""
Email Spam Detection using a Deep Learning model (Bi-LSTM + attention)
Requirements:
    pip install tensorflow scikit-learn pandas matplotlib
Dataset:
    CSV file 'spam.csv' with columns: 'text' and 'label' (label can be 'spam'/'ham' or 1/0)
Usage:
    python spam_detector.py
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# -----------------------------
# Config / Hyperparameters
# -----------------------------
DATA_PATH = "spam.csv"       # update path if needed
RANDOM_SEED = 42
MAX_VOCAB_SIZE = 30000
MAX_SEQUENCE_LENGTH = 300   # truncate/pad email text tokens to this length
EMBEDDING_DIM = 128
BATCH_SIZE = 64
EPOCHS = 8
LEARNING_RATE = 1e-3

# -----------------------------
# Utilities: basic text cleaning
# -----------------------------
def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    # lowercase
    s = s.lower()
    # remove urls and emails
    s = re.sub(r'\S+@\S+\.\S+', ' ', s)
    s = re.sub(r'http\S+|www\.\S+', ' ', s)
    # remove non-alphanumeric (keep spaces)
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    # collapse multiple spaces
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# -----------------------------
# Load dataset
# -----------------------------
def load_dataset(path=DATA_PATH):
    df = pd.read_csv(path, encoding="latin1")

    # Handle common column names
    if "v1" in df.columns and "v2" in df.columns:
        df = df.rename(columns={"v1": "label", "v2": "text"})
    
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must include 'text' and 'label' columns.")

    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].map(lambda x: 1 if str(x).strip().lower() in ["1","spam","true","yes"] else 0)
    df["text"] = df["text"].astype(str).map(clean_text)
    return df


# -----------------------------
# Build text vectorization (tokenizer)
# -----------------------------
def build_text_vectorizer(texts, max_tokens=MAX_VOCAB_SIZE, seq_len=MAX_SEQUENCE_LENGTH):
    vectorizer = layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode='int',
        output_sequence_length=seq_len,
        standardize=None  # we already cleaned text
    )
    vectorizer.adapt(texts)
    return vectorizer

# -----------------------------
# Attention layer (simple)
# -----------------------------
class AttentionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1),
                                 initializer="glorot_uniform",
                                 trainable=True,
                                 name="attn_w")
        super().build(input_shape)

    def call(self, inputs, mask=None):
        # inputs: (batch, time, features)
        # compute scores
        scores = tf.squeeze(tf.tensordot(inputs, self.W, axes=1), axis=-1)  # (batch, time)
        if mask is not None:
            # mask is boolean (batch, time)
            paddings = tf.fill(tf.shape(scores), -1e9)
            scores = tf.where(mask, scores, paddings)
        alphas = tf.nn.softmax(scores, axis=1)  # (batch, time)
        alphas = tf.expand_dims(alphas, axis=-1)  # (batch, time, 1)
        context = tf.reduce_sum(inputs * alphas, axis=1)  # (batch, features)
        return context

    def get_config(self):
        base = super().get_config()
        return base

# -----------------------------
# Build model
# -----------------------------
def build_model(vectorizer):
    inputs = layers.Input(shape=(1,), dtype=tf.string)  # raw string input
    x = vectorizer(inputs)                             # (batch, seq_len)
    x = layers.Embedding(input_dim=vectorizer.vocabulary_size(), output_dim=EMBEDDING_DIM, mask_zero=True)(x)
    x = layers.SpatialDropout1D(0.2)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    attn = AttentionLayer()(x)
    x = layers.Dense(64, activation='relu')(attn)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    return model

# -----------------------------
# Training & evaluation pipeline
# -----------------------------
def train_eval(df):
    # split
    X = df['text'].values
    y = df['label'].values
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED, stratify=y_temp)

    vectorizer = build_text_vectorizer(X_train, max_tokens=MAX_VOCAB_SIZE, seq_len=MAX_SEQUENCE_LENGTH)
    model = build_model(vectorizer)

    # handle class imbalance if present
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = None
    if len(np.unique(y_train)) == 2:
        cw = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        class_weights = {i: cw_i for i, cw_i in enumerate(cw)}

    # callbacks
    cb = [
        callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
    ]

    # model.fit expects inputs shaped like (batch,) of strings when using a vectorizer layer
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=cb,
        verbose=2
    )

    # Evaluate
    preds_prob = model.predict(X_test, batch_size=BATCH_SIZE).ravel()
    preds = (preds_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    cm = confusion_matrix(y_test, preds)

    print("\nTest results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1: {f1:.4f}")
    print("Confusion matrix (rows: true, cols: pred):")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_test, preds, digits=4))

    # Plot training curves
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.title('Loss')

    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.legend()
    plt.title('Accuracy')
    plt.tight_layout()
    plt.show()

    # Save model and vectorizer
    model_save_path = "spam_bilstm_model"
    model.save("spam_bilstm_model.keras")
    # Save vectorizer via Keras serialization (it is a layer inside the model, but if you want vectorizer alone:)
    # Create a small model that contains the vectorizer
    vect_model = tf.keras.Sequential([layers.Input(shape=(1,), dtype=tf.string), vectorizer])
    vect_model.save("text_vectorizer")
    print(f"Saved model to '{model_save_path}' and vectorizer to 'text_vectorizer'.")

    # Return for further use
    return model, vectorizer

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print("Loading dataset...")
    df = load_dataset(DATA_PATH)
    print(f"Loaded {len(df)} examples. Spam ratio: {df['label'].mean():.3f}")
    model, vectorizer = train_eval(df)
