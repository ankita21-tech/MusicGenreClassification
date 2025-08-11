import os
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import EarlyStopping

CSV_PATH = 'model_generation/features_30_sec.csv'
MODEL_DIR = 'model'
MODEL_FILE = os.path.join(MODEL_DIR, 'genre_classifier.keras')
ENCODER_FILE = os.path.join(MODEL_DIR, 'label_encoder.pkl')
EPOCHS = 50
BATCH_SIZE = 32
TEST_SIZE = 0.2

# saving our model at specific path here
os.makedirs(MODEL_DIR, exist_ok=True)
    
# reading the dataset from the given path
df = pd.read_csv(CSV_PATH)

# Drop non-feature columns
X = df.drop(columns=['filename', 'label'])
y = df['label']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save the encoder
with open(ENCODER_FILE, 'wb') as f:
    pickle.dump(label_encoder, f)
print(f"Label encoder saved to: {ENCODER_FILE}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=TEST_SIZE, stratify=y_encoded, random_state=42
)

# ✅ Step 4: Build model
def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = build_model(input_shape=X.shape[1:], num_classes=len(label_encoder.classes_))
model.summary()             # for printing the model's summary

print("🚀 Training model...")
early_stop = EarlyStopping(patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# ✅ Step 6: Evaluate model
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.2f}")

# ✅ Step 7: Generate confusion matrix & classification report
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_test, y_pred)
labels = label_encoder.classes_

print("\n Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=labels, zero_division=0))

# ✅ Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# ✅ Step 8: Save model
model.save(MODEL_FILE)
print(f"💾 Model saved at: {MODEL_FILE}")