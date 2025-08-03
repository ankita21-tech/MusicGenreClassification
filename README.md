# 🎵 Music Genre Classification → Deep Learning 💻

▶️ This project uses a Deep Learning model to Classify the given music file, but the file must be in extension *(.wav)* format. But for using it must have the required modules to be installed together with the Python Environment !

## 🖥️ Project Details

📌 In this project, I have categorized everything in the available different folder in the structure of the Project, also each folder's explaination regarding details is been explained below 👇🏻

- **model** → Contains the model file with extension *.keras* and label-encoder file *.pkl* regarding model's usage in future.

- **model_generation** → In this folder, all the files required to create the folder is been given also the Python Script file for generation of the **DL Model** for Music Genre Classification.

- **model_testing** → This folder is enriched with all the files required for testing the Deep Learning Model together with the file required for testing the Model.

- **Visuals** → Contains all the required files as a proof, also the details of Deep Learning Model, regarding different of its pointers and requirements for measures.

- **README.md** & **SECURITY.md** → Also regarding the Detailed overview and the Security purpose of the Project these are some files required for it.

## 📄 Model's Details

- **Frameworks** → Tensorflow & Keras.
- - **Input Features**:  
  - MFCCs (20 mean & variance = 40)
  - Chroma, spectral, rolloff, zero crossing, tempo, harmony/percussive
  - Total: 58 features
- **Model Type** : Sequential Dense Neural Network
- **Loss Function**: Categorical Crossentropy
- **Optimizer** : Adam
- **Accuracy** : ~`XX%` (add your actual result)

## 📚 Dataset Details

- Source: [GTZAN Genre Collection](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification?resource=download)
- Processed via `librosa` to extract:
  - MFCCs
  - Chroma
  - Spectral Centroid/Bandwidth
  - RMS
  - Tempo
- Final dataset: `features_30_sec.csv`

## 🙌🏻 How to use it ?

▶️ First, you need to install the dependencies before running the model's files or testing model's file,

```bash

# installing the required modules before running the model's generation file
pip install librosa tensorflow pandas numpy scikit-learn

# for generating the model
cd model_generation
python model_generation.py

# redirecting to the model's directory
cd ..
cd model_testing
python model_test.py

```

## 📈 Evaluation

- **Confusion Matrix** : [Checkout]()
- **Classification Report** : [Checkout]()

## ✅ To-Do

- Add top-3 genre probability output
- Extend to support .mp3 via ffmpeg/librosa
- Build a web interface using Streamlit or Flask
- Train with YAMNet embeddings for improved performance

## 🧑🏻‍💻 Author

Abhay Chaudhary
[GitHub](https://github.com/ackwolver335)
[LinkdIn](https://www.linkedin.com/in/abhaychaudhary335/)
Email : abhaych335@gmail.com