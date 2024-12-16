import os
import numpy as np
import pandas as pd
import librosa
import whisper
from whisper import load_model, pad_or_trim, load_audio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import joblib  # 用于保存和加载模型
import matplotlib.pyplot as plt


# Whisper speech recognition
def recognize_whisper(audio_path):
    model = load_model("base")
    audio = load_audio(audio_path)
    audio = pad_or_trim(audio)
    result = model.transcribe(audio)
    return result["text"]


# Audio to text conversion
def audio_to_text(audio_path, output_text_path):
    print(f"Start recognizing {audio_path} ...")
    model = whisper.load_model("base").to("cuda")
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    result = model.transcribe(audio)
    text = result['text']

    with open(output_text_path, "w") as f:
        f.write(text)
    return text


# Extract audio features
def extract_audio_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    rms = np.mean(librosa.feature.rms(y=y))
    return np.hstack([mfcc, zcr, rms])


# Extract text features
def extract_text_features(texts):
    vectorizer = TfidfVectorizer(max_features=17)
    return vectorizer.fit_transform(texts).toarray()


# Train and evaluate the model
def train_and_evaluate(X_text, X_audio, y, model_save_path=None):
    X_text_train, X_text_test, X_audio_train, X_audio_test, y_train, y_test = train_test_split(
        X_text, X_audio, y, test_size=0.2, random_state=32
    )
    print(f"Number of text features: {X_text_train.shape[1]}")
    print(f"Audio feature count: {X_audio_train.shape[1]}")

    # Audio classifiers
    scaler = StandardScaler()
    X_audio_train = scaler.fit_transform(X_audio_train)
    X_audio_test = scaler.transform(X_audio_test)

    models_audio = {
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "GaussianNB": GaussianNB(),
    }

    predictions_audio = {}
    accuracy_audio = {}

    for name, model in models_audio.items():
        print(f"Training {name} (Audio features)...")
        model.fit(X_audio_train, y_train)
        pred = model.predict(X_audio_test)
        predictions_audio[name] = pred
        accuracy = accuracy_score(y_test, pred)
        accuracy_audio[name] = accuracy
        print(f"{name} (Audio features) Accuracy: {accuracy:.2f}")

    models_text = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=15, random_state=32),
    }

    predictions_text = {}
    accuracy_text = {}

    for name, model in models_text.items():
        print(f"Training {name} (Text features)...")
        model.fit(X_text_train, y_train)
        pred = model.predict(X_text_test)
        predictions_text[name] = pred
        accuracy = accuracy_score(y_test, pred)
        accuracy_text[name] = accuracy
        print(f"{name} (Text features) Accuracy: {accuracy:.2f}")

    accuracies = {**accuracy_text, **accuracy_audio}

    final_predictions = []
    for i in range(len(y_test)):
        votes = []
        for name, pred in {**predictions_text, **predictions_audio}.items():
            weight = accuracies.get(name, 0)
            votes.extend([pred[i]] * int(weight * 10))
        final_predictions.append(1 if votes.count(1) > votes.count(0) else 0)

    accuracy = accuracy_score(y_test, final_predictions)
    print(f"Final voting classification accuracy: {accuracy:.2f}")

    cm = confusion_matrix(y_test, final_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Deceptive", "True"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

    # Save the model if specified
    if model_save_path:
        print(f"Saving model to {model_save_path}")
        joblib.dump({"text_models": models_text, "audio_models": models_audio, "scaler": scaler}, model_save_path)


# Use the trained model to predict new data
def predict_with_model(model_load_path, X_text, X_audio):
    print(f"Loading model from {model_load_path}")
    model_data = joblib.load(model_load_path)

    models_text = model_data["text_models"]
    models_audio = model_data["audio_models"]
    scaler = model_data["scaler"]

    # Standardize audio features
    X_audio = scaler.transform(X_audio)

    predictions_text = {}
    for name, model in models_text.items():
        pred = model.predict(X_text)
        predictions_text[name] = pred

    predictions_audio = {}
    for name, model in models_audio.items():
        pred = model.predict(X_audio)
        predictions_audio[name] = pred

    # Voting mechanism for final prediction
    final_predictions = []
    for i in range(len(X_text)):
        votes = []
        for name, pred in {**predictions_text, **predictions_audio}.items():
            votes.extend([pred[i]] * 10)  # Equal weight for all models here
        final_predictions.append(1 if votes.count(1) > votes.count(0) else 0)

    return final_predictions


# Main program
def main():
    mode = input("Enter 'train' to train the model or 'predict' to use the model: ").strip().lower()

    if mode == "train":
        print("Start loading data")
        csv_file = "CBU0521DD_stories_attributes.csv"
        folder_path = "CBU0521DD_stories"
        df = pd.read_csv(csv_file)

        audio_files = df["filename"].values
        labels = df["Story_type"].map({"True Story": 1, "Deceptive Story": 0}).values

        texts = []
        print("Start audio to text conversion")
        for i, file in enumerate(audio_files):
            file_path = os.path.join(folder_path, file)
            output_text_path = os.path.join(folder_path, f"{file[:-4]}.txt")

            if os.path.exists(output_text_path):
                with open(output_text_path, "r") as f:
                    text = f.read()
            else:
                text = audio_to_text(file_path, output_text_path)
            texts.append(text)
        print("Audio to text conversion completed")

        print("Extract Text Features")
        X_text = extract_text_features(texts)

        print("Extract audio features")
        X_audio = np.array([extract_audio_features(os.path.join(folder_path, file)) for file in audio_files])

        print("Training and evaluating models")
        model_save_path = "trained_model.pkl"
        train_and_evaluate(X_text, X_audio, labels, model_save_path)

    elif mode == "predict":
        # Load model
        model_load_path = "trained_model.pkl"
        if os.path.exists(model_load_path) is False:
            print("Model does not exist")
            return
        print("Start loading data for prediction")
        test_folder_path = "predict"
        audio_files = [f for f in os.listdir(test_folder_path) if f.endswith(".wav")]
        texts = []
        file_name = []
        for file in audio_files:
            file_path = os.path.join(test_folder_path, file)
            file_name.append(file)
            output_text_path = os.path.join(test_folder_path, f"{file[:-4]}.txt")

            if os.path.exists(output_text_path):
                with open(output_text_path, "r") as f:
                    text = f.read()
            else:
                text = audio_to_text(file_path, output_text_path)
            texts.append(text)
        print("Audio to text conversion completed")

        print("Extract Text Features")
        X_text = extract_text_features(texts)

        print("Extract audio features")
        X_audio = np.array([extract_audio_features(os.path.join(test_folder_path, file)) for file in audio_files])

        predictions = predict_with_model(model_load_path, X_text, X_audio)
        with open("result.txt", "w") as f:
            for i in range(len(predictions)):
                print(f"Data: {file_name[i]}, Prediction: {predictions[i]}")
                f.write(f"Data: {file_name[i]}, Prediction: {predictions[i]}\n")

    else:
        print("Invalid option! Please enter 'train' or 'predict'.")


if __name__ == "__main__":
    main()
