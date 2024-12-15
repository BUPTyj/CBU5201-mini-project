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
    # Using Whisper base model and GPU acceleration
    model = whisper.load_model("base").to("cuda")
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    # Obtain the conversion result
    result = model.transcribe(audio)
    text = result['text']

    # Save transcribed text
    with open(output_text_path, "w") as f:
        f.write(text)
    return text


# Extract audio features
def extract_audio_features(file_path):
    # Loading audio files using the librosa library
    y, sr = librosa.load(file_path, sr=None)
    # Calculate Mel frequency cepstral coefficients
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    # Calculate the zero crossing rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    # Calculate root mean square
    rms = np.mean(librosa.feature.rms(y=y))
    # Splicing to obtain audio features
    return np.hstack([mfcc, zcr, rms])


# Extract text features
def extract_text_features(texts):
    vectorizer = TfidfVectorizer(max_features=50)
    return vectorizer.fit_transform(texts).toarray()


# Classification and voting mechanism (with added weights)
def train_and_evaluate(X_text, X_audio, y):
    X_text_train, X_text_test, X_audio_train, X_audio_test, y_train, y_test = train_test_split(
        X_text, X_audio, y, test_size=0.2, random_state=32
    )
    print(f"文本特征数: {X_text_train.shape[1]}")  # 输出文本特征数
    print(f"音频特征数: {X_audio_train.shape[1]}")  # 输出音频特征数

    # 音频分类器
    scaler = StandardScaler()
    X_audio_train = scaler.fit_transform(X_audio_train)
    X_audio_test = scaler.transform(X_audio_test)
    models_audio = {
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "GaussianNB": GaussianNB(),  # 使用高斯朴素贝叶斯
    }
    predictions_audio = {}
    accuracy_audio = {}
    for name, model in models_audio.items():
        print(f"训练 {name} (音频特征)...")
        model.fit(X_audio_train, y_train)
        pred = model.predict(X_audio_test)
        predictions_audio[name] = pred
        accuracy = accuracy_score(y_test, pred)
        accuracy_audio[name] = accuracy
        print(f"{name} (音频特征) 准确率: {accuracy:.2f}")

    # 如果使用文本特征，训练文本分类器
    models_text = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=15, random_state=32),
    }
    predictions_text = {}
    accuracy_text = {}
    for name, model in models_text.items():
        print(f"训练 {name} (文本特征)...")
        model.fit(X_text_train, y_train)
        pred = model.predict(X_text_test)
        predictions_text[name] = pred
        accuracy = accuracy_score(y_test, pred)
        accuracy_text[name] = accuracy
        print(f"{name} (文本特征) 准确率: {accuracy:.2f}")

    # 合并所有模型的准确率
    accuracies = {**accuracy_text, **accuracy_audio}

    # 投票机制
    final_predictions = []
    for i in range(len(y_test)):
        votes = []
        for name, pred in {**predictions_text, **predictions_audio}.items():
            weight = accuracies.get(name, 0)
            votes.extend([pred[i]] * int(weight * 10))
        final_predictions.append(1 if votes.count(1) > votes.count(0) else 0)

    # 输出最终准确率
    accuracy = accuracy_score(y_test, final_predictions)
    print(f"最终投票分类准确率: {accuracy:.2f}")

    # 绘制混淆矩阵
    cm = confusion_matrix(y_test, final_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Deceptive", "True"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()


# 主程序
def main():
    # 加载数据
    print("开始加载数据")
    csv_file = "CBU0521DD_stories_attributes.csv"
    folder_path = "CBU0521DD_stories"
    df = pd.read_csv(csv_file)

    # 限制使用的前 num_samples 个数据
    audio_files = df["filename"].values
    labels = df["Story_type"].map({"True Story": 1, "Deceptive Story": 0}).values

    # 音频转文本
    texts = []
    print("开始音频转文本")
    for i, file in enumerate(audio_files):
        file_path = os.path.join(folder_path, file)
        output_text_path = os.path.join(folder_path, f"{file[:-4]}.txt")

        # 检查是否存在已转换的文本
        if os.path.exists(output_text_path):
            with open(output_text_path, "r") as f:
                text = f.read()
        else:
            text = audio_to_text(file_path, output_text_path)
        texts.append(text)
    print("音频转文本完成")

    # 提取文本特征
    print("提取文本特征...")
    X_text = extract_text_features(texts)

    # 提取音频特征
    print("提取音频特征...")
    X_audio = np.array([extract_audio_features(os.path.join(folder_path, file)) for file in audio_files])

    # 分类与评估
    train_and_evaluate(X_text, X_audio, labels)


if __name__ == "__main__":
    main()
