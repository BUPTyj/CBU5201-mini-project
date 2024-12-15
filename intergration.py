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
from sklearn.naive_bayes import GaussianNB  # 导入高斯朴素贝叶斯
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# Whisper 语音识别
def recognize_whisper(audio_path):
    model = load_model("base")
    audio = load_audio(audio_path)
    audio = pad_or_trim(audio)
    result = model.transcribe(audio)
    return result["text"]


# 音频转文字
def audio_to_text(audio_path, output_text_path):
    print(f"开始识别 {audio_path} ...")
    model = whisper.load_model("base").to("cuda")  # 使用 Whisper base 模型
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    result = model.transcribe(audio)
    text = result['text']

    # 保存转录的文本
    with open(output_text_path, "w") as f:
        f.write(text)

    return text


# 提取音频特征
def extract_audio_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    rms = np.mean(librosa.feature.rms(y=y))
    return np.hstack([mfcc, zcr, rms])


# 提取文本特征
def extract_text_features(texts):
    vectorizer = TfidfVectorizer(max_features=50)
    return vectorizer.fit_transform(texts).toarray()


# 分类与投票机制 (加入权重)
def train_and_evaluate(X_text, X_audio, y, use_audio_to_text=True):
    # 如果不使用文本特征，直接跳过文本相关部分的处理
    if use_audio_to_text:
        X_text_train, X_text_test, X_audio_train, X_audio_test, y_train, y_test = train_test_split(
            X_text, X_audio, y, test_size=0.2, random_state=32
        )
        print(f"文本特征数: {X_text_train.shape[1]}")  # 输出文本特征数
        print(f"音频特征数: {X_audio_train.shape[1]}")  # 输出音频特征数
    else:
        # 仅使用音频特征
        X_audio_train, X_audio_test, y_train, y_test = train_test_split(
            X_audio, y, test_size=0.2, random_state=32
        )
        print(f"音频特征数: {X_audio_train.shape[1]}")  # 输出音频特征数
        X_text_train, X_text_test = np.array([]), np.array([])  # 如果没有文本特征，设为空

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
    if use_audio_to_text:
        models_text = {
            "LogisticRegression": LogisticRegression(max_iter=500),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
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
    accuracies = {**accuracy_text, **accuracy_audio} if use_audio_to_text else accuracy_audio

    # 投票机制
    final_predictions = []
    for i in range(len(y_test)):
        votes = []
        for name, pred in {**predictions_text, **predictions_audio}.items() if use_audio_to_text else predictions_audio.items():
            if use_audio_to_text or name in predictions_audio:
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
def main(num_samples=10, use_audio_to_text=True):  # 添加 use_audio_to_text 参数
    # 加载数据
    print("开始加载数据")
    csv_file = "CBU0521DD_stories_attributes.csv"
    folder_path = "CBU0521DD_stories"
    df = pd.read_csv(csv_file)

    # 限制使用的前 num_samples 个数据
    audio_files = df["filename"].values[:num_samples]
    labels = df["Story_type"].map({"True Story": 1, "Deceptive Story": 0}).values[:num_samples]

    # 音频转文本
    texts = []
    if use_audio_to_text:
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
    else:
        X_text = np.array([])  # 如果不使用文本特征，设置为空数组

    # 提取音频特征
    print("提取音频特征...")
    X_audio = np.array([extract_audio_features(os.path.join(folder_path, file)) for file in audio_files])

    # 分类与评估
    train_and_evaluate(X_text, X_audio, labels, use_audio_to_text)


if __name__ == "__main__":
    # 指定前多少个数据进行处理，并控制是否使用音频转文本
    num_samples = int(input("请输入要处理的样本数量（例如 10）: ").strip())
    use_audio_to_text = input("是否使用音频转文本 (y/n): ").strip().lower() == 'y'
    main(num_samples, use_audio_to_text)
