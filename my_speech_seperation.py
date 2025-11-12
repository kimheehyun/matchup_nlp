import speech_recognition as sr
import librosa
import numpy as np
from sklearn.cluster import KMeans
import soundfile as sf
import tempfile
import os


# 🔹 librosa 이용한 특성 추출 함수
def extract_features(y, sample_rate):
    pitches, magnitudes = librosa.piptrack(y=y, sr=sample_rate)
    pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
    volume = np.mean(magnitudes)
    return [pitch, volume]


# 🔹 화자 분리 + 음성 인식
def transcribe_audio_with_diarization(audio_file):
    # 1️⃣ 오디오 로드
    y, sample_rate = librosa.load(audio_file, sr=None)

    # 2️⃣ 인식기 초기화
    recognizer = sr.Recognizer()

    # 3️⃣ 오디오를 3초 단위로 분할
    segment_length = 3 * sample_rate
    segments = [y[i:i + int(segment_length)] for i in range(0, len(y), int(segment_length))]

    # 4️⃣ 각 세그먼트의 피치/볼륨 특징 추출
    features = [extract_features(segment, sample_rate) for segment in segments]

    # 5️⃣ K-Means로 화자 클러스터링
    kmeans = KMeans(n_clusters=2, random_state=0).fit(features)

    results = []

    # 6️⃣ 각 세그먼트마다 음성 인식 수행
    for i, segment in enumerate(segments):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            sf.write(temp_wav.name, segment, sample_rate)

        try:
            with sr.AudioFile(temp_wav.name) as source:
                audio = recognizer.record(source)
            text = recognizer.recognize_google(audio, language="ko-KR")

            speaker = f"화자{kmeans.labels_[i] + 1}"
            results.append(f"{speaker}: {text}")

        except sr.UnknownValueError:
            print("음성을 인식할 수 없습니다.")
        except sr.RequestError as e:
            print(f"음성 인식 서비스 오류: {e}")
        finally:
            os.unlink(temp_wav.name)

    return results
