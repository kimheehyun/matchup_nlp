import speech_recognition as sr
import tempfile
import os
import soundfile as sf
import librosa
import numpy as np
from sklearn.cluster import KMeans

# 오디오 특징 추출 (피치 + 볼륨)
def extract_features(y, sr):
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
    volume = np.mean(magnitudes)
    return [pitch, volume]

#  화자 분리 + 음성 인식
def transcribe_audio_with_diarization(audio_file):
    # Recognizer 객체
    recognizer = sr.Recognizer()

   
    y, sr_rate = librosa.load(audio_file, sr=None)

    # 오디오를 3초 단위로 분할
    segment_length = 3 * sr_rate
    segments = [y[i:i + segment_length] for i in range(0, len(y), segment_length)]

    # 각 세그먼트 특징 추출
    features = [extract_features(seg, sr_rate) for seg in segments]

    # KMeans로 2명 화자 클러스터링
    kmeans = KMeans(n_clusters=2, random_state=0).fit(features)

    results = []

    #  각 세그먼트별 STT 수행
    for i, segment in enumerate(segments):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            sf.write(tmp_wav.name, segment, sr_rate)
            tmp_path = tmp_wav.name

        try:
            with sr.AudioFile(tmp_path) as source:
                audio_data = recognizer.record(source)
                try:
                    text = recognizer.recognize_google(audio_data, language="ko-KR")
                    speaker = f"화자{kmeans.labels_[i]+1}"
                    results.append(f"{speaker}: {text}")
                except sr.UnknownValueError:
                    results.append(f"화자{kmeans.labels_[i]+1}: [인식 불가]")
                except sr.RequestError as e:
                    results.append(f"화자{kmeans.labels_[i]+1}: [STT 오류 {e}]")
        finally:
            os.unlink(tmp_path)

    return results


#   테스트
if __name__ == "__main__":
    audio_path = "sample.wav"
    summary = transcribe_audio_with_diarization(audio_path)
    for line in summary:
        print(line)

