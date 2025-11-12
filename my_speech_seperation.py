from pyannote.audio import Pipeline
import speech_recognition as sr
import tempfile
import os
import soundfile as sf


def transcribe_audio_with_diarization(audio_file):
    """
    pyannote.audio를 사용한 화자 분리 + 음성 인식
    """

    # 1️⃣ 임시 wav 파일 저장 (Streamlit의 UploadedFile 등 대응)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    try:
        # 2️⃣ pyannote pipeline 불러오기 (HuggingFace 인증 토큰 필요)
        # huggingface-cli login 으로 미리 토큰 등록해두어야 합니다.
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

        # 3️⃣ 화자 분리 수행
        diarization = pipeline(tmp_path)

        recognizer = sr.Recognizer()
        results = []

        # 4️⃣ 각 화자 세그먼트별 음성 인식
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start_time = turn.start
            end_time = turn.end

            # 해당 구간만 잘라 임시 wav로 저장
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as segment_wav:
                y, sr_rate = sf.read(tmp_path)
                start_sample = int(start_time * sr_rate)
                end_sample = int(end_time * sr_rate)
                sf.write(segment_wav.name, y[start_sample:end_sample], sr_rate)

                # STT 수행
                with sr.AudioFile(segment_wav.name) as source:
                    audio_data = recognizer.record(source)
                    try:
                        text = recognizer.recognize_google(audio_data, language="ko-KR")
                        results.append(f"{speaker}: {text}")
                    except sr.UnknownValueError:
                        pass
                    except sr.RequestError as e:
                        print(f"Google STT 오류: {e}")
                os.unlink(segment_wav.name)

        return results

    finally:
        # 임시 파일 정리
        os.unlink(tmp_path)


