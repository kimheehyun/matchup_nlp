import streamlit as st
import numpy as np
import requests
import soundfile as sf
from my_speech_seperation import transcribe_audio_with_diarization
from groq import Groq
import os

st.write("### 회의록 요약 챗봇 (Groq LLM 버전) ###") 

uploaded_file = st.file_uploader("Upload a sound file", type=["wav"])

if uploaded_file:
    summary_list = transcribe_audio_with_diarization(uploaded_file)
    text_content = "\n".join(summary_list)

    # Groq API 키 설정
    groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
    client = Groq(api_key=groq_api_key)

    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    # 회의 요약 함수
    def summarize_meeting_with_groq(transcript):
        prompt = f"""
        다음은 회의의 내용입니다.
        핵심 요점을 정리하고 간결하게 요약해주세요.

        회의 내용:
        {transcript}
        """

        response = client.chat.completions.create(
            model="allama-3.3-70b-versatile",   
            messages=[
                {"role": "system", "content": "너는 전문 회의 요약 비서야."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )

        return response.choices[0].message.content.strip()

    result = summarize_meeting_with_groq(text_content)

    st.write("### 요약 결과 ###")
    st.write(result)

else:
    st.warning("오디오 파일을 업로드해주세요!")




