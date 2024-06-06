from typing import Iterable
import time
from io import BytesIO
import requests
from pytube import YouTube
from groq.types.chat import ChatCompletionMessageParam
from llama_index.core import Document, VectorStoreIndex
from config import GROQ_CLIENT, EMBED_MODEL, VECTOR_INDEX, PIPELINE

def combine_text_with_markers_and_speaker(data):
    combined_text = ""
    for item in data:
        speaker_text = " ".join(sentence["text"] for sentence in item["sentences"])
        speaker_info = f"Speaker {item['speaker']}:"
        combined_text += f"{speaker_info} {speaker_text}\n"
    return combined_text

def read_from_url(url: str) -> BytesIO:
    res = requests.get(url)
    audio_bytes = BytesIO(res.content)
    return audio_bytes

def read_from_youtube(url: str):
    yt = YouTube(url)
    video = yt.streams.filter(only_audio=True, mime_type="audio/webm").first()
    
    if video is None:
        raise ValueError("No audio/webm stream found for the given YouTube URL.")
    
    buffer = BytesIO()
    video.stream_to_buffer(buffer)
    buffer.seek(0)
    
    audio_data = buffer.read()
    
    print(f"Audio retrieved as audio/webm (mimetype: {video.mime_type})")
    
    return BytesIO(audio_data)

def prerecorded(source, model: str, options: dict[str, str] = None) -> None:
    print(f"Source: {source} ")
    start = time.time()
    audio_bytes: BytesIO = source['buffer']
    file_type = source.get("mimetype", "audio/wav")
    if not file_type:
        file_type = "audio/wav"
    file_type = file_type.split("/")[1]
    print(f"Final filetype: {file_type}")
    transcription = GROQ_CLIENT.audio.transcriptions.create(
        file=(f"audio.{file_type}", audio_bytes.read()),
        model=model,
    )
    end = time.time()
    audio_bytes.seek(0)
    return {
        'text':transcription.text,
        'time_taken': end - start
    }

def create_vectorstore(transcript: str):
    global VECTOR_INDEX
    nodes = PIPELINE.run(documents=[Document(text=transcript)])
    globals()['VECTOR_INDEX'] = VectorStoreIndex(embed_model=EMBED_MODEL, nodes=nodes)
    return VECTOR_INDEX

def chat_stream(model: str, messages: Iterable[ChatCompletionMessageParam], **kwargs):
    # Retrieve documents from the vectorstore
    stream_response = GROQ_CLIENT.chat.completions.create(
        messages=messages,
        model=model,
        stream=True,
        **kwargs
    )

    for chunk in stream_response:
        tokens = chunk.choices[0].delta.content
        if tokens:
            yield tokens
        if x_groq := chunk.x_groq:
            if not x_groq.usage:
                continue
            usage = x_groq.usage
            completion_time = usage.completion_time
            completion_tokens = usage.completion_tokens
            tps = completion_tokens/completion_time
            yield f"\n\n_Tokens/sec: {round(tps, 2)}_"