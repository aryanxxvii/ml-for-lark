import gradio as gr
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import phonemizer
import librosa
import math
import io
import base64
from strsimpy.jaro_winkler import JaroWinkler


def speechToPhonemeWS(audioAsB64):
    wav_data = base64.b64decode(audioAsB64.encode("utf-8"))
    processor = Wav2Vec2Processor.from_pretrained(
        "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
    )
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")

    waveform, sample_rate = librosa.load(
        io.BytesIO(wav_data), sr=16000
    )  # Downsample 44.1kHz to 8kHz

    input_values = processor(
        waveform, sampling_rate=sample_rate, return_tensors="pt"
    ).input_values

    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    speechToPhonemeTranscription = transcription[0]
    speechToPhonemeTranscription = speechToPhonemeTranscription.replace(" ", "")
    return speechToPhonemeTranscription


def speechToTextToPhonemeWS(audioAsB64):
    wav_data = base64.b64decode(audioAsB64.encode("utf-8"))

    waveform, sample_rate = librosa.load(
        io.BytesIO(wav_data), sr=16000
    )  # Downsample 44.1kHz to 8kHz
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    input_values = processor(
        waveform, sampling_rate=sample_rate, return_tensors="pt"
    ).input_values

    logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    speechToTextTranscription = processor.batch_decode(predicted_ids)

    graphemeToPhonemeTranscription = phonemizer.phonemize(speechToTextTranscription[0])
    graphemeToPhonemeTranscription = graphemeToPhonemeTranscription.replace(" ", "")
    return [speechToTextTranscription[0], graphemeToPhonemeTranscription]


def similarity(S2P, G2P2T):
    jarowinkler = JaroWinkler()
    similarity_score = jarowinkler.similarity(S2P, G2P2T)
    return similarity_score


def similarityScoreToBand(similarity_score):
    if similarity_score >= 0.91:
        return 9
    elif similarity_score >= 0.81:
        return 8
    elif similarity_score >= 0.73:
        return 7
    elif similarity_score >= 0.65:
        return 6
    elif similarity_score >= 0.60:
        return 5
    elif similarity_score >= 0.46:
        return 4
    elif similarity_score >= 0.35:
        return 3
    elif similarity_score >= 0.1:
        return 2
    else:
        return 1


def lark(audioAsB64):
    s2p = speechToPhonemeWS(audioAsB64)
    [s2t, s2t2p] = speechToTextToPhonemeWS(audioAsB64)
    ss = similarity(s2t2p, s2p)
    band = similarityScoreToBand(ss)
    return [ss, band, s2t]


iface = gr.Interface(fn=lark, inputs="text", outputs=["text", "text", "text"])
iface.launch()
