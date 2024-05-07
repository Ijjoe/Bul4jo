import streamlit as st
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer, WhisperProcessor, pipeline
import itertools
from streamlit_extras import buy_me_a_coffee
from st_audiorec import st_audiorec
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from faster_whisper import WhisperModel
from io import BytesIO
import soundfile as sf
import itertools
import os
import nemo.collections.asr as nemo_asr
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

####JAMO####


start_time = time.monotonic()

INITIAL = 0x001
MEDIAL = 0x010
FINAL = 0x100
CHAR_LISTS = {
    INITIAL: list(map(chr, [
        0x3131, 0x3132, 0x3134, 0x3137, 0x3138, 0x3139,
        0x3141, 0x3142, 0x3143, 0x3145, 0x3146, 0x3147,
        0x3148, 0x3149, 0x314a, 0x314b, 0x314c, 0x314d,
        0x314e
    ])),
    MEDIAL: list(map(chr, [
        0x314f, 0x3150, 0x3151, 0x3152, 0x3153, 0x3154,
        0x3155, 0x3156, 0x3157, 0x3158, 0x3159, 0x315a,
        0x315b, 0x315c, 0x315d, 0x315e, 0x315f, 0x3160,
        0x3161, 0x3162, 0x3163
    ])),
    FINAL: list(map(chr, [
        0x3131, 0x3132, 0x3133, 0x3134, 0x3135, 0x3136,
        0x3137, 0x3139, 0x313a, 0x313b, 0x313c, 0x313d,
        0x313e, 0x313f, 0x3140, 0x3141, 0x3142, 0x3144,
        0x3145, 0x3146, 0x3147, 0x3148, 0x314a, 0x314b,
        0x314c, 0x314d, 0x314e
    ]))
}
CHAR_INITIALS = CHAR_LISTS[INITIAL]
CHAR_MEDIALS = CHAR_LISTS[MEDIAL]
CHAR_FINALS = CHAR_LISTS[FINAL]
CHAR_SETS = {k: set(v) for k, v in CHAR_LISTS.items()}
CHARSET = set(itertools.chain(*CHAR_SETS.values()))
CHAR_INDICES = {k: {c: i for i, c in enumerate(v)}
                for k, v in CHAR_LISTS.items()}


def is_hangul_syllable(c):
    return 0xac00 <= ord(c) <= 0xd7a3  # Hangul Syllables


def is_hangul_jamo(c):
    return 0x1100 <= ord(c) <= 0x11ff  # Hangul Jamo


def is_hangul_compat_jamo(c):
    return 0x3130 <= ord(c) <= 0x318f  # Hangul Compatibility Jamo


def is_hangul_jamo_exta(c):
    return 0xa960 <= ord(c) <= 0xa97f  # Hangul Jamo Extended-A


def is_hangul_jamo_extb(c):
    return 0xd7b0 <= ord(c) <= 0xd7ff  # Hangul Jamo Extended-B


def is_hangul(c):
    return (is_hangul_syllable(c) or
            is_hangul_jamo(c) or
            is_hangul_compat_jamo(c) or
            is_hangul_jamo_exta(c) or
            is_hangul_jamo_extb(c))


def is_supported_hangul(c):
    return is_hangul_syllable(c) or is_hangul_compat_jamo(c)


def check_hangul(c, jamo_only=False):
    if not ((jamo_only or is_hangul_compat_jamo(c)) or is_supported_hangul(c)):
        raise ValueError(f"'{c}' is not a supported hangul character. "
                         f"'Hangul Syllables' (0xac00 ~ 0xd7a3) and "
                         f"'Hangul Compatibility Jamos' (0x3130 ~ 0x318f) are "
                         f"supported at the moment.")


def get_jamo_type(c):
    check_hangul(c)
    assert is_hangul_compat_jamo(c), f"not a jamo: {ord(c):x}"
    return sum(t for t, s in CHAR_SETS.items() if c in s)


def split_syllable_char(c):
    check_hangul(c)
    if len(c) != 1:
        raise ValueError("한개만")

    init, med, final = None, None, None
    if is_hangul_syllable(c):
        offset = ord(c) - 0xac00
        x = (offset - offset % 28) // 28
    init, med, final = None, None, None
    if is_hangul_syllable(c):
        offset = ord(c) - 0xac00
        x = (offset - offset % 28) // 28
        init, med, final = x // 21, x % 21, offset % 28
        if not final:
            final = None
        else:
            final -= 1
    else:
        pos = get_jamo_type(c)
        if pos & INITIAL == INITIAL:
            pos = INITIAL
        elif pos & MEDIAL == MEDIAL:
            pos = MEDIAL
        elif pos & FINAL == FINAL:
            pos = FINAL
        idx = CHAR_INDICES[pos][c]
        if pos == INITIAL:
            init = idx
        elif pos == MEDIAL:
            med = idx
        else:
            final = idx
    return tuple(CHAR_LISTS[pos][idx] if idx is not None else None
                 for pos, idx in
                 zip([INITIAL, MEDIAL, FINAL], [init, med, final]))


def split_syllables(s, ignore_err=True, pad=None):

    def try_split(c):
        try:
            return split_syllable_char(c)
        except ValueError:
            if ignore_err:
                return (c,)
            raise ValueError(f"encountered an unsupported character: "
                             f"{c} (0x{ord(c):x})")

    s = map(try_split, s)
    if pad is not None:
        tuples = map(lambda x: tuple(pad if y is None else y for y in x), s)
    else:
        tuples = map(lambda x: filter(None, x), s)
    return "".join(itertools.chain(*tuples))



def join_jamos_char(init, med, final=None):
    chars = (init, med, final)
    for c in filter(None, chars):
        check_hangul(c, jamo_only=True)

    idx = tuple(CHAR_INDICES[pos][c] if c is not None else c
                for pos, c in zip((INITIAL, MEDIAL, FINAL), chars))
    init_idx, med_idx, final_idx = idx

    final_idx = 0 if final_idx is None else final_idx + 1
    return chr(0xac00 + 28 * 21 * init_idx + 28 * med_idx + final_idx)

def join_jamos(s, ignore_err=True):
    last_t = 0
    queue = []
    new_string = ""

    def flush(n=0):
        new_queue = []
        while len(queue) > n:
            new_queue.append(queue.pop())
        if len(new_queue) == 1:
            if not ignore_err:
                raise ValueError(f"invalid jamo character: {new_queue[0]}")
            result = new_queue[0]
        elif len(new_queue) >= 2:
            try:
                result = join_jamos_char(*new_queue)
            except (ValueError, KeyError):
                if not ignore_err:
                    raise ValueError(f"invalid jamo characters: {new_queue}")
                result = "".join(new_queue)
        else:
            result = None
        return result

    for c in s:
        if c not in CHARSET:
            if queue:
                new_c = flush() + c
            else:
                new_c = c
            last_t = 0
        else:
            t = get_jamo_type(c)
            new_c = None
            if t & FINAL == FINAL:
                if not (last_t == MEDIAL):
                    new_c = flush()
            elif t == INITIAL:
                new_c = flush()
            elif t == MEDIAL:
                if last_t & INITIAL == INITIAL:
                    new_c = flush(1)
                else:
                    new_c = flush()
            last_t = t
            queue.insert(0, c)
        if new_c:
            new_string += new_c
    if queue:
        new_string += flush()
    return new_string


####PIPE LINE####

pipe = pipeline(model="Ljrabbit/wav2vec2-large-xls-r-300m-korean-test0")

####TEXT COMBINE####

####STREAMLIT INTERFACE####


#### transcribe #####

#Whisper
def transcribe_whisper(audio_data):
        audio_buffer = BytesIO(audio_data)
        temp_filename = "temp_audio.wav"
        data, samplerate = sf.read(audio_buffer)
        sf.write(temp_filename, data, samplerate)
    
        segments, info = model.transcribe(temp_filename, beam_size=5)
        transcription = ""
    
        for segment in segments:
            transcription += f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n"

        language = info.language if info.language_probability > 0.5 else "Uncertain"
        return transcription, language
#Nemo
def transcribe_nemo(audio_data):
        audio_buffer = BytesIO(audio_data)
        data, samplerate = sf.read(audio_buffer)

        if len(data.shape) > 1 and data.shape[1] == 2:
            data = data.mean(axis=1)

        temp_filename = "temp.wav"
        sf.write(temp_filename, data, samplerate)
        text = asr_model.transcribe([temp_filename])[0]
        return text
        
#Wav2Vec
def transcribe(audio_data):
        result = pipe(audio_data)
        text = result["text"]
        composed_text = join_jamos(text)
        return composed_text
#Custom Whisper

feature_extractor = WhisperFeatureExtractor.from_pretrained("Dearlie/whisper-noise4")
tokenizer = WhisperTokenizer.from_pretrained("Dearlie/whisper-noise4", language="korean", task="transcribe")
processor = WhisperProcessor.from_pretrained("Dearlie/whisper-noise4", language="korean", task="transcribe")
whisper_pipe = pipeline(model="Dearlie/whisper-noise4") 

def transcribe_whisper_custom(audio_data):
    audio_buffer = BytesIO(audio_data)
    data, samplerate = sf.read(audio_buffer, dtype='float32')
    result = whisper_pipe(audio_buffer)
    text = result["text"]
    return text

####STREAMLIT INTERFACE####


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Streamlit Interface
st.title("불사조 음성 인식 STT 프로젝트 모델 비교")
st.subheader("모델을 선택 해주세요:")

model_options = ['Fast-Whisper 모델 테스트', 'Nvidia-Nemo 모델 테스트', 'Wav2Vec-Kor-LJ 모델 테스트', 'Whisper-Noise-D 모델 테스트']
model_selection = st.radio("Choose a model to test:", model_options)

if 'model_selected' in st.session_state:
    if st.session_state['model_selected'] != model_selection:
        st.session_state['model_selected'] = model_selection
        st.session_state.pop('wav_audio_data', None) 
else:
    st.session_state['model_selected'] = model_selection

if 'model_selected' in st.session_state and 'wav_audio_data' not in st.session_state:
    st.write("Please record your voice.")
    wav_audio_data = st_audiorec()
    if wav_audio_data:
        st.session_state['wav_audio_data'] = wav_audio_data

if 'wav_audio_data' in st.session_state and st.session_state['wav_audio_data'] is not None:
    if st.button("Process Audio"):
        if model_selection == model_options[0]:
            model = WhisperModel("medium", device="cuda", compute_type="float16")
            recognized_text, language = transcribe_whisper(st.session_state['wav_audio_data'])
            st.audio(st.session_state['wav_audio_data'], format='audio/wav')
            st.write("Detected language:", language)
            st.write("Output:")
            st.write(recognized_text)
        elif model_selection == model_options[1]:
            asr_model = nemo_asr.models.ASRModel.from_pretrained("eesungkim/stt_kr_conformer_transducer_large")
            recognized_text = transcribe_nemo(st.session_state['wav_audio_data'])
            st.audio(st.session_state['wav_audio_data'], format='audio/wav')
            st.write("Output:")
            st.write(recognized_text)
        elif model_selection == model_options[2]:
            pipe = pipeline(model="Ljrabbit/wav2vec2-large-xls-r-300m-korean-test0")
            composed_text = transcribe(st.session_state['wav_audio_data'])
            st.audio(st.session_state['wav_audio_data'], format='audio/wav')
            st.write("Output:")
            st.write(composed_text)
        elif model_selection == model_options[3]:
            recognized_text = transcribe_whisper_custom(st.session_state['wav_audio_data'])
            st.audio(st.session_state['wav_audio_data'], format='audio/wav')
            st.write("Output:")
            st.write(recognized_text)
