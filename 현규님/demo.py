import os
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from transformers import pipeline
import pysubs2
import nlptutti as metrics

feature_extractor = WhisperFeatureExtractor.from_pretrained("Dearlie/whisper-noise")
tokenizer = WhisperTokenizer.from_pretrained("Dearlie/whisper-noise", language="korean", task="transcribe")
processor = WhisperProcessor.from_pretrained("Dearlie/whisper-noise", language="korean", task="transcribe")
pipe = pipeline(model="Dearlie/whisper-noise", device=0)  # change to "your-username/the-name-you-picked"

def transcribe(audio):
    text = pipe(audio)["text"]
    return text

file_list = sorted(os.listdir('C:/rar/tpnlp2/wav/'))
text_list = sorted(os.listdir('C:/rar/tpnlp2/text/'))    
preds = []
script = []

for file in file_list:
    print(f'Transcrpit:')
    if file.endswith('wav'):
        pred = transcribe(f'C:/rar/tpnlp2/wav/{file}')
    else:
        continue
    preds.append(pred)
    print(pred)

for text in text_list:
    subs = pysubs2.load(f'C:/rar/tpnlp2/text/{text}', encoding='utf-8')
    txt = ''
    for line in subs:
        txt += line.text+' '
    script.append(txt)    

cers = 0
wers = 0
crrs = 0
for i in range(len(preds)):
    refs = script[i]
    predicts = preds[i]

    result_cer_1 = metrics.get_cer(refs, predicts)
    result_cer = result_cer_1['cer']

    result_wer_1 = metrics.get_wer(refs, predicts)
    result_wer = result_wer_1['wer']

    result_crr_1 = metrics.get_crr(refs, predicts)
    result_crr = result_crr_1['crr']

    cers += result_cer
    wers += result_wer
    crrs += result_crr
    print(f'{i}번 => cer : {result_cer}, wer : {result_wer}, crr : {result_crr}')

#전체 instance 로 나누어, 평균 내주기
print('평균 CER : ',cers/len(preds), '\n평균 WER : ', wers/len(preds), '\n평균 CRR : ', crrs/len(preds))