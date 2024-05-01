# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

from huggingface_hub import login
login()

# from transformers import WhisperFeatureExtractor
# from transformers import WhisperTokenizer
# from transformers import WhisperProcessor

# feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
# tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base", language="korean", task="transcribe")
# processor = WhisperProcessor.from_pretrained("openai/whisper-base", language="korean", task="transcribe")

# import pandas as pd
# from tqdm import tqdm

# ### df 만들기전에 실행 / 만든 이후로는 pass

# # 1. 오디오 파일 경로 취합
# import glob

# path = "D:/aihub/136-1.극한 소음 음성 인식 데이터/01-1.정식개방데이터/Training/01.원천데이터/*"

# raw_data_list = glob.glob(path)
# raw_data_list = sorted(raw_data_list)

# import pysubs2

# # 2. 텍스트 파일 경로 취합
# path = "D:/aihub/136-1.극한 소음 음성 인식 데이터/01-1.정식개방데이터/Training/02.라벨링데이터/*"
# labeled_data_list = glob.glob(path)
# # 레이블 데이터에는 json 데이터가 폴더별로 하나씩 있으므로 txt 파일만을 골라낸다.
# labeled_data_list = sorted([file for file in labeled_data_list])

# transcript_list = []
# for labeled_data in labeled_data_list:
#     subs = pysubs2.load(labeled_data, encoding='utf-8') 
#     text = ''
#     for line in subs:
#         text += line.text+' '
#     transcript_list.append(text)
# df = pd.DataFrame(data=transcript_list, columns = ["transcript"])

# # 텍스트 데이터로 만든 데이터프레임에 음성 파일 경로 컬럼을 추가
# df["raw_data"] = raw_data_list

# # 필요에 따라 데이터프레임을 csv 파일로 저장한다.
# df.to_csv("C:/rar/tpnlp2/pat_noise.csv", index=False, encoding='utf-8-sig')

###

####### 데이터셋 전처리 & Hub에 올리기 전엔 실행

# df = pd.read_csv("C:/rar/tpnlp2/path_and_transcript.csv")

# from datasets import Dataset, DatasetDict
# from datasets import Audio

# ds = Dataset.from_dict({"audio": [path for path in df["raw_data"]],
#                        "transcripts": [transcript for transcript in df["transcript"]]}).cast_column("audio", Audio(sampling_rate=16000))

# train_testvalid = ds.train_test_split(test_size=0.1)
# # test_valid = train_testvalid["test"].train_test_split(test_size=0.5)
# datasets = DatasetDict({
#     "train": train_testvalid["train"],
#     "valid": train_testvalid["test"]})
#     # "test": test_valid["test"],
#     # "valid": test_valid["train"]})
    
# def prepare_dataset(batch):
#     audio = batch["audio"]
#     batch["input_length"] = len(batch["audio"])
#     batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
#     batch["labels"] = tokenizer(batch["transcripts"]).input_ids
#     batch["labels"] = batch['labels'][:448]
#     batch["labels_length"] = len(tokenizer(batch["transcripts"], add_special_tokens=False).input_ids)
#     return batch

# MAX_DURATION_IN_SECONDS = 30.0
# max_input_length = MAX_DURATION_IN_SECONDS * 16000

# def filter_inputs(input_length):
#     """Filter inputs with zero input length or longer than 30s"""
#     return 0 < input_length < max_input_length

# def filter_labels(labels_length):
#     """Filter empty label sequences"""
#     return 0 < len(labels_length)

# low_call_voices = datasets.map(prepare_dataset, remove_columns=datasets.column_names["train"], num_proc=None)
# low_call_voices.push_to_hub("Dearlie/noise_train")

######

# Hub로부터 전처리가 완료된 데이터셋을 로드
from datasets import load_dataset
low_call_voices_prepreocessed = load_dataset("Dearlie/noise_train")

import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # 인풋 데이터와 라벨 데이터의 길이가 다르며, 따라서 서로 다른 패딩 방법이 적용되어야 한다. 그러므로 두 데이터를 분리해야 한다.
        # 먼저 오디오 인풋 데이터를 간단히 토치 텐서로 반환하는 작업을 수행한다.
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Tokenize된 레이블 시퀀스를 가져온다.
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # 레이블 시퀀스에 대해 최대 길이만큼 패딩 작업을 실시한다.
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # 패딩 토큰을 -100으로 치환하여 loss 계산 과정에서 무시되도록 한다.
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # 이전 토크나이즈 과정에서 bos 토큰이 추가되었다면 bos 토큰을 잘라낸다.
        # 해당 토큰은 이후 언제든 추가할 수 있다.
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

# 훈련시킬 모델의 processor, tokenizer, feature extractor 로드
from transformers import WhisperTokenizer,  WhisperFeatureExtractor
from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained("openai/whisper-base", language="korean", task="transcribe")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base", language="korean", task="transcribe")
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")

# 데이터 콜레이터 초기화
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

import evaluate

metric = evaluate.load('cer')

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # pad_token을 -100으로 치환
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # metrics 계산 시 special token들을 빼고 계산하도록 설정
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    cer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}

from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

model.generation_config.language = "korean"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None

from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="whisper-noise",  # 원하는 리포지토리 이름을 임력한다.
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # 배치 크기가 2배 감소할 때마다 2배씩 증가
    learning_rate=1e-5,
    warmup_steps=500,
    # warmup_steps=125,
    max_steps=4000,  # epoch 대신 설정
    # max_steps=1000,  # epoch 대신 설정
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    # save_steps=250,
    # eval_steps=250,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="cer",  # 한국어의 경우 'wer'보다는 'cer'이 더 적합할 것
    greater_is_better=False,
    push_to_hub=True,
)

from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=low_call_voices_prepreocessed["train"],
    eval_dataset=low_call_voices_prepreocessed["valid"],  # or "test"
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

processor.save_pretrained(training_args.output_dir)

trainer.train()

kwargs = {
    "dataset_tags": "AIHub/noise",
    "dataset": "Noise Data",  # a 'pretty' name for the training dataset
    "dataset_args": "config: ko, split: test",
    "language": "ko",
    "model_name": "Whisper Base Noise Ko - Dearlie",  # a 'pretty' name for our model
    "finetuned_from": "openai/whisper-base",
    "tasks": "automatic-speech-recognition",
}

trainer.push_to_hub(**kwargs)