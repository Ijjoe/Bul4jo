#코드 테스트는 안됨 검수 진행중

from torchtext.datasets import IMDB
import random
from sklearn.model_selection import train_test_split
random.seed(6)



train_iter = IMDB(split='train')
test_iter = IMDB(split='test')
print(type(train_iter)) #class 'torch.utils.data.datapipes.iter.sharding.ShardingFilterIterDataPipe'>


train_lists = list(train_iter)  # ShardingFilterIterDataPipe -> list 변환 가능
test_lists = list(test_iter)

train_lists_small = random.sample(train_lists,1000)  # 1000개의 랜덤 뽑아서
test_lists_small = random.sample(test_lists,1000)




#print(train_lists_small[0])
#print(test_lists_small[0])

train_texts = []
train_labels=[]

for label,text in train_lists_small:
    train_labels.append(1 if label ==2 else 0)
    train_texts.append(text)
    
    
test_texts =[]
test_labels=[]

for label, text in test_lists_small:
    test_labels.append(1 if label==2 else 0)
    test_texts.append(text)
    
# print(train_texts[0])   
# print(train_labels[0])
# print(test_texts[0])
# print(test_labels[0])


train_texts, val_texts, train_labels,val_labels = train_test_split(train_texts
                                                                   ,train_labels
                                                                   ,test_size=0.2
                                                                   ,random_state=3)

#print(len(train_texts))
#print(len(train_labels))
#print(len(val_texts))
#print(len(val_labels))

#pip install scikit-learn
#pip install -U "huggingface_hub[cli]"
#huggingface-cli login



import pandas as pd
import os

#from transformers import DistilBertTokenizerFast 
#tokenizer = DistilBertTokenizerFast.from_pretrained('distil-base-uncased',token='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') # 허깅페이스 토큰 필요 https://huggingface.co/docs/huggingface_hub/main/ko/guides/cli 


from transformers import DistilBertTokenizerFast 

#mod = AutoModel.from_pretrained("distilbert-base-uncased", use_auth_token="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")
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased') # 허깅페이스 토큰 필요 https://huggingface.co/docs/huggingface_hub/main/ko/guides/cli 

# #토크나이저 실행
train_encodings = tokenizer(train_texts,truncation=True,padding=True)
val_encodings = tokenizer(val_texts,truncation=True,padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

#print(train_encodings["input_ids"][0][:5])

#print(tokenizer.decode(train_encodings["input_ids"][0][:5]))

import torch

data =[[1,2],[3,4]]
x_data = torch.tensor(data)
x_data
##---------------------------------------------------------------------------------
list = ['Apple',2,'한글']
len(list)


##---------------------------------------------------------------------------------

import torch

class IMDbDataset(torch.utils.data.Dataset):
    '''
    생성자 __init__()
    자신을 가리키는 매개변수 self 포함
    변수를 저장하기 위해 self.변수명을 사용
    '''
    def __init__(self,encodings,labels):
        self.encodings = encodings
        self.labels = labels 
        
    def __getitem__(self,idx):
        #self.eencoding에 담긴 키와 키값을 items()로 추출
        #이 값을 key와 val 변수에 담아 새로운 키와
        #키값을 갖는 딕셔너리 생성
        
        item={key:torch.tensor(val[idx]) for key,val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)
    


train_dataset = IMDbDataset(train_encodings,train_labels)    
val_dataset= IMDbDataset(val_encodings,val_labels)
test_dataset=IMDbDataset(test_encodings,test_labels)


##---------------------------------------------------------------------------------

for i in train_dataset:
    i
    
    break

##---------------------------------------------------------------------------------

from transformers import DistilBertForSequenceClassification

model=DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

#model

##---------------------------------------------------------------------------------
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir='./results'
    ,num_train_epochs=8
    ,per_device_train_batch_size=16
    ,per_device_eval_batch_size=64
    ,warmup_steps=500
    ,weight_decay=0.01
    ,logging_dir='./logs'
    ,logging_steps=10
)

##---------------------------------------------------------------------------------
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
#model
#!nvidia-smi
##---------------------------------------------------------------------------------

input_tokens = tokenizer(["I feel fantastic","My life is going something wrong","I have not figured out what the chosen title has to do with the movie."]
                         ,truncation=True, padding=True)

outputs = model(torch.tensor(input_tokens['input_ids']))

label_dict = {0:'positive',1:'negative'}

print([label_dict[i] for i in torch.argmax(outputs['logits'], axis=1).cpu().numpy()])

##---------------------------------------------------------------------------------

from transformers import Trainer

trainer = Trainer(
    model=model
    ,args=training_args
    ,train_dataset=train_dataset
    ,eval_dataset=val_dataset
)

trainer.train()

##---------------------------------------------------------------------------------
def test_inference(model,tokenizer):
    input_tokens = tokenizer(["I feel fantastic","My life is going something wrong","I have not figured out what the chosen title has to do with the movie."]
                             ,truncation=True  #?
                             ,padding=True 
                             )
    
    outputs = model(torch.tensor(input_tokens['input_ids']).to(device))   # ? **_ids는 어디서
    label_dict={0,'Positive',1,'Negative'}
    return [label_dict[i] for i in torch.argmax(outputs['logits'],axis=1).cpu().numpy()]  #?
##---------------------------------------------------------------------------------
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification,AdamW
from transformers import DistilBertTokenizer

#1) 사전학습 모델과 토크나이저 불러오기 모델실행결과에 .to(device) 코드로 GPU전달
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
model.to(device)

#파인튜닝 이전 모델을 사용하여 test_inference 함수 실행
print(test_inference(model,tokenizer))

#2)DataLoader 인스턴스 화
train_loader = DataLoader(train_dataset,batch_size=16,shuffle=True)

#3)최적화 함수 정의
optim =AdamW(model.parameters(),lr=5e-5)

#4)모델을 학습 모드로 전환 이는 드롭아웃 및 배치 정규화에 영향을 미침
model.train()

losses =[]

#5)에포크 횟수 만큼 루프 반복
for epoch in range(8):
    print(f'epoch:{epoch}')
    for batch in train_loader:
        #6) 최적화 함수의 기울기(그래디언트) 초기화
        optim.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        #7)모델을 사용한 추론
        outputs = model(input_ids,attention_mask=attention_mask,labels=labels)
        
        #8)손실 계산
        loss=outputs[0]
        losses.append(loss)
        
        #9)오차역전파
        loss.backward()
        
        #10)가중치 업데이트
        optim.step()
        
model.eval()

#eval 모드를 사용하여 test_inference 함수 실행
print(test_inference(model,tokenizer))        


##---------------------------------------------------------------------------------
print(losses)
type(losses)

##---------------------------------------------------------------------------------
new_losses = [ip.item for i in losses]
##---------------------------------------------------------------------------------
new_losses[:5]

##---------------------------------------------------------------------------------
new_losses[-5:]
##---------------------------------------------------------------------------------
import matplotlib.pyplot as plt
plt.plot(new_losses)
plt.show()
##---------------------------------------------------------------------------------
#3장 모델성능평가

model.eval()
l=[]
for test_text in test_texts:
    #토크나이즈 통한 인코딩
    input_tokens = tokenizer([test_text],truncation=True ,padding=True)
    
    outputs = model(torch.tensor(input_tokens['input_ids']).to(device))
    #아웃풋츠에 담긴 로짓값을 기준으로 행단위로,
    #즉 입력 문장 단위로 가장 큰 로짓값의 인덱스 출력 후
    #아이템()을 사용하여 결과물 텐서의 값을 추출하고 cpu 전송
    #이렇게 처리되 ㄴ값을 입력 문장별로 컨테이너 리스트 l에 하나씩저장
    l.append(torch.argmax(outputs['logits'], axis=1).item())
    
corrent_cnt =0 

#zip l과 test_labels를 쌍으로 묶은 후
#각기 pred,ans로 변수 추출
#zip()안에 데이터가 쌍으로 소진될때까 반복
for pred,ans in zip(l,test_labels):
    if pred == ans:
        corrent_cnt +=1
        

#정확도(accuracy) 계산        
# 레이블 개수의 총합, 입력문장 전체건수
print(corrent_cnt/len(test_labels))
    
##---------------------------------------------------------------------------------
tp,tn,fp,fn =0

for pred,ans in zip(l,test_labels):
    if pred == ans:
        if pred == 1:
            tp +=1
        else:
            tn +=1
            
    elif pred ==0:
        fn +=1
    elif pred ==1:
        fp +=1                                
##---------------------------------------------------------------------------------
#1 재현율
recall =tp/(tp+fn)
print(recall)
##---------------------------------------------------------------------------------
#2 정밀도
precision =tp/(tp+fp)
print(precision)
##---------------------------------------------------------------------------------
#3 F1값
(2* precision * recall) / (precision + recall)
##---------------------------------------------------------------------------------
from sklearn.metrics import classification_report

print(classification_report(test_labels,l))
##---------------------------------------------------------------------------------
#4장 GPT를 활용한 작문
#!pip install transformers sentencepiece
from transformers import GPTNeoForCausalLM,GPT2Tokenizer

model = GPTNeoForCausalLM.from_pretrained("EleutherAI/get-neo-1.3B")
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/get-neo-1.3B")

##---------------------------------------------------------------------------------
input = tokenizer.encode("I evaluated the performance of GPT-Neo developed by OpenAI."
                         ,return_tensors="pt")

print(input[0])

#
print(tokenizer.decode(input[0]))

##---------------------------------------------------------------------------------
tokenizer.add_special_tokens({'pad_token' : '[PAD]'})

input = tokenizer.batch_encode_plus(["I evaluated the performance of GPTNeo developed by OpenAI.","I evaluated the performance of GPT developed by OpenAI."],padding=True,truncation=True,return_tensors="pt")
## 
print(input['input_ids'])

print([tokenizer.decode(input['input_ids'][i]) for i in range(len(input['input_ids']))])
##---------------------------------------------------------------------------------
#토크나이징
input = tokenizer.batch_encode_plus(["I evaluated the performance of GPTNeo developed by OpenAI.","Vaccine for new coronavirus in the U"],max_length=5,padding=True,truncation=True,return_tensors="pt")

input('inputs_ids')

##---------------------------------------------------------------------------------
#인코딩 결과 투입
generated = model.generate(input['input_ids'])

len(generated)
##---------------------------------------------------------------------------------
#디코딩
generated_text = tokenizer.batch_decode(generated)

for i , sentence in enumerate(generated_text):
    print(f'No.{i+1}')
    print(f'{sentence}\n')

##---------------------------------------------------------------------------------
from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

model = AutoModelWithLMHead("distilgpt2")

##---------------------------------------------------------------------------------
input_ids = tokenizer.encode("I like gpt because it's",return_tensors='pt')

greedy_output = model.generate(input_ids,max_length=12)

print("Output : \n" + 100 * '-')
print(tokenizer.decode(greedy_output[0],skip_special_tokens=True))

##---------------------------------------------------------------------------------
from transformers import AutoTokenizer,AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-small")

##---------------------------------------------------------------------------------
input_ids = tokenizer.encode("I like gpt because it's",return_tensors='pt')

greedy_output = model.generate(input_ids,max_length=30)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))

##---------------------------------------------------------------------------------
input_ids= tokenizer.encode("Covid19 delta is spreading", return_tensors=True)

greedy_output= model.generate(input_ids,max_length=50)


print("Output:\n" + 100 * '-')
print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))


##---------------------------------------------------------------------------------
#5장 MLM
from transformers import pipeline

unmasker=pipeline('fill-mask',model='bert-base-uncased')
unmasker("MLM and NSP is the [MASK] task of BERT.")
##---------------------------------------------------------------------------------
from transformers import pipeline

#모델 변경으로 어떤 결과가~
unmasker=pipeline('fill-mask',model='distilbert-base-uncased')
unmasker("MLM and NSP is the [MASK] task of BERT.")

##---------------------------------------------------------------------------------
from transformers import pipeline
unmasker=pipeline('fill-mask',model='albert-base-v2')
unmasker("MLM and NSP is the [MASK] task of BERT.")

##---------------------------------------------------------------------------------
#6장 CLIP - 이미지 인식과 자연어 처리의 연동
#!pip install transformers
#!pip install ftfy
#!pip install transformers==4.6.1  원본책
#!pip install ftfy==6.0.3          원본책

from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url,stream=True).raw)
image

##---------------------------------------------------------------------------------
from transformers import CLIPProcessor, CLIPModel

model = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

model

##---------------------------------------------------------------------------------
candidates =  ["three cats lying on the couch", "a photo of a cat", "a photo of a dog", "a lion", "two cats lying on the cusion"]

inputs = processor(text=candidates,images=image,return_tensors='pt', padding=True)

inputs

##---------------------------------------------------------------------------------
import numpy as np
a=np.array([1,2,3])

print(type(a))
print(a)
##---------------------------------------------------------------------------------
import matplotlib.pyplot as plt
plt.imshow(inputs['pixel_values'][0][0]);

inputs['pixel_values'].shape
##---------------------------------------------------------------------------------
inputs['input_ids'][0]
##---------------------------------------------------------------------------------
processor.tokenizer.decode(inputs['input_ids'][0])
##---------------------------------------------------------------------------------
model.eval()

outputs = model(**inputs)
outputs.keys()

##---------------------------------------------------------------------------------
logits_per_image = outputs.logits_per_image
print(logits_per_image)
##---------------------------------------------------------------------------------
probs = logits_per_image.softmax(dim=1)
import torch 

print(candidates[torch.argmax(probs).item()])

##---------------------------------------------------------------------------------
#7장 Wav2Vec2 자동 음성 인식
#!pip install datasets==1.4.1
#!pip install transformers
#!pip install soundfile
#!pip install jiwer

import soundfile as sf
import torch 
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC,Wav2Vec2Processor

model_name = "elgeish/wav2vec2-base-timit-ar"

processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

model.evel()

##---------------------------------------------------------------------------------
dataset = load_dataset("timit_asr")
##---------------------------------------------------------------------------------
import soundfile as sf
import IPython.display as ipd
import numpy as np
import random

speech,_=sf.read(dataset['test']["file"][0])
ipd.Audio(data=np.asarry(speech), autoplay=True,rate=16000)

##---------------------------------------------------------------------------------
inputs = processor(speech,sampling_rate=16000, return_tensors='pt' , padding='longest')

inputs['input_values'].shape
##---------------------------------------------------------------------------------
with torch.no_grad():
    outputs = model(inputs.input_values)
    print(outputs.logits.shape)
    predicted_ids = outputs.logits.argmax(-1)
    print(predicted_ids)
    print(f'predicted:{processor.tokenizer.batch_decode(predicted_ids)[0]}')
    print(f'answer:{dataset['test']['text'][0]}')
##---------------------------------------------------------------------------------
#8장 BERT 다중 클래스 분류
#!pip install transformers

from transformers import AutoTokenizer,AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased')

## ?? model.evel()

##---------------------------------------------------------------------------------
import torch 

dic ={0:'positive',1:'neutral',2:'negative'}

eval_list = ["I like apple", "I like pear", "I go to school", "I dislike mosquito", "I felt very sad", "I feel so good"]

ans=torch.tensor([0,0,1,2,2,0])

##---------------------------------------------------------------------------------
model.evel()

with torch.no_grad():
    for article in eval_list:
        inputs = tokenizer.encode(article,return_tensors='pt', padding=True , truncation=True)
        outputs = model(inputs)
        
        logits = outputs.logits
        print(f'{dic[logits.argmax(-1).item()]}:{article}')

##---------------------------------------------------------------------------------
from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=1e-5)
model.train()

epochs= 50
losses=[]

for epoch in epochs:
    optimizer.zero_grad()
    inputs = tokenizer.batch_decode(eval_list,return_tensors='pt',padding=True,truncation=True)
    outputs = model(**inputs,labels=ans)
    logits = outputs.logits
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    losses.append(loss)
    print(f'epoch:{epoch+1}, loss:{loss}')

##---------------------------------------------------------------------------------
new_losses = [i.item() for i in losses]

##---------------------------------------------------------------------------------
import matplotlib.pyplot as plt
plt.plot(new_losses)
##---------------------------------------------------------------------------------
import torch
preds = torch.tensor(preds)

##---------------------------------------------------------------------------------
preds
##---------------------------------------------------------------------------------
print(f'Accuracy:{100 * sum(ans.detach().clone()==preds)/len(ans.detach().clone())}%')
##---------------------------------------------------------------------------------
#9장 BART 자동요약
from transformers import BartTokenizer,TFBartForConditionalGeneration

model = TFBartForConditionalGeneration.from_pretrained('facebook/bart-large')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

##---------------------------------------------------------------------------------
import re
 
article = '''
 Google LLC is an American multinational technology company
 –Abbreviated for copyright reasons
monopoly position.
 '''
 
print("before:")
print(article)
article = re.sub(r"[:.]\[[0-9]+\](.*?)\([0-9]+\)|.?[([][0-9]+[])]|\n|\r", r"", article)
print("after:")
print(article)
##---------------------------------------------------------------------------------

inputs= tokenizer([article],max_length=1024, return_tensors='tf', truncation=True)
inputs
##---------------------------------------------------------------------------------
inputs['input_ids'].numpy()
##---------------------------------------------------------------------------------
summary_ids = model.generate(inputs['input_ids'],num_beams=5,max_length=25)
summary_ids
##---------------------------------------------------------------------------------
print(''.join([tokenizer.decode(g, skip_special_tokens=True,
                                clean_up_tokenization_spaces=False) for g in summary_ids]))

##---------------------------------------------------------------------------------
#10장 BART 앙상블 학습

dataset = [["What music do you like?", "I like Rock music.", 1],
         ["What is your favorite food?", "I like sushi the best", 1],
         ["What is your favorite color?", "I’m going to be a doctor", 0], 
         ["What is your favorite song?", "Tokyo olympic game in 2020 was postponed", 0],
         ["Do you like watching TV shows?", "Yeah, I often watch it in my spare time", 1]]

##---------------------------------------------------------------------------------
from transformers import BertPreTrainedModel,BertConfig,BertModel,BertTokenizer,AdamW
from torch import nn

class BertEnsembleForNextSentencePrediction(BertPreTrainedModel):
    def __init__(self,config,*args,**kwargs):
        super().__init__(config)
        self.bert_model_1  =BertModel(config)
        self.bert_model_2 = BertModel(config)
        self.cls = nn.Linear(2 * self.config.hidden_size,2)
        self.init_weights()
        
        
    def forward(
        self
        ,input_ids=None
        ,attention_mask=None
        ,token_type_ids=None
        ,position_ids=None
        ,head_mask=None
        ,inputs_embeds=None
        ,next_sentence_label=None):
        
        outputs=[]
        input_ids_1 = input_ids[0]
        attention_mask_1=attention_mask[0]
        outputs.append(self.bert_model_1(input_ids_1,attention_mask=attention_mask_1))
        
        input_ids_2 = input_ids[1]
        attention_mask_2=attention_mask[1]
        outputs.append(self.bert_model_2(input_ids_2,attention_mask=attention_mask_2))
        
        last_hidden_states = torch.cat([output[1] for output in outputs],dim=1)
        logits = self.cls(last_hidden_states)
        
        if next_sentence_label is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            next_sentence_loss = loss_fct(logits.view(-1,2), next_sentence_label.view(-1))
            return next_sentence_loss, logits
        else:
            return logits
    
##---------------------------------------------------------------------------------
import torch

x = torch.randint(0,10(2,3))
y = torch.randn(2,3,4)
x
##---------------------------------------------------------------------------------
x = x.view(3,2)
x = x.view(6,-1) 
y = y.view(-1) #1차원 평탄화 가능
y.size()

##---------------------------------------------------------------------------------
import torch 
from transformers import AdamW

device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')

config = BertConfig()
model = BertEnsembleForNextSentencePrediction(config)
model.to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

learning_rate = 1e-5
no_decay = ['bias','LayerNorm.weight']

optimizer_grouped_parameters=[{
    'params' : [p for nmp in model.named_parameters() if not any(nd in n for nd in nd_decay)],}]

optimizer=AdamW(optimizer_grouped_parameters, lr=learning_rate)
##---------------------------------------------------------------------------------
def prepare_data(dataset,qa=Trun):
    input_ids,attention_masks = [],[]
    labels =[]
    
    for point in dataset:
        if qa is True:
            q, a, _ = point
        else:
            a, q, _ = point
            
    encoded_dict = tokenizer.encode_plus(
        q
        ,a
        ,add_special_tokens=True
        ,max_length=123
        ,pad_to_max_length=True
        ,return_attention_mask=True
        ,return_tensors='pt'
        ,truncation=True
    )    
    
    input_ids.append(encoded_dict['input_ids'])                    
    attention_masks.append(encoded_dict['attention_mask'])                    
    labels.append(point[-1])                    
    
    input_ids = torch.cat(input_ids,dim=0)
    attention_masks = torch.cat(attention_mask,dim=0)
    
    return input_ids,attention_masks,labels
            

##---------------------------------------------------------------------------------
from torch.utils.data import DataLoader,RandomSampler,Dataset,SequentialSampler

class QADataset(Dataset):
    def __init__(self,input_ids,attention_masks,labels=None):
        self.input_ids = np.array(input_ids)
        self.attention_masks = np.array(attention_masks)
        self.labels = torch.tensor(labels,dtype=torch.long)
        
    def __getitem__(self, index):
        return self.input_ids[index],self.attention_masks[index],self.labels[index]
    
    
    def __len__(self):
        return self.input_ids.shape[0]
        
##---------------------------------------------------------------------------------
input_ids_qa, attention_masks_qa,labels_qa = prepare_data(dataset)

train_dataset_qa = QADataset(input_ids_qa,attention_masks_qa,labels_qa)

input_ids_aq, attention_masks_aq,labels_aq = prepare_data(dataset,qa=False)

train_dataset_aq = QADataset(input_ids_aq,attention_masks_aq,labels_aq)

dataloader_qa = DataLoader(dataset=train_dataset_qa,batch_size=5,sampler=SequentialSampler(train_dataset_qa))

dataloader_aq = DataLoader(dataset=train_dataset_qa,batch_size=5,sampler=SequentialSampler(train_dataset_aq))

##---------------------------------------------------------------------------------
epochs =30

for epoch in range(epochs):
    for step,combined_batch in enumerate(zip(dataloader_qa,dataloader_aq)):
        batch_1, batch_2 = combined_batch
        model.train()
        
        batch_1 = tuple(t.to(device) for t in batch_1)
        batch_2 = tuple(t.to(device) for t in batch_2)
        
        inputs={
            'input_ids':[batch_1[0],batch_2[0]]
            ,'attention_mask':[batch_1[0],batch_2[0]]
            ,'next_sentence_label': batch_1[2]
        }
        
        outputs =model(**inputs)
        loss = outputs[0]
        loss.backward()
        print(f'epoch:{epoch+1}, loss:{loss}')
        
        optimizer.step()
        model.zero_grad()

##---------------------------------------------------------------------------------

input_ids_qa, attention_masks_qa,labels_qa = prepare_data(dataset)

train_dataset_qa = QADataset(input_ids_qa,attention_masks_qa,labels_qa)

input_ids_aq, attention_masks_aq,labels_aq = prepare_data(dataset,qa=False)

train_dataset_aq = QADataset(input_ids_aq,attention_masks_aq,labels_aq)

dataloader_qa = DataLoader(dataset=train_dataset_qa,batch_size=5,sampler=SequentialSampler(train_dataset_qa))

dataloader_aq = DataLoader(dataset=train_dataset_qa,batch_size=5,sampler=SequentialSampler(train_dataset_aq))

complete_outputs, complete_label_ids =[],[]

for step,combined_batch in enumerate(zip(dataloader_qa,dataloader_aq)):
    model.eval()
        
    batch_1, batch_2 = combined_batch
    batch_1 = tuple(t.to(device) for t in batch_1)
    batch_2 = tuple(t.to(device) for t in batch_2)
        
    with torch.no_grad():
        inputs={
            'input_ids':[batch_1[0],batch_2[0]]
            ,'attention_mask':[batch_1[0],batch_2[0]]
            ,'next_sentence_label': batch_1[2]
        }
        
        outputs =model(**inputs)
        
        tmp_eval_loss,logits = outputs[:2]
        logits = logits.detach().cpu().numpy()
        output = np.argmax(logits,axis=1)
        label_ids = inputs['next_sentence_label'].detach().cpu().numpy()
        
        loss.backward()
        print(f'epoch:{epoch+1}, loss:{loss}')
        
    complete_outputs.extend(outputs)
    complete_label_ids.extend(label_ids)
    
print(complete_outputs, complete_label_ids)    
    
##---------------------------------------------------------------------------------

dataset = [["What music do you like?", "I like Rock music.", 1],
         ["What is your favorite food?", "I like sushi the best", 1],
         ["What is your favorite color?", "I’m going to be a doctor", 0], 
         ["What is your favorite song?", "Tokyo olympic game in 2020 was postponed", 0],
         ["Do you like watching TV shows?", "Yeah, I often watch it in my spare time", 1]]


input_ids_qa, attention_masks_qa,labels_qa = prepare_data(dataset)

train_dataset_qa = QADataset(input_ids_qa,attention_masks_qa,labels_qa)

input_ids_aq, attention_masks_aq,labels_aq = prepare_data(dataset,qa=False)

train_dataset_aq = QADataset(input_ids_aq,attention_masks_aq,labels_aq)

dataloader_qa = DataLoader(dataset=train_dataset_qa,batch_size=16,sampler=SequentialSampler(train_dataset_qa))

dataloader_aq = DataLoader(dataset=train_dataset_qa,batch_size=16,sampler=SequentialSampler(train_dataset_aq))

complete_outputs, complete_label_ids =[],[]

for step,combined_batch in enumerate(zip(dataloader_qa,dataloader_aq)):
    model.eval()
        
    batch_1, batch_2 = combined_batch
    batch_1 = tuple(t.to(device) for t in batch_1)
    batch_2 = tuple(t.to(device) for t in batch_2)
        
    with torch.no_grad():
        inputs={
            'input_ids':[batch_1[0],batch_2[0]]
            ,'attention_mask':[batch_1[0],batch_2[0]]
            ,'next_sentence_label': batch_1[2]
        }
        
        outputs =model(**inputs)
        
        tmp_eval_loss,logits = outputs[:2]
        logits = logits.detach().cpu().numpy()
        output = np.argmax(logits,axis=1)
        label_ids = inputs['next_sentence_label'].detach().cpu().numpy()
        
        loss.backward()
        print(f'epoch:{epoch+1}, loss:{loss}')
        
    complete_outputs.extend(outputs)
    complete_label_ids.extend(label_ids)
    
print(complete_outputs, complete_label_ids)    
##---------------------------------------------------------------------------------
#11장 BigBird
#!pip install sentencepiece

from transformers import BigBirdTokenizer,BigBirdForMaskedLM
import torch 

tokenizer = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-base')
model = BigBirdForMaskedLM.from_pretrained('google/bigbird-roberta-base')
model

##---------------------------------------------------------------------------------
 
inputs = ["I like reading [MASK].", "I like driving a [MASK].","The world is facing with a [MASK] [MASK] crisis. We are all suffering fro"]
answers = ["I like reading book.", "I like driving a car.", "The world is facing with a pandemic crisis. We are all suffering from infectious dis"]

encoded_inputs,encoded_labels=[],[]

for i ,l in zip(inputs,answers):
    encoded_inputs.append(tokenizer(i,return_tonsors='pt'))
    encoded_labels.append(tokenizer(l,return_tonsors='pt')['input_ids'])

##---------------------------------------------------------------------------------
for input,label in zip(encoded_inputs,encoded_labels):
    outputs = model(**input,labels=label)
    loss = outputs.loss
    logits = outputs.logits
    print(f'loss : {loss.item()}')
    print(f'prediction: {''.join([tokenizer.decode(logits[0][i].argmax(-1)) for i in range(1,len(logits[0]))])}')
    print(f'answer : {tokenizer.decode(label[0][1:-1])}')
    print('\n')
    
##---------------------------------------------------------------------------------
#12장 PEGASUS
#!pip install 

from transformers import PegasusTokenizer,PegasusForConditionalGeneration
import torch

model_name = 'google/pegasus-xsum'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)


##---------------------------------------------------------------------------------
inputs = [ '''          
          Pretraining large neural language models, such as BERT, has led to impressive gains on many natural language processing (NLP) tasks. However, most pretraining efforts focus on general domain corpora, such as newswire and Web. A prevailing assumption is that even domain-specific pretraining can benefit by starting from general-domain language models. Recent work shows that for domains with abundant unlabeled text, such as biomedicine, pretraining language models from scratch results in substantial gains over continual pretraining of general-domain language models.
          ''']

batch = tokenizer(inputs,truncation=True, padding='longest',return_tonsers='pt').to(device)

##---------------------------------------------------------------------------------
translated = model.generate(**batch)

generated_text = tokenizer.batch_decode(translated,skip_special_tokens=True)

print(generated_text[0])
##---------------------------------------------------------------------------------
#13장 M2M100 자동번역
#!pip install sentencepiece
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

model_name = 'facebook/m2m100_418M'
tokenizer=M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)

model
##---------------------------------------------------------------------------------
korea_text = "인생은 초콜릿 상자와 같다"

tokenizer.src_lang='kr'

encoded_kr = tokenizer(korea_text,return_tensors='pt')

generated_tokens = model.generate(**encoded_kr,forced_bos_token_id=tokenizer.get_lang_id("en"))
tokenizer.batch_decode(generated_tokens,skip_special_tokens=True)
##---------------------------------------------------------------------------------
#14장 Mobile BERT
from transformers import MobileBertTokenizer,MobileBertModel
import torch 

model_name = 'google/mobilebert-uncased'
tokenizer= MobileBertTokenizer.from_pretrained(model_name) 
model_mbert=MobileBertModel.from_pretrained(model_name) 

model_mbert
##---------------------------------------------------------------------------------
from transformers import BertTokenizer,BertModel
import torch 

model_name = 'bert-base-uncased'
tokenizer=BertTokenizer.from_pretrained(model_name) 
model_bert=BertModel.from_pretrained(model_name) 

model_bert
##---------------------------------------------------------------------------------
text = 'Mobile bert is more practical than bert.'

inputs = tokenizer_mbert.tokenize(text)
print(inputs)

inputs = tokenizer_bert.tokenize(text)
print(inputs)

##---------------------------------------------------------------------------------
text = 'Mobile bert is more practical than bert.'

inputs = tokenizer_mbert.encode(text)
outputs = model_mbert(torch.tensor(inputs).unsqueeze(0))
print(outputs.last_hidden_state.shape)

inputs = tokenizer_bert.encode(text)
outputs = model_bert(torch.tensor(inputs).unsqueeze(0))
print(outputs.last_hidden_state.shape)

##---------------------------------------------------------------------------------
import torch

x = torch.randint(0,10(3,1))
print(x , x.size())
print(x.squeeze(), x.squeeze().size())
print(x.squeeze().unsqueeze(0), x.squeeze().unsqueeze(0).size())

##---------------------------------------------------------------------------------
from transformers import MobileBertTokenizer,MobileBertForMaskedLM
import torch 

model_name = 'google/mobilebert-uncased'
tokenizer= MobileBertTokenizer.from_pretrained(model_name) 
model=MobileBertForMaskedLM.from_pretrained(model_name) 

inputs = tokenizer("The capital of Korea is [MASK].", return_tensors='pt')
labels = tokenizer("The capital of Korea is Seoul.", return_tensors='pt')['input_ids']

outputs = model(**inputs,labels=labels)

loss = outputs.loss
logits = outputs.logits

print(''.join([tokenizer.decode(i.item()).replace(" ","") for i in logits.argmax(-1)[0]][1:-1]))
##---------------------------------------------------------------------------------
from transformers import BertTokenizer,BertForMaskedLM
import torch 


model_name = 'google/mobilebert-uncased'
tokenizer= BertTokenizer.from_pretrained(model_name) 
model=BertForMaskedLM.from_pretrained(model_name) 

inputs = tokenizer("The capital of Korea is [MASK].", return_tensors='pt')
labels = tokenizer("The capital of Korea is Seoul.", return_tensors='pt')['input_ids']

outputs = model(**inputs,labels=labels)

loss = outputs.loss
logits = outputs.logits

print(''.join([tokenizer.decode(i.item()).replace(" ","") for i in logits.argmax(-1)[0]][1:-1]))

##---------------------------------------------------------------------------------

#15장 GPT,DialoGPT, DistilGPT2
from transformers import AutoTokenizer,AutoModelWithLMHead 

model_name = 'distilgpt2'
tokenizer=AutoTokenizer.from_pretrained(model_name)
model=AutoModelWithLMHead.from_pretrained(model_name)

##---------------------------------------------------------------------------------
input_ids = tokenizer.encode("I like gpt because it's",return_tensors='pt')

greedy_output = model.generate(input_ids,max_length=12)


print("Output : \n" + 100 * '-')
print(tokenizer.decode(greedy_output[0],skip_special_tokens=True))


##---------------------------------------------------------------------------------
from transformers import AutoTokenizer,AutoModelWithLMHead 

model_name = 'microsoft/DialoGPT-small'
tokenizer=AutoTokenizer.from_pretrained(model_name)
model=AutoModelWithLMHead.from_pretrained(model_name)
##---------------------------------------------------------------------------------
input_ids = tokenizer.encode("I like gpt because it's",return_tensors='pt')

greedy_output = model.generate(input_ids,max_length=12)


print("Output : \n" + 100 * '-')
print(tokenizer.decode(greedy_output[0],skip_special_tokens=True))


##---------------------------------------------------------------------------------

##---------------------------------------------------------------------------------
#16장 자연어 처리 실습 BERT 및 tSNE
#!pip install wikipedia

import wikipedia

keyword = 'Moderna'
wikipedia.set_lang('en')
search_response =wikipedia.search(keyword)
print(search_response)


moderna_text_t = wikipedia.search(search_response[2])
type(moderna_text_t)

moderna_text= wikipedia.search(search_response[2]).content
moderna_text
##---------------------------------------------------------------------------------
keyword= 'PFizer'

wikipedia.set_lang('en')
search_response =wikipedia.search(keyword)
print(search_response)

pfizer_text = wikipedia.page(search_response[1]).content

##---------------------------------------------------------------------------------
from transformers import pipeline,AutoTokenizer
get_feature= pipeline('feature-extraction',model='bert-base-uncased',tokenizer='bert-base-uncased')

##---------------------------------------------------------------------------------
sample_word = 'vaccine'
hidden_state = get_feature(sample_word)

import numpy as np
np.array(hidden_state).shape
##---------------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
print(tokenizer.decode(tokenizer(sample_word)['input_ids']))
##---------------------------------------------------------------------------------
def get_cls_vector(sample_text):
    hidden_state = get_feature(sample_text,padding=True,truncation=True,max_length=512)
    cls_vec = np.array(hidden_state)[0,0]
    return cls_vec

ml = moderna_text.split('\n')
##---------------------------------------------------------------------------------
moderna_vecs = np.array([get_cls_vector(text) for text in ml])
pfizer_vecs = np.array([get_cls_vector(text) for text in pfzl])

print(moderna_vecs)
print(pfizer_vecs)
##---------------------------------------------------------------------------------
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2,random_state=0)

moderna_vecs_reduced = tsne.fit_transform(moderna_vecs)

##---------------------------------------------------------------------------------
import matplotlib.pyplot as plt
plt.figure(figsize=(13.7))
plt.scatter(moderna_vecs_reduced[:0],moderna_vecs_reduced[:,1] c=[i for i in range(len(ml))])
plt.colorbar()
plt.show();
##---------------------------------------------------------------------------------
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2,random_state=0)

pfizer_vecs_reduced = tsne.fit_transform(pfizer_vecs)

import matplotlib.pyplot as plt
plt.figure(figsize=(13.7))
plt.scatter(pfizer_vecs_reduced[:0],pfizer_vecs_reduced[:,1] c=[i for i in range(len(pfzl))])
plt.colorbar()
plt.show();
##---------------------------------------------------------------------------------



