import pandas as pd

import pickle
import numpy as np

from konlpy.tag import Okt
okt = Okt()

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# data load
data = pd.read_table('questions_MA_GE_1125.txt')

# split train data/test data
l = 5
for_test = np.array([(i % l) == (l - 1) for i in range(data.shape[0])])
for_train = ~for_test

train_data = data[for_train].copy()
test_data = data[for_test].copy()

# preprocessing
# train_data에 중복 샘플 제거
train_data.drop_duplicates(subset=['contents'], inplace=True)

# Null 값을 가진 샘플 제거
train_data = train_data.dropna(how='any')

# 한글과 공백을 제외하고 모두 제거 (정규 표현식)
train_data['contents'] = train_data['contents'].\
    str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z<> ]", "")

# 빈 값 제거
train_data['contents'].replace('', np.nan, inplace=True)
train_data = train_data.dropna(how='any')

test_data.drop_duplicates(subset=['contents'], inplace=True)
test_data['contents'] = test_data['contents'].\
    str.replace("[^ㄱ-ㅎㅣ-ㅣ가-힣a-zA-Z<> ]", "")
test_data['contents'].replace('', np.nan, inplace=True)
test_data = test_data.dropna(how='any')

# tokenize
# 불용어 정의
stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍',
             '과', '도', '를', '으로', '자', '에', '와', '한', '하다',
             '이상', '이하', '이내', '미만', '최대', '최소', '자']

# train_data에 형태소 분석기를 사용하여 토큰화
X_train = []
for sentence in train_data['contents']:
    temp_X = []
    temp_X = okt.morphs(sentence, stem=True)    # 토큰화
    temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
    X_train.append(temp_X)

# test_data도 동일하게 토큰화
X_test = []
for sentence in test_data['contents']:
    temp_X = []
    temp_X = okt.morphs(sentence, stem=True)    # 토큰화
    temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
    X_test.append(temp_X)

# Integer encoding
# 훈련 데이터에 대해서 단어 집합(vocabulary) 생성
# 단어 집합이 생성되는 동시에 각 단어에 고유한 정수 부여
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

# 등장 빈도수가 6회 미만인 단어들이 이 데이터에서 얼만큼의 비중을 차지하는지 확인
threshold = 6
total_cnt = len(tokenizer.word_index)   # 단어의 수
rare_cnt = 0    # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0  # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0   # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if value < threshold :
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

# 전체 단어 개수 중 빈도수 2이하인 단어 제거
# 0번 패딩 토큰과 1번 OOV 토큰을 고려하여 +2
vocab_size = total_cnt - rare_cnt + 2

# 제한한 단어 집합의 크기를 Keras Tokenizer의 인자로 넘겨주면,
# Keras Tokenizer는 텍스트 시퀀스를 숫자 시퀀스로 변환 (정수 인코딩)
# 이보다 큰 숫자가 부여된 단어들은 OOV로 변환 (정수 1번 할당)
tokenizer = Tokenizer(vocab_size, oov_token='OOV')
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# saving src tokenizer
with open('../Final/classify_tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# 각 샘플 내의 단어들은 각 단어에 대한 정수로 변환
# 0번 단어는 패딩을 위한 토큰, 1번 단어는 OOV를 위한 토큰
# total_cnt - rare_cnt + 2 이상의 정수는 더 이상 훈련 데이터에 존재하지 않음

# y_train과 y_test를 별도로 저장 (label data)
y_train = np.array(train_data['label'])
y_test = np.array(test_data['label'])

# 빈 샘플(empty sample) 제거
drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]

X_train = np.delete(X_train, drop_train, axis=0)
y_train = np.delete(y_train, drop_train, axis=0)

# padding
max_len = 50

X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

# 데이터 감성 분류

# 임베딩 벡터의 차원 = 100
# 분류를 위해 LSTM 사용
model = Sequential()
model.add(Embedding(vocab_size, 100))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))

# 검증 데이터 손실(val_loss)이 증가 -> 과적합 징후
# => 검증 데이터 손실이 4회 증가하면 학습을 조기 종료(Early Stopping)
# ModelCheckpoint를 사용하여 검증 데이터의 정확도(val_acc)가 이전보다 좋아질 경우에만 모델 저장
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('../Final/classify_best_model.h5', monitor='val_acc', mode='max',
                     verbose=1, save_best_only=True)

# epoch 총 15번 수행
# 훈련 데이터 중 20%를 검증 데이터로 사용하면서 정확도 확인
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=15, callbacks=[es, mc],
                    batch_size=60, validation_split=0.2)

# 훈련 과정에서 검증 데이터의 정확도가 가장 높았을 때 저장된 모델인 'best_model.h5' load
loaded_model = load_model('../Final/classify_best_model.h5')

print("complete")