import pickle
import numpy as np
np.random.seed(seed=0)

from nltk.corpus import stopwords

from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from attention import AttentionLayer

import re
from bs4 import BeautifulSoup

# import nltk
# nltk.download('stopwords')

import json
import urllib.request
import _private as pr

class Summary():
    """
    Summary inputed drafts by user
    """
    def __init__(self):
        self.text_max_len = 50
        self.summary_max_len = 8

        self.src_vocab = 8000
        # loading src tokenizer
        with open('APP/Data/src_tokenizer_v1.pickle', 'rb') as handle:
            self.src_tokenizer = pickle.load(handle)

        self.tar_vocab = 2000
        # loading tar tokenizer
        with open('APP/Data/tar_tokenizer_v1.pickle', 'rb') as handle:
            self.tar_tokenizer = pickle.load(handle)

        self.embedding_dim = 128
        self.hidden_size = 256

        # 전처리 함수 내 사용
        self.contractions = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because",
                             "could've": "could have", "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
                             "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                             "he'd": "he would", "he'll": "he will", "he's": "he is", "how'd": "how did",
                             "how'd'y": "how do you", "how'll": "how will", "how's": "how is", "I'd": "I would",
                             "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have", "I'm": "I am",
                             "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",
                             "i'll've": "i will have", "i'm": "i am", "i've": "i have", "isn't": "is not",
                             "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have",
                             "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not",
                             "might've": "might have", "mightn't": "might not", "mightn't've": "might not have",
                             "must've": "must have", "mustn't": "must not", "mustn't've": "must not have",
                             "needn't": "need not", "needn't've": "need not have", "o'clock": "of the clock",
                             "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                             "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would",
                             "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have",
                             "she's": "she is", "should've": "should have", "shouldn't": "should not",
                             "shouldn't've": "should not have", "so've": "so have", "so's": "so as", "this's": "this is",
                             "that'd": "that would", "that'd've": "that would have", "that's": "that is",
                             "there'd": "there would", "there'd've": "there would have", "there's": "there is",
                             "here's": "here is", "they'd": "they would", "they'd've": "they would have",
                             "they'll": "they will", "they'll've": "they will have", "they're": "they are",
                             "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would",
                             "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                             "we've": "we have", "weren't": "were not", "what'll": "what will",
                             "what'll've": "what will have", "what're": "what are", "what's": "what is",
                             "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did",
                             "where's": "where is", "where've": "where have", "who'll": "who will",
                             "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is",
                             "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                             "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",
                             "y'all": "you all", "y'all'd": "you all would", "y'all'd've": "you all would have",
                             "y'all're": "you all are", "y'all've": "you all have", "you'd": "you would",
                             "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                             "you're": "you are", "you've": "you have"}
        # NLTK의 불용어
        self.stop_words_en = set(stopwords.words('english'))

        self.encoder_model = None
        self.decoder_model = None

        self.tar_word_to_index = None
        self.tar_index_to_word = None

    def decode_layer(self):
        # 인코더
        encoder_inputs = Input(shape=(self.text_max_len,))

        # 인코더의 임베딩 층
        enc_emb = Embedding(self.src_vocab, self.embedding_dim)(encoder_inputs)

        # 인코더의 LSTM 1
        encoder_lstm1 = LSTM(self.hidden_size, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.4)
        encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

        # 인코더의 LSTM 2
        encoder_lstm2 = LSTM(self.hidden_size, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.4)
        encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

        # 인코더의 LSTM 3
        encoder_lstm3 = LSTM(self.hidden_size, return_state=True, return_sequences=True, dropout=0.4, recurrent_dropout=0.4)
        encoder_outputs, state_h, state_c = encoder_lstm3(encoder_output2)

        # 디코더
        decoder_inputs = Input(shape=(None,))

        # 디코더의 임베딩 층
        dec_emb_layer = Embedding(self.tar_vocab, self.embedding_dim)
        dec_emb = dec_emb_layer(decoder_inputs)

        # 디코더의 LSTM
        decoder_lstm = LSTM(self.hidden_size, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.2)
        decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=[state_h, state_c])

        # 어텐션 층(어텐션 함수)
        attn_layer = AttentionLayer(name='attention_layer')
        attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

        # 어텐션의 결과와 디코더의 hidden state들을 연결
        decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

        # 디코더의 출력층
        decoder_softmax_layer = Dense(self.tar_vocab, activation='softmax')
        decoder_softmax_outputs = decoder_softmax_layer(decoder_concat_input)

        # 모델 정의
        model = Model([encoder_inputs, decoder_inputs], decoder_softmax_outputs)
        model.summary()
        model.load_weights('APP/Data/attention_best_model_v1.h5')
        model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

        # seq2seq + attention으로 요약
        self.tar_word_to_index = self.tar_tokenizer.word_index  # 요약 단어 집합에서 단어 -> 정수를 얻음
        self.tar_index_to_word = self.tar_tokenizer.index_word  # 요약 단어 집합에서 정수 -> 단어를 얻음

        # 인코더 설계
        self.encoder_model = Model(inputs=encoder_inputs,
                                   outputs=[encoder_outputs, state_h, state_c])

        # 이전 시점의 상태들을 저장하는 텐서
        decoder_state_input_h = Input(shape=(self.hidden_size,))
        decoder_state_input_c = Input(shape=(self.hidden_size,))

        dec_emb2 = dec_emb_layer(decoder_inputs)
        # 문장의 다음 단어를 예측하기 위해서 초기 상태(initial_state)를 이전 시점의 상태로 사용
        decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h,
                                                                                          decoder_state_input_c])

        # 어텐션 함수
        decoder_hidden_state_input = Input(shape=(self.text_max_len, self.hidden_size))
        attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
        decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])

        # 디코더의 출력층
        decoder_outputs2 = decoder_softmax_layer(decoder_inf_concat)

        # 최종 디코더 모델
        self.decoder_model = Model(
            [decoder_inputs] + [decoder_hidden_state_input, decoder_state_input_h, decoder_state_input_c],
            [decoder_outputs2] + [state_h2, state_c2])

    def decode_sequence(self, input_seq):
        # 입력으로부터 인코더의 상태를 얻음
        e_out, e_h, e_c = self.encoder_model.predict(input_seq)

        # <SOS>에 해당하는 토큰 생성
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = self.tar_word_to_index['sostoken']

        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:  # stop_condition이 True가 될 때까지 루프 반복

            output_tokens, h, c = self.decoder_model.predict([target_seq] + [e_out, e_h, e_c])
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_token = self.tar_index_to_word[sampled_token_index]

            if (sampled_token != 'eostoken'):
                decoded_sentence += ' ' + sampled_token

            #  <eos>에 도달하거나 최대 길이를 넘으면 중단.
            if (sampled_token == 'eostoken' or len(decoded_sentence.split()) >= (self.summary_max_len - 1)):
                stop_condition = True

            # 길이가 1인 타겟 시퀀스를 업데이트
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index

            # 상태 업데이트
            e_h, e_c = h, c

        return decoded_sentence

    # 전처리 함수
    def preprocess_sentence(self, sentence, remove_stopwords=True):
        sentence = sentence.lower()  # 텍스트 소문자화
        sentence = BeautifulSoup(sentence, "lxml").text  # <br />, <a href = ...> 등의 html 태그 제거
        sentence = re.sub(r'\([^)]*\)', '',
                          sentence)  # 괄호로 닫힌 문자열  제거 Ex) my husband (and myself) for => my husband for
        sentence = re.sub('"', '', sentence)  # 쌍따옴표 " 제거
        sentence = ' '.join([self.contractions[t] if t in self.contractions else t for t in sentence.split(" ")])  # 약어 정규화
        sentence = re.sub(r"'s\b", "", sentence)  # 소유격 제거. Ex) roland's -> roland
        sentence = re.sub("[^a-zA-Z]", " ", sentence)  # 영어 외 문자(숫자, 특수문자 등) 공백으로 변환
        sentence = re.sub('[m]{2,}', 'mm', sentence)  # m이 3개 이상이면 2개로 변경. Ex) ummmmmmm yeah -> umm yeah

        # 불용어 제거 (Text)
        if remove_stopwords:
            tokens = ' '.join(word for word in sentence.split() if not word in self.stop_words_en if len(word) > 1)
        # 불용어 미제거 (Summary)
        else:
            tokens = ' '.join(word for word in sentence.split() if len(word) > 1)
        return tokens

    def translate(self, txt, src, tar):
        encText = urllib.parse.quote(txt)
        data = "source=" + src + "&target=" + tar + "&text=" + encText
        url = "https://openapi.naver.com/v1/papago/n2mt"
        request = urllib.request.Request(url)
        request.add_header("X-Naver-Client-Id", pr.client_id)
        request.add_header("X-Naver-Client-Secret", pr.client_secret)
        response = urllib.request.urlopen(request, data=data.encode("utf-8"))
        rescode = response.getcode()

        if (rescode == 200):
            response_body = response.read()
            return json.loads(response_body)['message']['result']['translatedText']
        else:
            print("Error Code:" + rescode)
            return

    def summarization(self, draft):
        self.decode_layer()

        test_text = self.translate(draft, "ko", "en")
        # Text 열 전처리
        test_text = self.preprocess_sentence(test_text)
        test_text_list = test_text.split(' ')

        # 텍스트 시퀀스를 정수 시퀀스로 변환
        test_sequence = self.src_tokenizer.texts_to_sequences(test_text_list)
        test_sequence = [s for seq in test_sequence for s in seq]

        test_sequence = pad_sequences([test_sequence], maxlen=self.text_max_len, padding='post')

        print("예측 요약문 :", self.decode_sequence(test_sequence.reshape(1, self.text_max_len)))

        test_sequence = [s for seq in test_sequence for s in seq]
        test_sequence = np.array(test_sequence)

        summary = self.translate(self.decode_sequence(test_sequence.reshape(1, test_sequence.size)), "en", "ko")
        return summary