import pickle

from konlpy.tag import Okt
okt = Okt()

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Classify():
    """
    classify question in two categories
    """
    def __init__(self):
        self.stop_words_ko = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '을', '를', '으로', '자',
                              '에', '와', '한', '하다', '이상', '이하', '이내', '미만', '최대', '최소', '자']

        # loading classify tokenizer
        with open('APP/Data/classify_tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)

        self.max_len = 50
        self.loaded_model = load_model('APP/Data/classify_best_model.h5')

    def classification(self, question):
        if list(question)[:-1] == list("Summary_Test"):
            return 2

        new_sentence = okt.morphs(question, stem=True)                                      # 토큰화
        new_sentence = [word for word in new_sentence if not word in self.stop_words_ko]    # 불용어 제거
        encoded = self.tokenizer.texts_to_sequences([new_sentence])                         # 정수 인코딩
        pad_new = pad_sequences(encoded, maxlen=self.max_len)                               # 패딩
        score = float(self.loaded_model.predict(pad_new))                                   # 예측

        if score > 0.5:
            # 지원 동기 및 포부 관련 질문
            return 0
        else:
            # 경험, 역량 관련 질문
            return 1