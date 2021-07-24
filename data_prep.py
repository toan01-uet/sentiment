import regex as re
from typing import Text
import utils
import pandas as pd
from pyvi import ViTokenizer
EMAIL = re.compile(r"([\w0-9_\.-]+)(@)([\d\w\.-]+)(\.)([\w\.]{2,6})")
URL = re.compile(r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))")
PHONE = re.compile(r"(09|01[2|6|8|9])+([0-9]{8})\b")
MENTION = re.compile(r"@.+?:")
NUMBER = re.compile(r"\d+.?\d*")
DATETIME = '\d{1,2}\s?[/-]\s?\d{1,2}\s?[/-]\s?\d{4}'

RE_HTML_TAG = re.compile(r'<[^>]+>')
RE_CLEAR_1 = re.compile("[^_<>\s\p{Latin}]")
RE_CLEAR_2 = re.compile("__+")
RE_CLEAR_3 = re.compile("\s+")

# Import Vietnamese stop words
stopwords = set()
with open("./vietnamese-stopwords-dash.txt", "r") as reader:
    content = reader.readlines()
    for x in content:
        stopwords.add(x.strip())


def remove_stopwords(line):
    words = []
    for word in line.strip().split():
        if word not in stopwords:
            words.append(word)
    return ' '.join(words)


class TextPreprocess:
    @staticmethod
    def replace_common_token(txt):
        """Remove email, link, date, number,... from the text"""
        txt = re.sub(EMAIL, ' ', txt)
        txt = re.sub(URL, ' ', txt)
        txt = re.sub(MENTION, ' ', txt)
        txt = re.sub(DATETIME, ' ', txt)
        txt = re.sub(NUMBER, ' ', txt)
        return txt

    @staticmethod
    def remove_emoji(txt):
        """ Remove commonly used emoji """
        txt = re.sub(':v', '', txt)
        txt = re.sub(':D', '', txt)
        txt = re.sub(':3', '', txt)
        txt = re.sub(':\(', '', txt)
        txt = re.sub(':\)', '', txt)
        return txt

    @staticmethod
    def remove_html_tag(txt):
        return re.sub(RE_HTML_TAG, ' ', txt)

    @staticmethod
    def remove_stopwords(line, stopwords):
        words = []
        for word in line.strip().split():
            if word not in stopwords:
                words.append(word)
        return ' '.join(words)

    def preprocess(self, txt, stopwords, tokenize=True):
        txt = self.replace_common_token(txt)
        txt = self.remove_html_tag(txt)
        txt = re.sub('&.{3,4};', ' ', txt)
        txt = utils.convertwindown1525toutf8(txt)
        if tokenize:
            txt = ViTokenizer.tokenize(txt)
        txt = txt.lower()
        txt = self.remove_emoji(txt)
        txt = re.sub(RE_CLEAR_1, ' ', txt)
        txt = re.sub(RE_CLEAR_2, ' ', txt)
        txt = re.sub(RE_CLEAR_3, ' ', txt)
        txt = utils.chuan_hoa_dau_cau_tieng_viet(txt)
        txt = txt.strip()
        return self.remove_stopwords(txt, stopwords)


if __name__ == '__main__':
    tp = TextPreprocess()
    df = pd.read_csv("./train_data.csv", sep=',')

    df['comment'] = [tp.preprocess(val, stopwords) for val in df['comment']]

    df.to_csv("./clean_train_data.csv", sep='\t')

    # print(tp.preprocess(""))