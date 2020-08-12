import langid as language
from nltk.corpus import stopwords
from nltk.corpus import indian
from nltk.tag import tnt
import stopwordsiso as stopword


def languageDetector(data):
    language_type,accuracy =language.classify(data)
    return stopWords(language_type)

def stopWords(language_type):
    return language_type,stopword.stopwords(language_type)







