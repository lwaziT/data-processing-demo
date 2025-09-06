import nltk
import logging
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# nltk.download('punkt_tab')
# nltk.download('stopwords')
# nltk.download('wordnet')

with open("C:/Users/tobos/OneDrive/Desktop/UNISA 2025/COS4861_NLP/Assignments/Assignment 2/Data preprocessing pipeline/data/Corpus.txt", "r") as file:
    content = file.read()

def tokenize(para):
    """ Method to split the text into tokens."""
    tokenizer = nltk.data.load('tokenizers/punkt/PY3/english.pickle')
    para_to_sentence = tokenizer.tokenize(para) # Accepts paragraph and tokenizes to sentences.
    tokenizer = TreebankWordTokenizer() # Creating an instance of TreebankWordTokenizer() to further segment sentences to words.
    sentence_list = []
    word_list = []
    for sentence in para_to_sentence:
        sentence_list.append(sentence)

    for sentence in sentence_list:
        sent_to_word = tokenizer.tokenize(sentence) # Accepts sentence and tokenizes to words.
        for word in sent_to_word:
            word_list.append(word) # Appends individual words to list.

    return sentence_list, word_list # Returns a list of tokens.

def stop_words_removal(list_of_words):
    """ Method to remove common stop words from a list of words. """
    stop_words = set(stopwords.words('english'))

    return [word for word in list_of_words if word.lower() not in stop_words]

def stemming(para):
    """ Method used to perform stemming on a list of
    words to reduce words to their base forms. """
    stemmer = PorterStemmer()
    stemmed_list = []
    for word in para:
        stemmed_list.append(stemmer.stem(word))
    return stemmed_list

def lemmatizing(para):
    """ Method used to perform lemmatization on a list
    of words to reduce words to their dictionary form. """
    lemmatizer = WordNetLemmatizer()
    lemmatized_list = []
    for word in para:
        lemmatized_list.append(lemmatizer.lemmatize(word, pos='v'))
    return lemmatized_list

if __name__ == '__main__':

    _,tokenized_para = tokenize(content)
    logging.info(f"tokenized_text: {tokenized_para}")

    stopworded_para = stop_words_removal(tokenized_para)
    logging.info(f"stopworded_text: {stopworded_para}")

    stemmitized_para = stemming(stopworded_para)
    logging.info(f"stemmitized_text: {stemmitized_para}")

    logging.info(f"Lemmitized_text: {lemmatizing(stemmitized_para)}")


