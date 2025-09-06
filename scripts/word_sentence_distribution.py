import nltk # Import python library that is used for NLP and text analytics.
from nltk.tokenize import TreebankWordTokenizer # Import TreebankWordTokenizer class to help us tokenize sentences into words.
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import seaborn as sns

with open("C:/Users/tobos/OneDrive/Desktop/UNISA 2025/COS4861_NLP/Assignments/Assignment 2/Data preprocessing pipeline/data/Corpus.txt", "r") as file:
    content = file.read()

def tokenize(para):
    """ Method to split the text into tokens."""
    tokenizer = nltk.data.load('tokenizers/punkt/PY3/english.pickle') # Creates a tokenizer that helps us tokenize English sentences (loaded from the pickle file)
    para_to_sentence = tokenizer.tokenize(para) # Accepts paragraph and tokenizes to sentences.
    tokenizer = TreebankWordTokenizer() # Creating an instance of TreebankWordTokenizer() to further segment sentences to words.
    sentence_list = [] # List that will store sentences.
    word_list = [] # list that will store tokens (words).

    for sentence in para_to_sentence: # for-loop extracts each sentence from the list "para_to_sentence"
        sentence_list.append(sentence) # Accepts sentence, and appends it to the list that will store sentences.

    for sentence in sentence_list: # for-loop extracts sentence from the list of sentences
        sent_to_word = tokenizer.tokenize(sentence) # Accepts sentence and tokenizes to words, then stores the tokenized word into variable sent_to_word, creating a list of tokenized words.
        for word in sent_to_word: # for-loop then extracts each word from the list "sent_to_word"
            word_list.append(word) # Appends individual words to list.

    return sentence_list, word_list # Returns a list of tokens.

if __name__ == '__main__':

    sentences, words = tokenize(content)
    # Word frequency distribution
    word_freq = FreqDist(words)
    # Sentence length distribution
    sentence_lengths = [len(nltk.word_tokenize(s)) for s in sentences]

    # Plot top 10 most common words
    common_words = word_freq.most_common(10)
    labels, counts = zip(*common_words)

    plt.figure(figsize=(8, 4))
    sns.barplot(x=list(labels), y=list(counts), palette="viridis")
    plt.title("Top 10 Most Common Words")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.show()

    plt.figure(figsize=(6, 4))
    sns.histplot(sentence_lengths, bins=5, kde=False, color="blue")
    plt.title("Sentence Length Distribution")
    plt.xlabel("Number of words per sentence")
    plt.ylabel("Number of sentences")
    plt.show()