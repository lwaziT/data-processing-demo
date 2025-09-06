import nltk # Import python library that is used for NLP and text analytics.
from nltk.tokenize import TreebankWordTokenizer # Import TreebankWordTokenizer class to help us tokenize sentences into words.
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import seaborn as sns

corpus = ("When data are noisy, it’s our job as data scientists "
          "to listen for signals so we can relay it to someone who can decide how to act. "
          "To amp up how loudly hidden signals speak over the noise of big and/or volatile data, "
          "we can deploy smoothing algorithms, which though traditionally used in time-series analyses, "
          "also come into their own when applied on other sequential data. "
          "Smoothing algorithms are either global or local because they take data and filter out "
          "noise across the entire, global series, or over a smaller, local series by summarizing a local or "
          "global domain of Y, resulting in an estimation of the underlying data called a smooth. "
          "The specific smoother you use depends on your analysis’ goal and data quirks, "
          "because as we’ll see below, there are trade-offs to consider. Below are a few options, "
          "along with their intuition, limitations, and formula so you can rapidly evaluate when and why to use "
          "one over the other.")

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

    sentences, words = tokenize(corpus)
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