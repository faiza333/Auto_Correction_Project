import numpy as np
import networkx as nx
import regex
from flask import Flask, request, jsonify, render_template


#------------------- functions -----------------#


# !/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import re
import numpy as np
import pandas as pd
from collections import Counter
import nltk

# In[2]:


with open('shakespeare.txt', 'r', encoding='ISO-8859-1') as f:
    file = f.readlines()


# In[3]:


def process_data(lines):
    """
    Input:
        A file_name which is found in your current directory. You just have to read it in.
    Output:
        words: a list containing all the words in the corpus (text file you read) in lower case.
    """
    words = []
    for line in lines:
        line = line.strip().lower()
        word = re.findall(r'\w+', line)
        words.extend(word)

    return words


# In[4]:


word_l = process_data(file)
vocab = set(word_l)


# print(f"The first ten words in the text are: \n{word_l[0:10]}")
# print(f"There are {len(vocab)} unique words in the vocabulary.")


# In[5]:


def find_wrong_word(sent, vocab):
    wrong_words = []
    sent = sent.strip().lower().split(" ")
    for word in sent:
        if word not in vocab:
            wrong_words.append(word)
    return wrong_words


# In[6]:


def delete_letter(word, verbose=False):
    '''
    Input:
        word: the string/word for which you will generate all possible words
                in the vocabulary which have 1 missing character
    Output:
        delete_l: a list of all possible strings obtained by deleting 1 character from word
    '''

    delete_l = []
    split_l = []
    split_l = [(word[:i], word[i:]) for i in range(len(word))]
    delete_l = [s[0] + s[1][1:] for s in split_l]
    if verbose: print(f"input word : {word} \nsplit_l = {split_l}, \ndelete_l = {delete_l}")

    return delete_l


# In[7]:


def switch_letter(word, verbose=False):
    '''
    Input:
        word: input string
     Output:
        switches: a list of all possible strings with one adjacent charater switched
    '''

    switch_l = []
    split_l = []
    split_l = [(word[:i], word[i:]) for i in range(len(word))]
    for s in split_l:
        if len(s[1]) > 2:
            temp = s[0] + s[1][1] + s[1][0] + s[1][2:]
        elif len(s[1]) == 2:
            temp = s[0] + s[1][1] + s[1][0]
        elif len(s[1]) == 1:
            continue
        switch_l.append(temp)

    if verbose: print(f"Input word = {word} \nsplit_l = {split_l} \nswitch_l = {switch_l}")

    return switch_l


# In[8]:


def replace_letter(word, verbose=False):
    '''
    Input:
        word: the input string/word
    Output:
        replaces: a list of all possible strings where we replaced one letter from the original word.
    '''

    letters = 'abcdefghijklmnopqrstuvwxyz'
    replace_l = []
    split_l = []
    split_l = [(word[:i], word[i:]) for i in range(len(word))]
    for s in split_l:
        if len(s[1]) == 1:
            for l in letters:
                if l != s[1][0]:
                    temp = l
                    replace_l.append(s[0] + temp)
        elif len(s) > 1:
            for l in letters:
                if l != s[1][0]:
                    temp = l + s[1][1:]
                    replace_l.append(s[0] + temp)

    replace_set = set(replace_l)

    # turn the set back into a list and sort it, for easier viewing
    replace_l = sorted(list(replace_set))

    if verbose: print(f"Input word = {word} \nsplit_l = {split_l} \nreplace_l {replace_l}")

    return replace_l


# In[9]:


def insert_letter(word, verbose=False):
    '''
    Input:
        word: the input string/word
    Output:
        inserts: a set of all possible strings with one new letter inserted at every offset
    '''
    letters = 'abcdefghijklmnopqrstuvwxyz'
    insert_l = []
    split_l = []
    split_l = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    for s in split_l:
        for l in letters:
            insert_l.append(s[0] + l + s[1])

    if verbose: print(f"Input word {word} \nsplit_l = {split_l} \ninsert_l = {insert_l}")

    return insert_l


# In[10]:


def edit_one_letter(word, allow_switches=True):
    """
    Input:
        word: the string/word for which we will generate all possible wordsthat are one edit away.
    Output:
        edit_one_set: a set of words with one possible edit. Please return a set. and not a list.
    """

    edit_one_set = set()
    insert_l = insert_letter(word)
    delete_l = delete_letter(word)
    replace_l = replace_letter(word)
    switch_l = switch_letter(word)

    if allow_switches:
        ans = insert_l + delete_l + replace_l + switch_l
    else:
        ans = insert_l + delete_l + replace_l

    edit_one_set = set(ans)

    return edit_one_set


# In[11]:


def edit_two_letters(word, allow_switches=True):
    '''
    Input:
        word: the input string/word
    Output:
        edit_two_set: a set of strings with all possible two edits
    '''

    edit_two_set = set()
    one_edit = edit_one_letter(word)
    ans = []
    for w in one_edit:
        ans.append(w)
        ans.extend(edit_one_letter(w))

    edit_two_set = set(ans)

    return edit_two_set


# In[12]:


def get_count(word_l):
    '''
    Input:
        word_l: a set of words representing the corpus.
    Output:
        word_count_dict: The wordcount dictionary where key is the word and value is its frequency.
    '''
    word_count_dict = {}
    word_count_dict = Counter(word_l)
    return word_count_dict


# In[13]:


word_count_dict = get_count(word_l)


# In[14]:


def get_probs(word_count_dict):
    '''
    Input:
        word_count_dict: The wordcount dictionary where key is the word and value is its frequency.
    Output:
        probs: A dictionary where keys are the words and the values are the probability that a word will occur.
    '''
    probs = {}
    total = 1
    for word in word_count_dict.keys():
        total = total + word_count_dict[word]

    for word in word_count_dict.keys():
        probs[word] = word_count_dict[word] / total
    return probs


# In[15]:


probs = get_probs(word_count_dict)


# In[16]:


def get_corrections(word, probs, vocab, n=2, verbose=False):
    '''
    Input:
        word: a user entered string to check for suggestions
        probs: a dictionary that maps each word to its probability in the corpus
        vocab: a set containing all the vocabulary
        n: number of possible word corrections you want returned in the dictionary
    Output:
        n_best: a list of tuples with the most probable n corrected words and their probabilities.
    '''

    suggestions = []
    n_best = []

    if word in probs.keys():
        suggestions.append(word)
    for w in edit_one_letter(word):
        if len(suggestions) == n:
            break
        if w in probs.keys():
            suggestions.append(w)
    for w in edit_two_letters(word):
        if len(suggestions) == n:
            break
        if w in probs.keys():
            suggestions.append(w)

    best_words = {}

    for s in suggestions:
        best_words[s] = probs[s]

    best_words = sorted(best_words.items(), key=lambda x: x[1], reverse=True)

    n_best = best_words

    if verbose: print("entered word = ", word, "\nsuggestions = ", suggestions)

    return n_best


# In[17]:


def get_correct_word(word, vocab, probs, n):
    corrections = get_corrections(word, probs, vocab, n, verbose=False)
    #    print(corrections)
    if len(corrections) == 0:
        return word

    final_word = corrections[0][0]
    final_prob = corrections[0][1]
    for i, word_prob in enumerate(corrections):
        # print(f"word {i}: {word_prob[0]}, probability {word_prob[1]:.6f}")
        if word_prob[1] > final_prob:
            final_word = word_prob[0]
            final_prob = word_prob[1]
    return final_word


# In[34]:


def autocorrect(sentence, vocab, probs):
    # print("Input sentence : ", sentence)
    wrong_words = find_wrong_word(sentence, vocab)
    # print("Wrong words : ", wrong_words)
    # print(wrong_words)
    correct_words = []
    for word in sentence.strip().lower().split(" "):
        if word in wrong_words:
            correct_word = get_correct_word(word, vocab, probs, 15)
            # print(word, correct_word)
            word = correct_word
        correct_words.append(word)
    x = "Output Sentence : " + " ".join(correct_words).capitalize()
    return x


# In[35]:


# x=autocorrect("honsty is the best pooliccy", vocab, probs)
# x


# In[19]:


def count_n_grams(data, n, start_token='<s>', end_token='<e>'):
    # Initialize dictionary of n-grams and their counts
    n_grams = {}

    for sentence in data:

        # prepend start token n times, and  append <e> one time
        sentence = [start_token] * n + sentence + [end_token]
        sentence = tuple(sentence)

        for i in range(len(sentence) - n):
            n_gram = sentence[i:i + n]
            if n_gram in n_grams.keys():
                n_grams[n_gram] += 1
            else:
                n_grams[n_gram] = 1
    return n_grams


# In[20]:


### SOME UTILITY

def split_to_sentences(data):
    # sentences = data.split("\n")
    sentences = [s.strip() for s in data]
    sentences = [s for s in sentences if len(s) > 0]
    return sentences


def tokenize_sentences(sentences):
    tokenized_sentences = []
    for sentence in sentences:
        sentence = sentence.lower()
        tokenized = nltk.tokenize.word_tokenize(sentence)
        tokenized_sentences.append(tokenized)
    return tokenized_sentences


def get_tokenized_data(data):
    sentences = split_to_sentences(data)
    tokenized_sentences = tokenize_sentences(sentences)
    return tokenized_sentences


# In[21]:


tokenized_data = get_tokenized_data(file)
bigram_counts = count_n_grams(tokenized_data, 2)


# In[22]:


def get_bigram_prob(word, prev_word, bigram_counts, factor):
    key = tuple([prev_word, word])
    # print(key)

    ksum = 0
    occ = 0
    for k, v in bigram_counts.items():
        if k[0] == prev_word:
            ksum = ksum + v
            occ = occ + 1
    # print(ksum)
    # print(occ)

    count = 0
    if key in bigram_counts.keys():
        count = bigram_counts[key]
    # print(type(occ))

    smooth_count = count + factor
    smooth_occ = ksum + occ * factor
    probability = smooth_count / smooth_occ
    # print(probability)
    return probability


# In[23]:


def get_corrections_bigram(word, prev_word, probs, vocab, bigram_counts, unigram_weight=0.3, bigram_weight=0.7, n=5,
                           verbose=False):
    '''
    Input:
        word: a user entered string to check for suggestions
        probs: a dictionary that maps each word to its probability in the corpus
        vocab: a set containing all the vocabulary
        n: number of possible word corrections you want returned in the dictionary
    Output:
        n_best: a list of tuples with the most probable n corrected words and their probabilities.
    '''

    suggestions = []
    n_best = []

    if word in probs.keys():
        suggestions.append(word)
    for w in edit_one_letter(word):
        if len(suggestions) == n:
            break
        if w in probs.keys():
            suggestions.append(w)
    for w in edit_two_letters(word):
        if len(suggestions) == n:
            break
        if w in probs.keys():
            suggestions.append(w)

    best_words = {}

    for s in suggestions:
        # best_words[s] = probs[s]
        unigram_prob = probs[s]
        # print(s)
        try:
            bigram_prob = get_bigram_prob(s, prev_word, bigram_counts, 1)
        except:
            bigram_prob = 0.0000000000000000001

        final_score = unigram_weight * unigram_prob + bigram_weight * bigram_prob

        best_words[s] = final_score

    best_words = sorted(best_words.items(), key=lambda x: x[1], reverse=True)

    n_best = best_words

    if verbose: print("entered word = ", word, "\nsuggestions = ", suggestions)

    return n_best


# In[24]:


def get_correct_word_bigram(word, prev_word, probs, vocab, bigram_counts, unigram_weight, bigram_weight, n):
    corrections = get_corrections_bigram(word, prev_word, probs, vocab,
                                         bigram_counts, unigram_weight, bigram_weight, n, verbose=False)
    # print(corrections)
    if len(corrections) == 0:
        return word

    final_word = corrections[0][0]
    final_prob = corrections[0][1]
    for i, word_prob in enumerate(corrections):
        # print(f"word {i}: {word_prob[0]}, probability {word_prob[1]:.6f}")
        if word_prob[1] > final_prob:
            final_word = word_prob[0]
            final_prob = word_prob[1]
    return final_word


# In[28]:


def autocorrect_bigram(sentence, vocab, probs, bigram_counts):
    # print("Input sentence : ", sentence)
    wrong_words = find_wrong_word(sentence, vocab)
    # print("Wrong words : ", wrong_words)
    # print(wrong_words)
    correct_words = []
    word_list = sentence.strip().lower().split(" ")
    for i, word in enumerate(word_list):
        # print(i, word)

        #### Previous word
        if i == 0:
            prev_word = '<s>'
        else:
            prev_word = word_list[i - 1]

        if word in wrong_words:
            correct_word = get_correct_word_bigram(word, prev_word, probs, vocab, bigram_counts, 0.3, 0.7, 10)
            # print(word, correct_word)
            word = correct_word
        correct_words.append(word)
    x = "Output Sentence : " + " ".join(correct_words).capitalize()
    return x





# x=autocorrect_bigram('she is really beutifule', vocab, probs, bigram_counts)
# x



# ----------FLASK-----------------------------#

app = Flask(__name__)


@app.route('/templates', methods=['POST'])
def original_text_form():
    text = request.form['input_text']
    model = request.form['model']
    # 		print("TEXT:\n",text)
    correction1 = get_corrections(text, probs, vocab, n=9, verbose=False)
    correction2 = autocorrect_bigram(text, vocab, probs, bigram_counts)

    if model == "model1":
        return render_template('index2.html', title="Auto Correction", original_text=text, output_summary=correction1)

    elif model == "model2":
        return render_template('index2.html', title="Auto Correction", original_text=text, output_summary=correction2)

    # summary = generate_summary(text,int(number_of_sent))


# 		print("*"*30)
# 		print(summary)

@app.route('/')
def homepage():
    title = "Auto Correction"
    return render_template('index2.html', title=title)


if __name__ == "__main__":
    app.debug = True
    app.run()
