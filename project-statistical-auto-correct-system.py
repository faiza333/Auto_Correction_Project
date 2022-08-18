#!/usr/bin/env python
# coding: utf-8

# # <p style="background-color:skyblue; font-family:newtimeroman; font-size:180%; text-align:center">Introduction</p>
# 
# I'm often amazed by the auto-correct systems that are used by Google, Grammerly, Android etc. I wanted to know how these systems work and I started reading about them. And as expected...They are really complex!! If you too want to read about them. Here you go - 
# - [Using the Web for Language Independent Spellchecking and
# Autocorrection](http://static.googleusercontent.com/media/research.google.com/en/us/pubs/archive/36180.pdf)
# - [How Difficult is it to Develop a Perfect Spell-checker?
# A Cross-linguistic Analysis through Complex Network Approach](http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=52A3B869596656C9DA285DCE83A0339F?doi=10.1.1.146.4390&rep=rep1&type=pdf)
# 
# > Then I found an intutive auto correct system that uses statistics, probability and dynamic programming. In this project I have tried to impliment this auto-correct system. [References at the bottom]
# 
# Let's begin!

# In[1]:


import os
import re
import numpy as np 
import pandas as pd 
from collections import Counter
import nltk




# In[2]:


with open('shakespeare2.txt', 'r', encoding='ISO-8859-1') as f:
    file = f.readlines()





# Cool. Now we need to process this corpus. Since it's pretty clean corpus we need to do only two thisga - Tokenizing and Lowercasing.

# In[4]:


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





# The data looks fine. Before moving to the next step let's first look at the architectire of our syste.

# # <p style="background-color:skyblue; font-family:newtimeroman; font-size:180%; text-align:center">Architecture</p>
# 
# <div>
# <img style="align:center", src="https://github.com/pashupati98/kaggle-archives/blob/main/img/architecture.png?raw=true">
#     <hr>
# </div>
# 
# This auto-correct architecture has 4 components -
# - 1) Filtering Mispells : One simple approach could be checking if a word is there in the vocabulary or not. 
# - 2) Word Suggestion Mechanism : This mechnism suggests candidate words based on deletion, insertion, switch or replace of one/two characters in the original word.
# - 3) Probability Distribution Mechanism : The probability distribution {key(word) : value(probability)} is created calculated using a large text corpus. Probability of each candidate is found using this distribution and the most probable candidate is the final one.
# - 4) Replace Mispells : Simple replace the mispelled word with the most probable suggestion.
# 
# We'll impliment each part separetely.

# ### Artchitecture Part 1 : (Filtering Mispells)
# 
# A function that tokenizes the sentences and checks the availability of each word in the vocabulary.

# In[6]:


def find_wrong_word(sent, vocab):
    wrong_words = []
    sent = sent.strip().lower().split(" ")
    for word in sent:
        if word not in vocab:
            wrong_words.append(word)
    return wrong_words





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
    delete_l = [s[0]+s[1][1:] for s in split_l]
    if verbose: print(f"input word : {word} \nsplit_l = {split_l}, \ndelete_l = {delete_l}")

    return delete_l





# In[10]:


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
        if len(s[1])>2:
            temp = s[0] + s[1][1] + s[1][0] + s[1][2:]
        elif len(s[1]) == 2:
            temp = s[0] + s[1][1] + s[1][0]
        elif len(s[1]) == 1:
            continue
        switch_l.append(temp)
    
    if verbose: print(f"Input word = {word} \nsplit_l = {split_l} \nswitch_l = {switch_l}") 

    return switch_l



# In[12]:


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
                    replace_l.append(s[0]+temp)
        elif len(s) > 1:
            for l in letters:
                if l != s[1][0]:
                    temp = l + s[1][1:]
                    replace_l.append(s[0]+temp)
        
    replace_set = set(replace_l)
    
    # turn the set back into a list and sort it, for easier viewing
    replace_l = sorted(list(replace_set))
    
    if verbose: print(f"Input word = {word} \nsplit_l = {split_l} \nreplace_l {replace_l}")   
    
    return replace_l




# In[15]:


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
    split_l = [(word[:i], word[i:]) for i in range(len(word)+1)]
    for s in split_l:
        for l in letters:
            insert_l.append(s[0]+l+s[1])

    if verbose: print(f"Input word {word} \nsplit_l = {split_l} \ninsert_l = {insert_l}")
    
    return insert_l




# In[17]:


def edit_one_letter(word, allow_switches = True):
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





# In[19]:


def edit_two_letters(word, allow_switches = True):
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




# In[23]:


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
        probs[word] = word_count_dict[word]/total
    return probs







# In[27]:


def get_corrections(word, probs, vocab, n=2, verbose = False):
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


# In[28]:


def get_correct_word(word, vocab, probs, n): 
    corrections = get_corrections(word, probs, vocab, n, verbose=False)
#    print(corrections)
    if len(corrections) == 0:
        return word
    
    final_word = corrections[0][0]
    final_prob = corrections[0][1]
    for i, word_prob in enumerate(corrections):
        #print(f"word {i}: {word_prob[0]}, probability {word_prob[1]:.6f}")
        if word_prob[1] > final_prob:
            final_word = word_prob[0]
            final_prob = word_prob[1]
    return final_word




# In[30]:


def autocorrect(sentence, vocab, probs):
    print("Input sentence : ", sentence)
    wrong_words = find_wrong_word(sentence, vocab)
    print("Wrong words : ", wrong_words)
    #print(wrong_words)
    correct_words = []
    for word in sentence.strip().lower().split(" "):
        if word in wrong_words:
            correct_word = get_correct_word(word, vocab, probs, 15)
            #print(word, correct_word)
            word = correct_word
        correct_words.append(word)
    print("Output Sentence : ", " ".join(correct_words).capitalize())


# ## Demo
# 
# Let's check this system on some examples

# In[31]:



# In[ ]:





# ### We can see that it is working!
# 
# This gives a overview of what auto-correct systems are and how they work.
# 
# Note - This is very simplified architecture compared to what is used in reality. You can see in the last example's output isn't good. It is supposed to be ("Life is a drink and love is a drug")
# 
# #### Drawbacks 
# - It has fixed outcome. i.e. 'hime' will be converted to 'time' only not 'home' oe anything else.
# - It is solely based on frequency of words in the corpus
# - Doesn't care about the contex.
# - Can't suggest something which is not in the vocabulary
# 
# #### Improvements
# - It can be further improved by introducing bi-gram probabilities. Hence, it will get some inference from previous words.
# - The suggestions that are less distance away from the misspelled word are more likely. Hence, the system can be further improved by introducing dynamic programming based min edit distance functionality.
# 
# Let's implement these improvements.

# <hr>
# 
# ## Improvement 1 : Introducing n-gram probabilities to get context from previous words

# This idea is taken from the n-grams language models. In a n-gram language model
# - Assume the probability of the next word depends only on the previous n-gram.
# - The previous n-gram is the series of the previous 'n' words.
# 
# The conditional probability for the word at position 't' in the sentence, given that the words preceding it are $w_{t-1}, w_{t-2} \cdots w_{t-n}$ is:
# 
# $$ P(w_t | w_{t-1}\dots w_{t-n}) \tag{1}$$
# 
# This probability cab be estimated by counting the occurrences of these series of words in the training data.
# - The probability can be estimated as a ratio, where
# - The numerator is the number of times word 't' appears after words t-1 through t-n appear in the training data.
# - The denominator is the number of times word t-1 through t-n appears in the training data.
# 
# $$ \hat{P}(w_t | w_{t-1}\dots w_{t-n}) = \frac{C(w_{t-1}\dots w_{t-n}, w_n)}{C(w_{t-1}\dots w_{t-n})} \tag{2} $$
# 
# In other words, to estimate probabilities based on n-grams, first find the counts of n-grams (for denominator) then divide it by the count of (n+1)-grams (for numerator).
# 
# - The function $C(\cdots)$ denotes the number of occurence of the given sequence. 
# - $\hat{P}$ means the estimation of $P$. 
# - The denominator of the above equation is the number of occurence of the previous $n$ words, and the numerator is the same sequence followed by the word $w_t$.

# Now the issue with above formula is that it doesn't work when a count of an n-gram is zero..
# - Suppose we encounter an n-gram that did not occur in the training data.  
# - Then, the equation (2) cannot be evaluated (it becomes zero divided by zero).
# 
# A way to handle zero counts is to add k-smoothing.  
# - K-smoothing adds a positive constant $k$ to each numerator and $k \times |V|$ in the denominator, where $|V|$ is the number of words in the vocabulary.
# 
# $$ \hat{P}(w_t | w_{t-1}\dots w_{t-n}) = \frac{C(w_{t-1}\dots w_{t-n}, w_n) + k}{C(w_{t-1}\dots w_{t-n}) + k|V|} \tag{3} $$
# 
# 
# For n-grams that have a zero count, the equation (3) becomes $\frac{1}{|V|}$.
# - This means that any n-gram with zero count has the same probability of $\frac{1}{|V|}$.
# 
# Now, let's define a function that computes the probability estimate (3) from n-gram counts and a constant $k$.

