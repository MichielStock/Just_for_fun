"""

Author: Michiel Stock
E-mail: michielfmstock@gmail.com

date: Thu 7 April 2015

"""
import random as rd
from math import log

word_file = '2000_families.txt'

file_handler = open(word_file, 'r')
word_list = []

for word in file_handler:
    word_list.append(word.rstrip())
    
file_handler.close()

number_words = len(word_list)

n_choices = 5

# entropy in bits
entropy = n_choices * log( number_words ) / log( 2 )

words_for_password = rd.sample( word_list , n_choices)

print '_'.join(words_for_password), 'entropy: ' , entropy

