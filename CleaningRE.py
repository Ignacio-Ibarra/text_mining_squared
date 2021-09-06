# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 13:24:49 2021

@author: Asus
"""
import string
import unicodedata
import re
from functools import reduce

def strip_accents(s):
   return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')
 
def del_punct_wsp(text): 
    text = text.replace(".","").replace('('," ").replace(")"," ")
    text = text.replace("\M:"," ").replace("M:"," ")
    text = re.sub(r'[!"\#\$%\'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}\~]',' ',text) #borra punct y agrega espacio
    text = re.sub(r'^\d+$',' ', text)
    text = strip_accents(text)
    return text

def acronyms(text): 
  if ''==text: 
    return ''
  else:
    text = text.upper()
    text = text.split(' ')
    while (text[-1] in abrev_dict) and (len(text)>2):
      text = text[:-1]
      acronyms(' '.join(text))
    return ' '.join(text)
 
def remove_digits(text): 
  splitted = text.split(' ') 
  cleanned = []
  for word in splitted:
    evaluation = [1 if i.isdigit() else 0 for i in word]
    suma = reduce(lambda x,y: x+y, evaluation,0)
    if suma==0:
      cleanned.append(word)
    elif suma<2:
      cleanned.append(word)
    else: 
      word = ''.join([i for i in word if not i.isdigit()])
      cleanned.append(word)
  return " ".join(cleanned)

def strip_spaces(text): 
  return text.upper().lstrip().rstrip()

def remove_within_wsp(text):
  return re.sub(r'\s', " ", text)

def sepecial_deletions(text, acronyms_list_or_dict): 
  return " ".join([word for word in text.split(" ") if word not in acronyms_list_or_dict])

def pre_processing(text, 
                   punctuation=True, 
                   within_spaces=True, 
                   digits=True, 
                   strip_space=True, 
                   acronyms_at_end=True,
                   special_deletions = None): 
    
    """1) Se borra puntuación, acentos y caracteres específicos como "\M:"
       2) Se borran dígitos
       3) Se remueven espacios en blanco de principio y final
       4) Se borran las siglas al final del texto
       5) Se remueven espacios dentro del texto"""
    if punctuation:
      text = del_punct_wsp(text)
      #print(text)
    if within_spaces:
      text = remove_within_wsp(text)
      #print(text)
    if digits:
      text = remove_digits(text)
      #print(text)
    if strip_space:
      text = strip_spaces(text)
      #print(text)
    if special_deletions:
      text = special_deletions(text, special_deletions) 
      #print(text)
    if acronyms_at_end:
      text = acronyms(text)
      #print(text)
    if within_spaces:
      text = remove_within_wsp(text)  

    return text

def ngrams(text, n=3): 
    ngrams = zip(*[text[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]