# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 13:24:49 2021

@author: Asus
"""
import pandas as pd
import numpy as np
import string
import unicodedata
import re
from functools import reduce

def strip_accents(s):
   return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')
 
def del_punct_wsp(text): 
    text = text.upper()
    text = re.sub('(?:MR|SR|SRA|SRTA|SRES|MISS)\.\s*','',text)
    text = re.sub(r'\([^)]*\)', '', text) #remueve paréntesis y todo lo de adentro
    text = text.replace(".","").replace('('," ").replace(")"," ")
    text = text.replace("\M:"," ").replace("M:"," ")
    text = re.sub(r'[!"\#\$%\'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}\~]',' ',text) #borra punct y agrega espacio
    text = re.sub(r'\d+\b',' ', text)
    text = strip_accents(text)
    return text

# =============================================================================
# glei = pd.read_csv('https://www.gleif.org/content/2-about-lei/7-code-lists/2-iso-20275-entity-legal-forms-code-list/2020-11-19_elf-code-list-v1.3.csv')
# ab = ";".join(glei[glei['Abbreviations Local language'].isna()==False]["Abbreviations Local language"].drop_duplicates().values.tolist())
# ab = np.unique(np.array(ab.split(";"))).tolist()
# abreviaturas = np.unique(np.array([x.upper() for x in ab])).tolist()+["REF"]
# indices = list(range(1,len(abreviaturas)+1))
# abrev_dict = dict()
# for k,v in zip(abreviaturas, indices): 
#   abrev_dict[k]=v
# =============================================================================

def special_corrections(text): 
    text = re.sub(r"\bCOMPA.*IA\b","COMPANIA",text)
    text = re.sub(r"\bLA VICU.*A\b","LA VICUNA",text)
    text = re.sub(r"\bMONTADZ.*A\b", "MONTANA", text)
    text = re.sub(r"DZ˝","N",text) 
    text = re.sub(r"\bASIJEMIN\b", "ASOCIACION SINDICAL DEL PERSONAL JERARQUICO PROFESIONAL Y TECNICO DE LA ACTIVIDAD MINERA ARGENTINA", text)
    text = re.sub(r"\bS A I C Y A\b","SAICYA", text)
    text = re.sub(r"\bS A C I\b","SACI",text)
    text = re.sub(r"\bSAIC Y F\b","SAICYF", text)
    text = re.sub(r"\bSA IC Y F\b","SAICYF",text)
    text = re.sub(r"\bPROD Y SERVICIOS\b","PRODUCTOS Y SERVICIOS",text)
    text = re.sub(r"\bSA\b|\bS A\b|\bSOCIEDAD ANONIMA\b","SA", text)
    text = re.sub(r"\bS R L\b|\bSOCIEDAD DE RESPONSABILIDAD LIMITADA\b","SRL", text)
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
  return " ".join(text.split())

def sepecial_deletions(text, acronyms_list_or_dict): 
  return " ".join([word for word in text.split(" ") if word not in acronyms_list_or_dict])

def pre_processing(text, 
                   punctuation=True, 
                   within_spaces=True, 
                   digits=True, 
                   strip_space=True, 
                   acronyms_at_end=True,
                   special_deletions = None,
                   specialcorr=True): 
    
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
    if specialcorr: 
        text=special_corrections(text)

    return text

def ngrams(text, n=3): 
    ngrams = zip(*[text[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

def AxB(listA, listaB, output_folder, vectorizing_by="A", analyze_by='word', lowerbound=0.8, topn=10, idfsmooth=True, sublinear=True): 
    #Vectorizer
    vectorizer = TfidfVectorizer(min_df=3, analyzer=analyze_by, lowercase=False, smooth_idf=idfsmooth, sublinear_tf=sublinear)
    
    ''' 
    * vectorizing_by="A" es producto de AxB.transpose() 
    y los features son de A
    
    * vectorizing_by="B" es producto de AxB.transpose()
    y los features son de B
    '''
     
    if vectorizing_by=="A": 
        print("TF-IDF Vectorizig...\n")
        A = vectorizer.fit_transform(listA)
        B = vectorizer.transform(listaB)
        print("Processing Matches...\n")
    
    if vectorizing_by=="B": 
        print("TF-IDF Vectorizig...\n")
        B = vectorizer.fit_transform(listaB)
        A = vectorizer.transform(listA)
        print("Processing Matches...\n")
    #Sparse Matrix dot product
    import time
    t1 = time.time()
    matches_ngrams = awesome_cossim_topn(A,B.transpose(), topn, lowerbound)
    t = time.time()-t1
    print('This program has runned in {} seconds\n'.format(t))
    
    #Saving Matrix
    from scipy import sparse
    from datetime import datetime
    outputpath = output_folder+"/"+"matches_{}.npz".format(datetime.now().strftime('%Y-%m-%d %H_%M_%S'))
    sparse.save_npz(outputpath, matches_ngrams)
    print("Matches save into {}".format(outputpath))
        
    return matches_ngrams