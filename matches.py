# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 11:19:05 2021

@author: Ignacio Ibarra
"""
#Paquetes
import pandas as pd
import numpy as np
import random
import os
import time
import matplotlib.pyplot as plt
from functools import reduce
from collections import Counter

#Ruta directorio
os.chdir("C:/Users/Asus/Desktop/CEP/text_mining_squared")
 
#Paquetes NLP 
import nltk
import re
import string
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from sparse_dot_topn import awesome_cossim_topn



#Lectura de datos
names_mineras = pd.read_excel("data/Cuits proveedores mineros.xlsx", sheet_name='Listado CUITs mineria')
names_mectra = pd.read_excel("data/Cuits proveedores mineros.xlsx", sheet_name='Total CUITs')
names_mectra = names_mectra[names_mectra.cuit!=' CHAUVIN GERIATRICA S.A.'].reset_index(drop=True)

#Obtengo Nombres
names_mineras = names_mineras[names_mineras.cuit.isna()].reset_index(drop=True)
names_mectra = names_mectra[names_mectra.denominacion.isna()==False].reset_index(drop=True)

#Regular Expressions
from CleaningRE import *
names_mineras['clean1'] = names_mineras['razon social'].apply(lambda x: pre_processing(str(x),
                                                                                  digits=False,
                                                                                  acronyms_at_end=False,
                                                                                  special_deletions=False))


names_mectra['clean1'] = names_mectra['denominacion'].apply(lambda x: pre_processing(str(x),
                                                                                  digits=False,
                                                                                  acronyms_at_end=False,
                                                                                  special_deletions=False))



merged = names_mineras.merge(names_mectra, how='left', left_on='clean1', right_on='clean1')

ok = merged[merged.denominacion.isna()==False]['clean1'].to_numpy()


def ngrams2(text, n=3): 
    ngrams = zip(*[text[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]


NameMineras = names_mineras.clean1.unique().tolist()
NameMectra = names_mectra.clean1.unique().tolist()

all_names = NameMineras + NameMectra

vectorize_all = TfidfVectorizer(min_df=3, analyzer=ngrams2, lowercase=False, smooth_idf=False, sublinear_tf=False)
tfidf_matrix_all = vectorize_all.fit(all_names)

tfidf_mineras = vectorize_all.transform(NameMineras)
tfidf_mectra = vectorize_all.transform(NameMectra)

match = awesome_cossim_topn(tfidf_mineras, tfidf_mectra.transpose(), ntop=1, lower_bound=0.80)

ver = np.nonzero(match.toarray())

Names = pd.DataFrame()
Names['NameMineras'] = NameMineras
Names['NameMectra'] = 'no_match'
Names['Similarity'] = np.nan

for k,v in zip(ver[0],ver[1]):
    Names.iloc[k,1]=NameMectra[v]
    Names.iloc[k,2] = match[k,v]

ok_plus = Names[Names.NameMectra!="no_match"]['NameMineras'].to_numpy()


Names.to_excel('data/primer_output.xlsx',index=False)

np.setdiff1d(ok_plus,ok)
names_mectra[names_mectra.clean1.str.contains("GOLDER")]


pre_processing("2219292 ALBER23TA LTD", acronyms_at_end=False, digits=False)

a = names_mineras[names_mineras['clean1']=="PRINGLES  SR LEONARDO ALBERTO"]['razon social'].values[0]

names_mectra[names_mectra.clean1.str.contains(r'(?=.*VIDELA)(?=.*JULIO)', regex=True)]['clean1']

b = names_mectra[names_mectra.clean1.str.contains('PATRICIO PALMERO', regex=True)]['clean1']

c = names_mectra[names_mectra.clean1==b]['denominacion'].values[0]

names_mectra[names_mectra.clean1.str.contains('VALLE DEL CURA', regex=True)] 

pre_processing(a, digits=False,acronyms_at_end=False,special_deletions=False)

re.sub('(?:MR|SR|SRES|MISS)\.\s*','',a.upper())

re.sub(r"COMPAďż˝IA|COMPADZ˝IA|COMPA#IA","COMPANIA",c) 

prefixes = ['mr', 'smr']
print('\b(?:' + '|'.join(prefixes) + r')\.\s*')