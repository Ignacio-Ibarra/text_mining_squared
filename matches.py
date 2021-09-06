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
 
#Paquetes NLP 
import nltk
import re
import string
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from sparse_dot_topn import awesome_cossim_topn
from CleaningRE import *


#Lectura de datos
names_minera = pd.read_excel("data/Cuits proveedores mineros.xlsx", sheet_name='Listado CUITs mineria')
names_mectra = pd.read_excel("data/Cuits proveedores mineros.xlsx", sheet_name='Total CUITs')
names_mectra = mectra[mectra.cuit!=' CHAUVIN GERIATRICA S.A.'].reset_index(drop=True)


#Regular Expressions
names_mineras['clean1'] = names_mineras['razon social'].apply(lambda x: pre_processing(str(x),
                                                                                  digits=False,
                                                                                  acronyms_at_end=False,
                                                                                  special_deletions=False))


names_mectra['clean1'] = names_mectra['denominacion'].apply(lambda x: pre_processing(str(x),
                                                                                  digits=False,
                                                                                  acronyms_at_end=False,
                                                                                  special_deletions=False))




#merged = names_mineras.merge(names_mectra, how='left', left_on='clean1', right_on='clean1')

def ngrams2(text, n=2): 
    ngrams = zip(*[text[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

#Representacion Matricial

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
    

NameMineras = names_mineras.clean1.unique().tolist()
NameMectra = names_mectra.clean1.unique().tolist()
    
matches = AxB(NameMineras, NameMectra, output_folder='data',
              analyze_by=ngrams2,
              vectorizing_by="B", lowerbound=0.90,
              topn=10, idfsmooth=False, sublinear=False)

print(matches[:,:])


print(NameMectra[490500])
print(NameMineras[544])



