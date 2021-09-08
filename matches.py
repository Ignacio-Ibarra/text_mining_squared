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



#merged = names_mineras.merge(names_mectra, how='left', left_on='clean1', rig#ht_on='clean1')

#ok = np.sort(merged[merged.denominacion.isna()==False]['clean1'].to_numpy())


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

match = awesome_cossim_topn(tfidf_mineras, tfidf_mectra.transpose(), ntop=1, lower_bound=0.8341)

ver = np.nonzero(match.toarray())

Names = pd.DataFrame()
Names['NameMineras'] = NameMineras
Names['NameMectra'] = np.nan
Names['Similarity'] = np.nan
Names['Matcheo'] = np.nan


for k,v in zip(ver[0],ver[1]):
    Names.iloc[k,1]=  NameMectra[v]
    Names.iloc[k,2] = match[k,v]
    if match[k,v] >= 0.8341: 
       Names.iloc[k,3] = "ok"
       

Names = Names.sort_values(by='Similarity', ascending=False).reset_index(drop=True)
#ok_plus = np.sort(Names[Names.NameMectra.isna()==False]['NameMineras'].to_numpy())

Names.to_excel('data/primer_output.xlsx',index=False)

# =============================================================================
# MERGE DE RESULTADOS
# =============================================================================
devolucion = pd.read_excel('data/primer_output.xlsx')

devolucion['cuit'] = "no_match"
devolucion['clae'] = "no_match"
for i in range(len(devolucion)):
    if devolucion.iloc[i,3]=="ok":
        nmectra = devolucion.iloc[i,1]
        clae = names_mectra.loc[names_mectra.clean1==nmectra,"clae6"].values
        cuit = names_mectra.loc[names_mectra.clean1==nmectra,"cuit"].values
        if len(cuit)>1:
            devolucion.iloc[i,4] = "multiple_match: "+" - ".join([str(int(x)) for x in cuit])        
        else:
            cuit = int(cuit[0])
            devolucion.iloc[i,4] = cuit
        if len(clae)>1:
            devolucion.iloc[i,5] = "multiple_match: "+" - ".join([str(int(x)) for x in clae])
        else: 
            clae = int(clae[0])
            devolucion.iloc[i,5]=clae
    if devolucion.iloc[i,3]=="no":
        devolucion.iloc[i,1]="no_match"
        devolucion.iloc[i,2] ="no_match"
        
        
devolucion.fillna("no_match", inplace=True)

a = names_mineras.merge(devolucion, left_on="clean1", right_on="NameMineras", how='left')
a.columns

a = a[['razon social','cuit_y','clae_y']]

a.columns = ['razon social','cuit','clae']

a[a.cuit!="no_match"].shape

mineras_ok = pd.read_excel("data/Cuits proveedores mineros.xlsx", sheet_name='Listado CUITs mineria')

mineras_ok = mineras_ok[mineras_ok.cuit.isna()==False].reset_index(drop=True)

mineras_ok = mineras_ok.append(a, ignore_index=True)


mineras_ok.to_excel('data/resultado_08-09-2021.xlsx', index=False)
