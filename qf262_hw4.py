import sys
import re
import math
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer 

from numpy import dot
from numpy.linalg import norm

the_lemma = WordNetLemmatizer()
the_stemmer = nltk.PorterStemmer()

closed_class_stop_words = ['a','the','an','and','or','but','about','above','after','along','amid','among',\
                           'as','at','by','for','from','in','into','like','minus','near','of','off','on',\
                           'onto','out','over','past','per','plus','since','till','to','under','until','up',\
                           'via','vs','with','that','can','cannot','could','may','might','must',\
                           'need','ought','shall','should','will','would','have','had','has','having','be',\
                           'is','am','are','was','were','being','been','get','gets','got','gotten',\
                           'getting','seem','seeming','seems','seemed',\
                           'enough', 'both', 'all', 'your' 'those', 'this', 'these', \
                           'their', 'the', 'that', 'some', 'our', 'no', 'neither', 'my',\
                           'its', 'his' 'her', 'every', 'either', 'each', 'any', 'another',\
                           'an', 'a', 'just', 'mere', 'such', 'merely' 'right', 'no', 'not',\
                           'only', 'sheer', 'even', 'especially', 'namely', 'as', 'more',\
                           'most', 'less' 'least', 'so', 'enough', 'too', 'pretty', 'quite',\
                           'rather', 'somewhat', 'sufficiently' 'same', 'different', 'such',\
                           'when', 'why', 'where', 'how', 'what', 'who', 'whom', 'which',\
                           'whether', 'why', 'whose', 'if', 'anybody', 'anyone', 'anyplace', \
                           'anything', 'anytime' 'anywhere', 'everybody', 'everyday',\
                           'everyone', 'everyplace', 'everything' 'everywhere', 'whatever',\
                           'whenever', 'whereever', 'whichever', 'whoever', 'whomever' 'he',\
                           'him', 'his', 'her', 'she', 'it', 'they', 'them', 'its', 'their','theirs',\
                           'you','your','yours','me','my','mine','I','we','us','much','and/or'
                           ]

def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3

f = open('cran.all.1400', 'r')
record = False
abs_tmp = []
for line in f:
    line = line.strip()
    
    if re.match(r'(\.[I]) ([0-9]+)', line):
       # num = re.match(r'(\.[I]) ([0-9]+)', line).group(2)
       # abs_tmp.append(num)
        record = False
        
    if record == True:
        abs_tmp.append(line)
        
    if line == ".W":
        record = True 
        abs_tmp.append("@@@")

f.close()

abs_tmp = ' '.join(abs_tmp)

abstract = abs_tmp.split("@@@")
abstract.remove(abstract[0])
abstract[575:577] = [''.join(abstract[575:577])]
abstract[577:579] = [''.join(abstract[577:579])]


#remove stop words, punctuation and numbers, lemmatize all the words 
abs_sent = []
abs_wrd = []
for abs_word in abstract:
    abs_word = re.sub('[^A-z]+', ' ',abs_word)
    abs_lemma = [the_lemma.lemmatize(word) for word in abs_word.strip().split() \
                 if (word not in closed_class_stop_words) and (len(word) != 1)]
    abs_wrd.append(abs_lemma)
    abs_lemma = ' '.join(abs_lemma)
    abs_sent.append(abs_lemma)

#construct abs_vec
num_doc_cont2 = 0
abs_vec = []
for abswd in abs_sent:
    abswd_ = abswd.split()
    vec_tmp = {}
    length = len(abswd_)
    for wd in abswd_:
        num_ins = abswd.count(wd)
        for sent in abs_sent:
            if (sent.find(wd) != -1):
                num_doc_cont2 += 1
        idf = math.log(1400 / num_doc_cont2)
        tf = math.log (num_ins / length)
        #tf = num_ins
        tf_idf = idf * tf
        vec_tmp[wd] = tf_idf
        num_doc_cont2 = 0
        
    abs_vec.append(vec_tmp)

#read 225 queries 
f = open('cran.qry', 'r')
qry_tmp = []
qry_final =[]
for line in f:
    res = line.strip()
    qry_tmp.append(res)

qry_tmp = ' '.join(qry_tmp).split(".W")
qry_tmp = ' '.join(qry_tmp)
qry_tmp = re.sub('\.[I] \d{3}','', qry_tmp).split("   ")
for qry in qry_tmp:
    if (len(qry) != 0):
        qry_final.append(qry.strip())
        
f.close()

#remove stop words, punctuation and numbers, lemmatize all the words 
qry_sent = []
qry_wrd = []
for qry_word in qry_final:
    qry_word = re.sub('[^A-z]+', ' ',qry_word)
    text_lemma = [the_lemma.lemmatize(word) for word in qry_word.strip().split() if word not in closed_class_stop_words]
    qry_wrd.append(text_lemma)
    text_lemma = ' '.join(text_lemma)
    qry_sent.append(text_lemma)

#construct qry_vec
num_doc_cont = 0
qry_vec = []
for qry in qry_wrd:
    vec_tmp = []
    length = len(qry)
    for wd in qry:
        num_ins = (' '.join(qry)).count(wd)
        for sent in abs_sent:
            if (sent.find(wd) != -1):
                num_doc_cont += 1
        if num_doc_cont == 0:
            idf = 0
        else:
            idf = math.log(225 / num_doc_cont)
        tf = math.log(num_ins / length)
        tf_idf = idf * tf
        vec_tmp.append(tf_idf)
        num_doc_cont = 0
        
    qry_vec.append(vec_tmp)

#calculate cosine similarity
cos_sim = []
for i in range(225):
    cos_sim_tmp = {}
    for j in range(1400):
        qry_abs_intersect = intersection(qry_wrd[i], abs_wrd[j])
        score_tmp = []
        for word in qry_wrd[i]:
            if word in qry_abs_intersect:
                score = abs_vec[j][word]
            else:
                score = 0
            score_tmp.append(score)
        
        
        if not np.any(score_tmp): #all zero score list 
            cos_sim_val = 0
        else:
            cos_sim_val = dot(score_tmp, qry_vec[i]) / (norm(score_tmp) * norm(qry_vec[i]))
            
        cos_sim_tmp[str(i+1)+" "+str(j+1)] = cos_sim_val
    
    cos_sim_tmp_sorted = sorted(cos_sim_tmp.items(), key=lambda d:d[1], reverse = True)
    for key, value in cos_sim_tmp_sorted:
        if value > 0.28:
            cos_sim.append(str(key) + " " + str(value))


#write to output.txt
write_to_file = open("output.txt","w")
for i in range(len(cos_sim)):
    write_to_file.write(cos_sim[i] + "\n")
            
write_to_file.close()