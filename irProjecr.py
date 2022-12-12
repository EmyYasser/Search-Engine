import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from natsort import natsorted
import pandas as pd #part3
import numpy as np  #part3
import math


stopWords= set(stopwords.words("english")) 
# print(stopWords)
stopWords.remove('in')
stopWords.remove('to')
stopWords.remove('where')

myFiles= natsorted(os.listdir('files'))
# print(myFiles)



DocTerms=[]
for printfiles in myFiles:
    with open(f'files/{printfiles}',"r") as fn:
        doc=fn.readline()
        words= word_tokenize(doc)
        terms=[]
        for word in words:
             if word not in stopWords:
                terms.append(word)
        DocTerms.append(terms)
        # print(DocTerms)  #true

        # print(dict.fromkeys(terms,0))
# //////////////////////////////////
# part3
allWords=[]
for docs in DocTerms:
    for word in docs:
        allWords.append(word)

# print(dict.fromkeys(allWords,0))

def getTerm_freq(docs):  # fun to check all words
    found_Words= dict.fromkeys(allWords,0)
    for word in docs:
        found_Words[word]+=1
    return found_Words

term_freq =pd.DataFrame(getTerm_freq(DocTerms[0]).values(),index=getTerm_freq(DocTerms[0]).keys())

for i in range(1,len(DocTerms)):
    term_freq[i]=getTerm_freq(DocTerms[i]).values()

term_freq.columns=['doc'+str(i)for i in range (1,11)]
print("*********************** Term Frequency(TF)***********************")
print(term_freq) # point 1.1

def getWeight_term(x):
    if x> 0:
        return math.log(x)+1 # Weighted Low
    return 0 

for i in range(1,len(DocTerms)+1):
    term_freq['doc'+str(i)]=term_freq['doc'+str(i)].apply(getWeight_term)
print("*********************** w tf(1+ log tf)***********************")
print(term_freq) #true point 1.2

#///////////////////////////////////////

tfd = pd.DataFrame(columns=['freq','idf'])

for i in range(len(term_freq)):
    freq = term_freq.iloc[i].values.sum()   #  (iloc) => pos term  in data frame

    tfd.loc[i,'freq']= freq

    tfd.loc[i,'idf'] = math.log10(10/ float(freq))
tfd.index=term_freq.index
print(tfd) #true point 2.1
term_freq_inverse= term_freq.multiply(tfd['idf'],axis=0)
print(term_freq_inverse) #true point 2.2

# print("*********************** tf*idf ***********************")

doc_length= pd.DataFrame() # sqrt of 

def getDoc_length(columon1):
    return np.sqrt(term_freq_inverse[columon1].apply(lambda x: x**2).sum())


for column in term_freq_inverse.columns:
    doc_length.loc[0, column+'_length'] = getDoc_length(column)
print("*********************** doc_length ***********************")
print(doc_length) ##  true point 3.1  

# *********************** Normalized tf.idf *********************** 
# point 3.2 

normalized_term_freq = pd.DataFrame()
# عشان تلف عليهم كلهم fun هنعمل
def get_normalized (columon1,x):
    try:
        return x / doc_length[columon1+'_length'].values[0]
    except:
        return 0

for column in term_freq_inverse.columns:
    normalized_term_freq[column]=term_freq_inverse[column].apply(lambda x : get_normalized(column ,x))

print("*********************** Normalized tf.idf ***********************")
print(normalized_term_freq)

# *********************** Cosine similaitry  ***********************

# //////////////////////////////////

# *********************** positional_index  ***********************
document_number =0
positional_index={}

for document in DocTerms:

    for positional ,term in enumerate(document):

        if term in positional_index:

            positional_index[term][0]=positional_index[term][0]+1

            if document_number in positional_index[term][1]:
                positional_index[term][1][document_number].append(positional)

            else:
                positional_index[term][1][document_number] = [positional]

        else:
                positional_index[term]=[]

                positional_index[term].append(1)

                positional_index[term].append({})

                positional_index[term][1][document_number]=[positional]

    document_number+=1
# print(positional_index)


#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

query='fools fear '
final_list =[[]for i in range(10)]

for word in query.split():
    for key in positional_index[word][1].keys():

        if final_list[key-1]!=[]:
            if final_list[key-1][-1]==positional_index[word][1][key][0]-1:
                final_list[key-1].append(positional_index[word][1][key][0])
        else:
            final_list[key-1].append(positional_index[word][1][key][0])
print(final_list)
for position,list in enumerate(final_list,start=1):
    # print(position,list)
    if len(list)== len(query.split()):
        print(position)


















# myFile = open("7.txt","r")
# print(myFile.read())
# myFile.close() 
# # print ([word for word in words if word not in stopWords and not 'to'])

# for printfiles in myFiles:
#     with open(f'files/{printfiles}',"r") as fn:
#         doc=fn.readline()
#         print(doc)
#         tokens=word_tokenize(doc)
        

# for printfiles in myFiles:
#     with open(f'files/{printfiles}',"r") as fn:
#         doc=fn.readline()
#         print(word_tokenize(doc))