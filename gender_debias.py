##########################################################Debias for gender_twitter.csv##########################################################################3
import pandas as pd
import numpy as np
import gensim
import re
import json
import nltk
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.tokenize import WordPunctTokenizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from google.colab import drive
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from nltk.corpus import stopwords
drive.mount('/content/drive')

###csv work (x and y)
d = pd.read_csv("/content/drive/My Drive/Colab Notebooks/gender-classifier-DFE-791531.csv",encoding="latin1")
df = pd.DataFrame(data=d)
df = df.fillna(method='ffill')
df=df[:30000]
x = df['text'].to_numpy()
df.gender = [1 if each =="female" else 0 for each in df.gender]
y = df['gender'].to_numpy()

###files to neutralize and equalize gender words#######
with open('/content/drive/My Drive/definitional_pairs.json') as f:
    defin_pairs = json.load(f)
    
with open('/content/drive/My Drive/equalize_pairs.json') as f:
    equal_pairs = json.load(f)

with open('/content/drive/My Drive/gender_specific_seed.json') as f:
    gender_words = json.load(f)
    
with open('/content/drive/My Drive/professions.json') as f:
        professions = json.load(f)
        professions = [p[0] for p in professions]
        professions = professions+ ["she", "he"]
professions = [re.sub('_', "-", p) for p in professions]

##############Required functions###########################################

########finding the embedding(vector) of the sentence
def sentence_embedding(sentence,embeddings):
    vector = np.zeros([model.vector_size], dtype='float32')
    sentence = sentence.lower()
    sentence_tokens = tokenizer.tokenize(sentence)
    divisor = 0
    for word in sentence_tokens:
        if word in embeddings:
            
            divisor += 1
            vector = vector + embeddings[word]
    
    if divisor != 0: vector /= divisor 
    return vector

##########finding pca components################
def gender_subspace(pairs, embedding, num_components=10):
    mat = []
    for x, y in pairs:
        if (x in embedding and y in embedding):
            mid = (embedding[x] + embedding[y])/2 
            mat.append(embedding[y] - mid)
            mat.append(embedding[y] - mid)
    mat = np.array(mat)
    a = PCA(n_components = num_components)
    a.fit(mat)
    return a.components_

# Normalization of a word
def normalize(embedding):
    for w in embedding:
        embedding[w] /= np.linalg.norm(embedding[w])

# Projection subspace of a word
def projection_subspace(subspace, w):
    t = subspace.dot(w)
    t = np.expand_dims(t, -1)
    return np.sum(t * subspace, axis=0)

#debias the embedding
def debias(embedding, gender_words, defin_pairs, equal_pairs):
    
    new_embedding = copy.deepcopy(embedding)
    normalize(new_embedding)
    genderSubspace = gender_subspace(defin_pairs, new_embedding)
    project = dict()
  
    gender_set = set(gender_words)
    for w in embedding:
        projectionSubspace = projection_subspace(genderSubspace, new_embedding[w])
        project[w] = projectionSubspace

        #word not in gender word subract the bias (neutralization)
        if w not in gender_set:           
            new_embedding[w] -= projectionSubspace
    normalize(new_embedding)
    
    # lower casing equal_pairs
    eq_pairs = [(x.lower(), y.lower()) for (x, y) in equal_pairs if x in new_embedding and y in new_embedding]
    
    # Equalization
    for (x, y) in eq_pairs:
        mean = (new_embedding[x] + new_embedding[y]) / 2
        meanProjection = projection_subspace(genderSubspace, mean)
        val = mean - meanProjection
        factorx = (project[x] - meanProjection) / np.linalg.norm(project[x] - meanProjection)
        new_embedding[x] = val + np.sqrt(1 - np.linalg.norm(val)**2) * factorx
        factory = (project[y] - meanProjection) / np.linalg.norm(project[y] - meanProjection)
        new_embedding[y] = val + np.sqrt(1 - np.linalg.norm(val)**2) * factory      
    normalize(new_embedding)
    return new_embedding


###splitting
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=42)

#####vocabulary sentences
description_list=[]
lemma = nltk.WordNetLemmatizer()
for description in x:
    description = re.sub("[^a-zA-z]"," ",description)
    description = description.lower()
    description = nltk.word_tokenize(description)
    description = [lemma.lemmatize(word) for word in description if word not in stopwords.words('english')]
    description_list.append(description)


tokenizer = WordPunctTokenizer()
model = Word2Vec(sentences=description_list, size=100, window=5, min_count=1, workers=4)
model.save("word2vec.model")
embeddings=dict()

#vocab to dictionary
for idx, key in enumerate(model.wv.vocab):
    embeddings[key] = model.wv.get_vector(key)


vector_matrix_x_train =[]
for i in x_train:
  vector_matrix_x_train.append(sentence_embedding(i, embeddings))
vector_matrix_x_test = []
for i in x_test:
  vector_matrix_x_test.append(sentence_embedding(i, embeddings))

######Random Forest classifier ###########
rfc = RandomForestClassifier(n_estimators= 90, min_samples_split= 2, min_samples_leaf= 2, max_features='auto', max_depth=None, bootstrap= True)
rfc.fit(vector_matrix_x_train,y_train)
y_pred_rfc1 = rfc.predict(vector_matrix_x_test)
print("RandomForest classifier accuracy training before bias: ",rfc.score(vector_matrix_x_train,y_train))
print("RandomForest classifier accuracy testing before bias: ",rfc.score(vector_matrix_x_test,y_test))
plot_confusion_matrix(rfc, vector_matrix_x_test,y_test) 
plt.show()
########SVM classifier################
clf_svc = SVC().fit(vector_matrix_x_train,y_train)
prediction_svc=clf_svc.predict(vector_matrix_x_test)
print("SVM classifier accuracy training before bias: ",clf_svc.score(vector_matrix_x_train,y_train))
print("SVM classifier accuracy testing before bias: ",clf_svc.score(vector_matrix_x_test,y_test))

#########voting classifier ##############
predictions_voting=[]
for i in range(len(y_pred_rfc1)):
    a=[y_pred_rfc1[i],prediction_svc[i]]
    predictions_voting.append(max(set(a), key=a.count))
print(" Voting classifier testing before-> ",accuracy_score(predictions_voting, y_test)*100)

############################debiasing embeddings#################################################
debiased_embeddings = debias(embeddings, gender_words, defin_pairs, equal_pairs)

vector_matrix_x_train =[]
for i in x_train:
  vector_matrix_x_train.append(sentence_embedding(i, debiased_embeddings))
vector_matrix_x_test = []
for i in x_test:
  vector_matrix_x_test.append(sentence_embedding(i, debiased_embeddings))

######Random Forest classifier ###########
rfc = RandomForestClassifier(n_estimators= 90, min_samples_split= 2, min_samples_leaf= 2, max_features='auto', max_depth=None, bootstrap= True)
rfc.fit(vector_matrix_x_train,y_train)
y_pred_rfc2 = rfc.predict(vector_matrix_x_test)
print("RandomForest classifier accuracy training after bias: ",rfc.score(vector_matrix_x_train,y_train))
print("RandomForest classifier accuracy testing after bias: ",rfc.score(vector_matrix_x_test,y_test))
plot_confusion_matrix(rfc, vector_matrix_x_test,y_test) 
plt.show()
########SVM classifier################
clf_svc = SVC().fit(vector_matrix_x_train,y_train)
prediction_svc=clf_svc.predict(vector_matrix_x_test)
print("SVM classifier accuracy training after bias: ",clf_svc.score(vector_matrix_x_train,y_train))
print("SVM classifier accuracy testing after bias: ",clf_svc.score(vector_matrix_x_test,y_test))

#########voting classifier ##############
predictions_voting=[]
for i in range(len(y_pred_rfc2)):
    a=[y_pred_rfc2[i],prediction_svc[i]]
    predictions_voting.append(max(set(a), key=a.count))
print(" Voting classifier testing after-> ",accuracy_score(predictions_voting, y_test)*100)
