
import urllib.request, json, string 
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# Creating list of urls containing json files    
template = "https://raw.githubusercontent.com/fergiemcdowall/reuters-21578-json/master/data/full/reuters-0" 
url_list = [template + str(i).zfill(2) + ".json" for i in range(0,22) ]

# 1. Storing all json files in a list
def load_json(url):
    with urllib.request.urlopen(url) as url1:
        return json.loads(url1.read().decode())
    
DUMP = []    
for i in range(0, len(url_list)):
    DUMP.append(load_json(url_list[i]))
    
    
# 2. Removing entries that don't contain "body" nor "topics" keys
def filter_entries(list_of_dicts):
    return [entry for entry in list_of_dicts if ("body" in entry.keys()) and ("topics" in entry.keys())]
        
DUMP1 = []
for i in range(0,len(DUMP)):
    DUMP1.append(filter_entries(DUMP[i]))
    

# 3. Creating Train and Test sets
# Grabbing all body text data (X: Input/samples)
text = [entry["body"] for sublist in DUMP1 for entry in sublist]
# Grabbing all topics data (Y: Output/targets)
topics = [entry["topics"] for sublist in DUMP1 for entry in sublist]

# Assigning class labels: "earn": class1 = 1; not "earn": class2 = 0
y = []
for i in range(0,len(topics)):
    if "earn" in topics[i]:
        y.append(1)
    else:
        y.append(0)
 
  
# 4. Implementing own Feature Hashing function and creating Matrix

"""
Description: This function implements the feature hashing vectorization trick
INPUT: A document (type:string), N dimension size of document (vector) encoding (type:int)
OUTPUT: A numpy row array (dims: 1xN) which is the encoded document using hashing trick  
"""    
def feature_hashing(document, buckets):
    # removing punctuation from document
    document = document.translate(str.maketrans("","", string.punctuation))
    # splitting into lowercase words
    document = document.lower().split()
    doc_vector = np.zeros((buckets,), dtype=np.int)
    for word in document:
        doc_vector[hash(word) % buckets] += 1
        
    return doc_vector


# Creating Feature Hash Matrix (equivalent to BoW but with less feature dims)
buckets = 1000
FH = np.zeros((len(text), buckets), dtype=np.int)
for i in range(0, len(text)):
    FH[i,:] = feature_hashing(text[i], buckets).T    

    
# 5. Transforming data to BoW Matrix with CountVectorizer
#              and Evaluating with RandomForest as Pipeline
seed = 7
pipeline = Pipeline([
        ('vect', CountVectorizer(stop_words="english", lowercase=True)),
        ('clf', RandomForestClassifier(n_estimators=50))
        ])
    

# 6. Results: Accuracy of BoW and FH implementations
   
# n_splits=5 is equivalent to 80% train: 20% test, but repeated 5x
kfold = KFold(n_splits=5, random_state=seed)
BoW_results = cross_val_score(pipeline, text, y, cv=kfold)
FH_results = cross_val_score(RandomForestClassifier(n_estimators=50),
                             FH, y, cv=kfold)
print("Bag of Words Accuracy: ", BoW_results.mean()) # We obtain 96% accuracy : )
print("Feature Hashing Accuracy: ", FH_results.mean()) # We obtain 94.5% accuracy : ) 
    

    

    
    



    










