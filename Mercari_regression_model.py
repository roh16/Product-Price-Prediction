# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 22:13:33 2017

@author: rohangupta8
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.metrics import explained_variance_score
import numpy as np
import string
import collections
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
import multiprocessing as mp
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn   import metrics
from sklearn.cross_validation import train_test_split
import xgboost as xgb






df_train_raw=pd.read_table('./train.tsv','\t',engine='python')
df_train=df_train_raw.copy()
df_train.dtypes


df_test_raw=pd.read_table('./test.tsv','\t',engine='python')
df_test=df_test_raw.copy()

## check how the data looks like now we know the type
# total of 518313 unique items
df_train.name.nunique()
df_train[df_train['name'].isnull()].shape[0]
# Total of 1287 categories, 6327 items have missing categories
df_train.category_name.nunique()
df_train.category_name.value_counts()[1:21]
df_train[df_train['category_name'].isnull()].shape[0]
# Total of 4809 brands, 632682 items have missing brands
df_train.brand_name.nunique()
df_train[df_train['brand_name'].isnull()].shape[0]
# 311 records with zero price so should be rmoved
df_train.price.describe(percentiles=[0.01,0.05,0.25,0.5,0.75,0.95,0.99])
df_train[df_train['price']==0].shape[0]
# Almost equal distibution
df_train.shipping.value_counts()
df_train[df_train['shipping'].isnull()].shape[0]
# 1281426 unique item descriptions: This is text data and text analytics can be performed here.
df_train.item_description.nunique()
df_train[df_train['item_description'].isnull()].shape[0]
df_train[df_train['item_description']=='No description yet'].shape[0]


def create_histogram(data,x_label,y_label,title,bin_size=10):
    _,ax=plt.subplots(figsize=(10,10))
    ax.hist(data,bins=bin_size,color='#539caf',alhpa=0.9,edgecolor="black")
    ax.set_title(title,fontsize=14)
    ax.set_xlabel(x_label,fontsize=14)
    ax.set_ylabel(y_label,fontsize=14)
    
create_histogram(df_train[df_train['price']<=170]['price'],'bins','frequency','Price distribution',bin_size=10)
    
    

#==============================================================================
# Feature Engineering
#==============================================================================

#
df_train=df_train[~df_train['category_name'].isnull()]

df_train['main_category']=df_train['category_name'].map(lambda x: x.split('/')[0])
df_train['sub_category_1']=df_train['category_name'].map(lambda x: x.split('/')[1])
df_train['sub_category_2']=df_train['category_name'].map(lambda x: x.split('/')[2])

df_train['has_brand']=df_train['brand_name'].fillna(0)
df_train['has_brand']=df_train['has_brand'].map(lambda x: 1 if x!=0 else 0)

#port=PorterStemmer()
#category_words=[]
#for i in range(df_train.shape[0]):
#    temp_list=df_train.category_name.iloc[i].split('/')
#    temp_list=[port.stem(word) for word in temp_list]
#    category_words=category_words+temp_list
#
#
#count_categories=FreqDist(category_words)
#count_categories=dict(count_categories)

#==============================================================================
# getting dummies of the category and condition ID column
#==============================================================================

category_dummies=pd.get_dummies(df_train['main_category'],prefix='main_category')
df_train=pd.concat([df_train,category_dummies],axis=1)

#category_dummies_columns=category_dummies.columns.tolist()
#
#final_category_columns=[]
#for column in category_dummies_columns:
#    if category_dummies[column].sum()>=1000:
#        final_category_columns.append(column)
        



subcategory_1_dummy=pd.get_dummies(df_train['sub_category_1'],prefix='sub_category_1')
df_train=pd.concat([df_train,subcategory_1_dummy],axis=1)


condition_id_dummy=pd.get_dummies(df_train['item_condition_id'],prefix='item_condition_id')
df_train=pd.concat([df_train,condition_id_dummy],axis=1)
df_train.drop(['category_name','main_category','sub_category_1','sub_category_2','brand_name','item_condition_id'],inplace=True,axis=1)

df_train.index=df_train['train_id']

df_train.drop(['train_id','name','item_description'],inplace=True,axis=1)

df_test=df_test[~df_test['category_name'].isnull()]
category_dummies_test=pd.get_dummies(df_test['category_name'])

category_dummies_columns_test=category_dummies_test.columns.tolist()
final_category_columns_test=[]
for column in final_category_columns:
    if column in category_dummies_columns_test:
        final_category_columns_test.append(column)
        
        

df_test=pd.concat([df_test,category_dummies_test[final_category_columns_test]],axis=1)
df_test.drop(['category_name'],inplace=True,axis=1)

condition_id_dummy_test=pd.get_dummies(df_test['item_condition_id'])
df_test=pd.concat([df_test,condition_id_dummy_test],axis=1)
df_test.drop(['item_condition_id'],inplace=True,axis=1)

#==============================================================================
# Clustering the text description
#==============================================================================

 
 
def process_text(text, stem=True):
    """ Tokenize text and stem words removing punctuation """
    
    text = text.translate(str.maketrans('','',string.punctuation))
    tokens = word_tokenize(text)
 
    if stem:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]
 
    return tokens
 
 
def cluster_texts(texts, clusters=3):
    """ Transform texts to Tf-Idf coordinates and cluster texts using K-Means """
    vectorizer = TfidfVectorizer(tokenizer=process_text,
                                 stop_words=stopwords.words('english'),
                                 max_df=0.5,
                                 min_df=0.1,
                                 lowercase=True)
    
    tfidf_model = vectorizer.fit_transform(texts)
    km_model = KMeans(n_clusters=clusters)
    km_model.fit(tfidf_model)
 
    clustering = collections.defaultdict(list)
 
    for idx, label in enumerate(km_model.labels_):
        clustering[label].append(idx)
 
    return clustering

df_train['Cluster']=np.nan
def hash_cluster(train_id,clusters):
    
    for cluster in list(clusters.keys()):
        if train_id in clusters[cluster]:
            return(cluster)
          
df_train.loc[:,'Cluster']=df_train['train_id'].apply(lambda x:hash_cluster(x,clusters)) 
df_train=df_train[~df_train.item_description.isnull()]
articles = df_train.item_description.tolist()
clusters = cluster_texts(articles, 5)
pprint(dict(clusters))
#==============================================================================
# Fitting crude lasso
#==============================================================================
columns=df_train.columns.tolist()


lasso = Lasso(alpha=0.0010)
Y=df_train['price']
train_columns=df_train.columns.tolist()
train_columns.remove('price')
X=df_train[train_columns]
pred_train, pred_test, tar_train, tar_test = train_test_split(X, Y,test_size=.25, random_state=123)
res = lasso.fit(pred_train,tar_train)
final_model_coeff={}
for col,val in zip(train_columns,res.coef_.tolist()):
    if val!=0.0:
        final_model_coeff[col]=val


    
y_predicted=lasso.predict(pred_test)

columns_test=df_test.columns.tolist()

X_test=df_test[[columns_test[3]]+columns_test[5:]].values
y_predicted_test=lasso.predict(X_test)

explained_variance_score(tar_test, y_predicted)
np.sqrt(metrics.mean_squared_error(tar_test,y_predicted))

#==============================================================================
# Submission
#==============================================================================

submission_file=pd.read_csv('./sample_submission.csv')



pool=mp.Pool()



##Random Forests

print("Random Forests All Features\n\n")

rf_regressor = RandomForestRegressor(n_estimators=10,oob_score=True)

kf = KFold(5, random_state=40)    
oos_y = []
oos_pred = []
fold = 0

for training, test in kf.split(df_train):

    fold+=1
    x_train_fold = df_train[train_columns].ix[training]
    y_train_fold = df_train['price'].ix[training]
    x_test_fold = df_train[train_columns].ix[test]
    y_test_fold = df_train['price'].ix[test]
    
    rf_regressor.fit(x_train_fold, y_train_fold)
    pred = rf_regressor.predict(x_test_fold)
    oos_y.append(y_test_fold)
    oos_pred.append(pred)        

# Build the oos prediction list and calculate the error.
oos_y = np.concatenate(oos_y)
oos_pred = np.concatenate(oos_pred)
score = np.sqrt(metrics.mean_squared_error(oos_y,oos_pred))
print("\n Average RMSE : {}".format(score))   


rf_regressor.fit(pred_train,tar_train)
y_predicted_test_rf = rf_regressor.predict(pred_test)

np.sqrt(metrics.mean_squared_error(tar_test,y_predicted_test_rf))



top_k=10
importances=rf_regressor.feature_importances_
indices = np.argsort(importances)
indices=indices[::-1]
importance_list=list(importances)
new_indices = indices[:top_k][::-1]
seleted_importance_list=[importance_list[i] for i in new_indices]
selected_columns=[train_columns[i] for i in new_indices]

plt.title('Feature Importances')
plt.barh(range(len(new_indices)), seleted_importance_list, color='#539caf', align='center')
plt.yticks(range(len(new_indices)),selected_columns) ## removed [indices]
plt.xlabel('Relative Importance')
plt.show()

#==============================================================================
# Applying xgboost
#==============================================================================

gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(pred_train, tar_train)
y_predicted_xgb=gbm(pred_test)




