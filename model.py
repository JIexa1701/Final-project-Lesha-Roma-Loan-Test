#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('bmh')

from catboost import CatBoostClassifier

from sklearn.metrics import classification_report

import mlxtend
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_score

import pickle


# In[2]:


# loading data to df

# df = pd.read_csv(r"C:\Users\User\Desktop\Phyton\Data Science\loan\loan.csv")
df = pd.read_csv('loan.csv')
df.head()


# In[25]:


df.shape


# In[3]:


df.replace(to_replace=['Current', 'Fully Paid', 'Issued', 'Charged Off', 'Late (31-120 days)','In Grace Period','Late (16-30 days)','Does not meet the credit policy. Status:Fully Paid','Default','Does not meet the credit policy. Status:Charged Off'],
           value= ['0','0','0','1','1','1','1','1','1','1'], 
           inplace=True)


# In[4]:


df['loan_status'] = df['loan_status'].fillna(0.0).astype(int)


# # Handling NaN

# In[5]:


# увеличивает число строк отображаемых в ноутбуке, не будет срезаться

pd.options.display.max_rows = 4000
df.isnull().sum()


# In[6]:


# удаляем фичи где кол-во nan > 250 000

for column in df:
    if sum(df[column].isnull()) >= 250000:
        df.drop(column, axis=1, inplace=True)


# In[7]:


pd.options.display.max_rows = 4000
df.isnull().sum()


# In[8]:


# ф-ция которая меняет порядок колонок

def movecol(df, cols_to_move=[], ref_col='', place='After'):
    
    cols = df.columns.tolist()
    if place == 'After':
        seg1 = cols[:list(cols).index(ref_col) + 1]
        seg2 = cols_to_move
    if place == 'Before':
        seg1 = cols[:list(cols).index(ref_col)]
        seg2 = cols_to_move + [ref_col]
    
    seg1 = [i for i in seg1 if i not in seg2]
    seg3 = [i for i in cols if i not in seg1 + seg2]
    
    return(df[seg1 + seg2 + seg3])


# In[9]:


# передвигаем наш таргет loan_status в самый конец

df = movecol(df, 
             cols_to_move=['loan_status'], 
             ref_col='total_rev_hi_lim',
             place='After')
df.head(1)


# # Choosing features

# In[10]:


columns_to_keep = ['int_rate', 'installment', 'total_rec_prncp', 
                   'total_rec_late_fee',  'tot_cur_bal', 'total_rev_hi_lim', 'recoveries', 
                   'out_prncp_inv', 'inq_last_6mths', 'sub_grade']

len(columns_to_keep)


# In[11]:


df = df[columns_to_keep + ['loan_status']]

df.head(1)


# In[13]:


# ячейка нужна только если добавляем sub_grade в модель

le = LabelEncoder()
df['sub_grade'] = le.fit_transform(df['sub_grade'])
df.head(1)


# # CatBoost Model class 1 weighted

# In[14]:


#train/test/validation split

X = df.drop('loan_status', axis=1)
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=1)


# In[17]:


# {'depth': 7, 'iterations': 200, 'learning_rate': 0.1, 'scale_pos_weight': 4}

model_cat = CatBoostClassifier(iterations=200, depth=7, learning_rate=0.1, scale_pos_weight=4)

model_cat.fit(X_train, y_train)


# In[18]:


# test_y_pred4 = model_cat.predict(X_test)


# In[20]:


# make predictions on a test set and get AUC score

# print("Sklearn CatBoost classifier:")
# y_pred = model_cat.predict_proba(X_test)

# print(f" - roc_auc_score: {roc_auc_score(y_test, y_pred[:,1]): .5f}")

# Accuracy score, f1_score

# print(f" - accuracy_score: {accuracy_score(y_test, test_y_pred4): .5f}")
# print(f" - f1_score: {f1_score(y_test, test_y_pred4): .5f}")


# In[21]:


# fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_pred[:,1], pos_label=1)

# roc_auc = auc(fpr, tpr)

# ROC Curve Visualization

#plt.step(fpr, tpr)
#plt.title('ROC Curve')
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.show()


# # Plot confusion matrix and classification_report

# In[23]:


#print(classification_report(y_test, test_y_pred4))


# In[24]:


#CM = confusion_matrix(y_test, test_y_pred4)

#fig, ax = plot_confusion_matrix(conf_mat=CM ,  figsize=(5, 5))


# # Catboost Validation test

# In[26]:


#test_y_pred_val = model_cat.predict(X_val)


# In[27]:


# make predictions on a test set and get AUC score

#print("Sklearn CatBoost classifier on val test:")
#y_pred_val = model_cat.predict_proba(X_val)

#print(f" - roc_auc_score: {roc_auc_score(y_val, y_pred_val[:,1]): .5f}")

# roc_auc_score(y_test, y_pred[:,1])

# Accuracy score, f1_score

#print(f" - accuracy_score: {accuracy_score(y_val, test_y_pred_val): .5f}")
#print(f" - f1_score: {f1_score(y_val, test_y_pred_val): .5f}")


# # Saving model to disk

# In[31]:


# Saving model to disk

pickle.dump(model_cat, open('model.pkl','wb'))


# In[ ]:




