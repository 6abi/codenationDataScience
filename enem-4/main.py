#!/usr/bin/env python
# coding: utf-8

# In[65]:


import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import pylab
plt.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


# In[141]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# dataframe de resposta para envio
answer = pd.DataFrame()


# In[142]:



#número de inscrição
answer['NU_INSCRICAO'] = test['NU_INSCRICAO']
ans = answer['NU_INSCRICAO']


# In[143]:



train.drop(['NU_INSCRICAO'], axis=1, inplace=True)
test.drop(['NU_INSCRICAO'],axis=1, inplace=True)


# In[144]:


numeric_features = train.select_dtypes(include=[np.number])
corr = numeric_features.corr()
print (corr['IN_TREINEIRO'].sort_values(ascending=False)[:10], '\n')


# In[145]:



train.shape, test.shape


# In[146]:



test.head()


# In[147]:


train = train[['NU_IDADE','TP_ST_CONCLUSAO', 'TP_PRESENCA_CN', 'TP_PRESENCA_CH', 'TP_PRESENCA_LC','TP_PRESENCA_MT', 'IN_TREINEIRO']]
test = test[['NU_IDADE','TP_ST_CONCLUSAO', 'TP_PRESENCA_CN', 'TP_PRESENCA_CH', 'TP_PRESENCA_LC','TP_PRESENCA_MT']]


# In[148]:


train.info()


# In[149]:


train['IN_TREINEIRO'].value_counts()
ax = sns.countplot(x='IN_TREINEIRO', data=train)
plt.ylabel('Qtd')
plt.title('Distribuição');


# In[150]:


from imblearn.over_sampling import SMOTE


# In[151]:


smt = SMOTE()
target = train['IN_TREINEIRO']
type(target)


# In[152]:


train, target = smt.fit_sample(train, target)
np.bincount(target)
ax = sns.countplot(x=target)
plt.ylabel('Quantidade')
plt.title('Distribuição das classes');


# In[153]:



features = ['TP_ST_CONCLUSAO', 'TP_PRESENCA_CN', 'TP_PRESENCA_CH', 'TP_PRESENCA_LC',
                 'TP_PRESENCA_MT']
features_corr = ['IN_TREINEIRO', 'TP_ST_CONCLUSAO', 'TP_PRESENCA_CN', 'TP_PRESENCA_CH', 'TP_PRESENCA_LC',
                 'TP_PRESENCA_MT']
df_train = train[features_corr]
df_test = test[features]


# In[154]:


df_train.info()


# In[155]:


y_train = df_train[['IN_TREINEIRO']]
df_train.drop('IN_TREINEIRO', axis=1, inplace=True)
x_train = df_train
x_test = df_test
x_train = sc.fit_transform(x_train)  
x_test = sc.transform(x_test)


# In[ ]:





# In[156]:


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor( 
           criterion='mae', 
           max_depth=8,
           max_leaf_nodes=None,
           min_impurity_split=None,
           min_samples_leaf=1,
           min_samples_split=2,
           min_weight_fraction_leaf=0.0,
           n_estimators= 500,
           n_jobs=-1,
           random_state=0,
           verbose=0,
           warm_start=False
)


# In[157]:


regressor.fit(x_train, y_train)


# In[158]:


x_test = df_test[features] 
x_test = sc.transform(x_test)


# In[159]:



y_pred_test = regressor.predict(x_test)
y_pred_train = regressor.predict(x_train)


# In[160]:


y_pred_test


# In[161]:


print('MAE:', metrics.mean_absolute_error(y_train, y_pred_train).round(8)  )
print('MSE:', metrics.mean_squared_error(y_train, y_pred_train).round(8) )  
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, y_pred_train)).round(8))


# In[162]:


y_pred_test


# In[163]:


answer['NU_INSCRICAO'] = ans
answer['IN_TREINEIRO'] = y_pred_test.astype(int)


# In[164]:


answer.sample(20)


# In[165]:


answer.to_csv('answer.csv', index=False, header=True)


# In[ ]:




