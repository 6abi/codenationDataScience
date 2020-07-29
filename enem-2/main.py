#!/usr/bin/env python
# coding: utf-8

# In[131]:



import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


# In[132]:



train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[133]:


numeric_features = train.select_dtypes(include=[np.number])
corr = numeric_features.corr()
print (corr['NU_NOTA_MT'].sort_values(ascending=False)[:10], '\n')
print (corr['NU_NOTA_MT'].sort_values(ascending=False)[-10:])


# In[134]:


features = ['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_REDACAO',
            'NU_NOTA_COMP1', 'NU_NOTA_COMP2','NU_NOTA_COMP3', 'NU_NOTA_COMP4', 'NU_NOTA_COMP5']
features_corr = ['NU_NOTA_MT', 'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_REDACAO',
                 'NU_NOTA_COMP1', 'NU_NOTA_COMP2','NU_NOTA_COMP3', 'NU_NOTA_COMP4', 'NU_NOTA_COMP5']
df_train = train[features_corr]
df_test = test[features]


# In[135]:


df_train.shape


# In[136]:



# Viewing training data:
train.head()


# In[137]:



df_train.isnull().sum()


# In[138]:


df_test.isnull().sum()


# In[139]:



df_train.fillna(0,inplace=True)


# In[140]:


df_test.fillna(0,inplace=True)


# In[141]:


y_train = df_train['NU_NOTA_MT']
df_train.drop('NU_NOTA_MT', axis=1, inplace=True)
x_train = df_train
x_test = df_test[features]
x_train = sc.fit_transform(x_train)  
x_test = sc.transform(x_test)


# In[142]:


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


# In[143]:



regressor.fit(x_train, y_train)


# In[144]:



x_test = df_test[features] 
x_test = sc.transform(x_test)


# In[145]:



y_pred_test = regressor.predict(x_test)
y_pred_train = regressor.predict(x_train)


# In[146]:



print('MAE:', metrics.mean_absolute_error(y_train, y_pred_train).round(8)  )
print('MSE:', metrics.mean_squared_error(y_train, y_pred_train).round(8) )  
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, y_pred_train)).round(8))


# In[147]:


y_pred_test


# In[148]:


submission = pd.DataFrame()
submission['NU_INSCRICAO'] = test.NU_INSCRICAO
submission['NU_NOTA_MT'] = y_pred_test.round(1)


# In[161]:





# In[150]:





# In[151]:





# In[152]:





# In[153]:





# In[154]:





# In[155]:





# In[156]:





# In[157]:





# In[158]:





# In[159]:





# In[160]:





# In[ ]:





# In[ ]:




