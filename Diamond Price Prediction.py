#!/usr/bin/env python
# coding: utf-8

# # Diamond Price Prediction

# The aim of this analysis is to predict the price of diamonds based on their characteristics. The dataset used for this analysis is the Diamonds dataset from Kaggle. The dataset contains 53940 observations and 10 variables. The variables are as follows:
# 
# 
# |Column Name|Description|
# |-----------|-----------|
# |carat|Weight of the diamond|
# |cut|Quality of the cut (Fair, Good, Very Good, Premium, Ideal)|
# |color|Diamond colour, from J (worst) to D (best)|
# |clarity|How clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))|
# |x|Length in mm|
# |y|Width in mm|
# |z|Depth in mm|
# |depth|Total depth percentage = z / mean(x, y) = 2 * z / (x + y) (43--79)|
# |table|Width of top of diamond relative to widest point (43--95)|
# |price|Price in US dollars (326--18,823)|
# 

# In[1]:


#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#loading the dataset
df = pd.read_csv('diamonds.csv')
df.head()


# ## Data Preprocessing

# In[3]:


df.shape


# In[4]:


#checking for null values
df.info()


# In[5]:


#checking descriptive statistics
df.describe()


# In[6]:


#values count of categorical variables
print(df.cut.value_counts(),'\n',df.color.value_counts(),'\n',df.clarity.value_counts())


# In[7]:


df.head(10)


# ## Exploratory Data Analysis

# In[8]:


sns.histplot(df['price'],bins = 20)


# In[9]:


sns.histplot(df['carat'],bins=20)


# Most of the diamonds are less then 1 carat in weight.

# In[10]:


plt.figure(figsize=(5,5))
plt.pie(df['cut'].value_counts(),labels=['Ideal','Premium','Very Good','Good','Fair'],autopct='%1.1f%%')
plt.title('Cut')
plt.show()


# In[11]:


plt.figure(figsize=(5,5))
plt.bar(df['color'].value_counts().index,df['color'].value_counts())
plt.ylabel("Number of Diamonds")
plt.xlabel("Color")
plt.show()


# In[12]:


plt.figure(figsize=(5,5))
plt.bar(df['clarity'].value_counts().index,df['clarity'].value_counts())
plt.title('Clarity')
plt.ylabel("Number of Diamonds")
plt.xlabel("Clarity")
plt.show()


# In[13]:


sns.histplot(df['table'],bins=10)
plt.title('Table')
plt.show()


# ### Comparing Diamond's features with Price

# In[14]:


sns.barplot(x='cut',y='price',data=df)


# In[15]:


sns.barplot(x='color',y='price',data=df)
plt.title('Price vs Color')
plt.show()


# In[16]:


sns.barplot(x = 'clarity', y = 'price', data = df)


# J color and I1 clarity are worst featiures for a diamond, however when the data is plotted on bar graph, it is seen that the price of diamonds with J color and I1 clarity is higher than the price of diamonds with D color and IF clarity, which is opposite to what I expected.

# ## Data Preprocessing 2

# In[17]:


#changing categorical variables to numerical variables
df['cut'] = df['cut'].map({'Ideal':5,'Premium':4,'Very Good':3,'Good':2,'Fair':1})
df['color'] = df['color'].map({'D':7,'E':6,'F':5,'G':4,'H':3,'I':2,'J':1})
df['clarity'] = df['clarity'].map({'IF':8,'VVS1':7,'VVS2':6,'VS1':5,'VS2':4,'SI1':3,'SI2':2,'I1':1})


# ## Coorelation

# In[18]:


#coorelation matrix
df.corr()


# In[19]:


#plotting the correlation heatmap
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True,cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# #### Ploting the relationship between Price and Carat

# In[20]:


sns.lineplot(x='carat',y='price',data=df)
plt.title('Carat vs Price')
plt.show()


# From the lineplot it is quite clear that the price of the diamond increases with the increase in the carat of the diamond. However, diamonds with less carat also have high price. This is because of the other factors that affect the price of the diamond.

# In[21]:


fig, ax = plt.subplots(2,3,figsize=(15,5))
sns.scatterplot(x='x',y='carat',data=df, ax=ax[0,0])
sns.scatterplot(x='y',y='carat',data=df, ax=ax[0,1])
sns.scatterplot(x='z',y='carat',data=df, ax=ax[0,2])
sns.scatterplot(x='x',y='price',data=df, ax=ax[1,0])
sns.scatterplot(x='y',y='price',data=df, ax=ax[1,1])
sns.scatterplot(x='z',y='price',data=df, ax=ax[1,2])
plt.show()


# Majority of the diamonds have x values between 4 and 8, y values between 4 and 10 and z values between 2 and 6. Diamonds with other dimensions are very rare.

# ## Train Test Split

# In[22]:


from sklearn.model_selection import train_test_split
x_test,x_train,y_test,y_train = train_test_split(df.drop('price',axis=1),df['price'],test_size=0.2,random_state=42)


# ## Model Building

# ### Decision Tree Regressor

# In[23]:


from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt


# In[24]:


#training the model
dt.fit(x_train,y_train)
#train accuracy
dt.score(x_train,y_train)


# In[25]:


#predicting the test set
dt_pred = dt.predict(x_test)


# ### Random Forest Regressor

# In[26]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf


# In[27]:


#training the model
rf.fit(x_train,y_train)
#train accuracy
rf.score(x_train,y_train)


# In[28]:


#predicting the test set
rf_pred = rf.predict(x_test)


# ## Model Evaluation

# In[29]:


from sklearn.metrics import mean_squared_error,mean_absolute_error


# ### Decision Tree Regressor

# In[30]:


#distribution plot for actual and predicted values
ax = sns.distplot(y_test,hist=False,color='r',label='Actual Value')
sns.distplot(dt_pred,hist=False,color='b',label='Fitted Values',ax=ax)
plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price')
plt.ylabel('Proportion of Diamonds')
plt.show()


# In[31]:


print('Decision Tree Regressor RMSE:',np.sqrt(mean_squared_error(y_test,dt_pred)))
print('Decision Tree Regressor Accuracy:',dt.score(x_test,y_test))
print('Decision Tree Regressor MAE:',mean_absolute_error(y_test,dt_pred))


# ### Random Forest Regressor

# In[32]:


#distribution plot for actual and predicted values
ax = sns.distplot(y_test,hist=False,color='r',label='Actual Value')
sns.distplot(rf_pred,hist=False,color='b',label='Fitted Values',ax=ax)
plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price')
plt.ylabel('Proportion of Diamonds')
plt.show()


# In[33]:


print('Random Forest Regressor RMSE:',np.sqrt(mean_squared_error(y_test,rf_pred)))
print('Random Forest Regressor Accuracy:',rf.score(x_test,y_test))
print('Random Forest Regressor MAE:',mean_absolute_error(y_test,rf_pred))


# ## Conclusion

# Both the models have almost same accuracy. However, the Random Forest Regressor model is slightly better than the Decision Tree Regressor model.
# 
# There is something interesting about the data. The price of the diamonds with J color and I1 clarity is higher than the price of the diamonds with D color and IF clarity which couldn't be explained by the models. This could be because of the other factors that affect the price of the diamond.
# 
# 

# In[ ]:




