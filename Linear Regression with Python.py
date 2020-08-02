
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Check out the Data

# In[256]:


USAhousing = pd.read_csv('USA_Housing.csv')


# In[257]:


USAhousing.head()


# In[258]:


USAhousing.info()


# In[259]:


USAhousing.describe()


# In[260]:


USAhousing.columns


# # EDA
# 
# Let's create some simple plots to check out the data!

# In[261]:


sns.pairplot(USAhousing)


# In[262]:


sns.distplot(USAhousing['Price'])


# In[263]:


sns.heatmap(USAhousing.corr())


# ## Training a Linear Regression Model
# 
# Let's now begin to train out regression model! We will need to first split up our data into an X array that contains the features to train on, and a y array with the target variable, in this case the Price column. We will toss out the Address column because it only has text info that the linear regression model can't use.
# 
# ### X and y arrays

# In[264]:


X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = USAhousing['Price']


# ## Train Test Split
# 
# Now let's split the data into a training set and a testing set. We will train out model on the training set and then use the test set to evaluate the model.

# In[265]:


from sklearn.model_selection import train_test_split


# In[266]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# ## Creating and Training the Model

# In[267]:


from sklearn.linear_model import LinearRegression


# In[268]:


lm = LinearRegression()


# In[269]:


lm.fit(X_train,y_train)


# ## Model Evaluation
# 
# Let's evaluate the model by checking out it's coefficients and how we can interpret them.

# In[270]:


# print the intercept
print(lm.intercept_)


# In[277]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


#

# ## Predictions from our Model
# 
# Let's grab predictions off our test set and see how well it did!

# In[279]:


predictions = lm.predict(X_test)


# In[282]:


plt.scatter(y_test,predictions)


# **Residual Histogram**

# In[281]:


sns.distplot((y_test-predictions),bins=50);



# 
# Comparing these metrics:
# 
# - **MAE** is the easiest to understand, because it's the average error.
# - **MSE** is more popular than MAE, because MSE "punishes" larger errors, which tends to be useful in the real world.
# - **RMSE** is even more popular than MSE, because RMSE is interpretable in the "y" units.
# 
# All of these are **loss functions**, because we want to minimize them.

# In[275]:


from sklearn import metrics


# In[276]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# This was your first real Machine Learning Project! Congrats on helping your neighbor out! We'll let this end here for now, but go ahead and explore the Boston Dataset mentioned earlier if this particular data set was interesting to you! 
# 
# Up next is your own Machine Learning Project!
# 
# ## Great Job!
