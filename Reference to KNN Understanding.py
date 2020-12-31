#!/usr/bin/env python
# coding: utf-8

# In[34]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


# In[2]:


data = np.load('./mnist_train_small.npy') # we have take mnist data set.


# We have dimensions of approx 20K rows and 785 columns. Out of 785 columns, 1 column is our y i.e Output and remaining 784 i.e. image of 28*28 pixels is our examples or input.  

# In[3]:


data


# In[4]:


data.shape


# In[5]:


X = data[:,1:] # All the rows and column from 1 to 785. 
Y = data[:,0] # All the rows and column with index 0.


# In[6]:


X


# In[7]:


Y


# In[8]:


X.shape, Y.shape # for each x examples we will have y answers.


# In[10]:


plt.imshow(X[0].reshape(28,28), cmap ='gray')


# In[12]:


Y[0]


# An example to show array of two features in cluster form.

# In[41]:


X, y = make_blobs(n_samples=100, centers = 3,random_state=42) # 42 value is something like standard value.
# random state is somewhat we check that improvements or changes made in our model should be on same dataset.


# In[42]:


plt.scatter(X[:, 0], X[:, 1])


# In[33]:


X


# In[34]:


X[:5], y[:5]


# In[37]:


plt.scatter(X[:, 0], X[:, 1], c=y)


# ## Custom Implementation Points :
# 1. <a href =" https://www.geeksforgeeks.org/self-in-python-class/#:~:text=By%20using%20the%20%E2%80%9Cself%E2%80%9D%20keyword,attributes%20with%20the%20given%20arguments.&text=self%20is%20parameter%20in%20function,increase%20the%20readability%20of%20code"> self keyword </a>.
# 2. <a href = "https://www.geeksforgeeks.org/constructors-in-python/#:~:text=Constructors%20are%20generally%20used%20for,when%20an%20object%20is%20created"> constructors </a>

# ## Understanding about zip function:

# In[2]:


x = [1,2,3,4,5] # x represent the point.
y = [10,9,8,7,6] # y represents which class it belongs to.
list(zip(x,y))  # taking two list in pair to create tuples.


# ## Understanding about sorted & unique functions:

# In[42]:


li =[ [12,1],
     [158,0],
     [23,2],
    [ 6,2], [20, 3],[90,2],[78,1], [7, 0],[5,0],[3,0]
]


# In[43]:


sorted(li) # if we want to sort according the first element among collection of lists in the list.


# In[44]:


get_ipython().run_line_magic('pinfo', 'sorted')


# In[45]:


sorted(li, key = lambda x: x[1])


# In[46]:


sorted_li = sorted(li)


# In[47]:


top_k = sorted_li[:5] # slicing


# In[48]:


top_k


# In[49]:


np.unique(np.array(top_k)[:, 1]) # to find array of unique classes with all rows and first column.


# In[50]:


np.unique(np.array(top_k)[:, 1], return_counts = True) # to find the count i.e frequency of occurence of classes.


# In[52]:


li, counts = np.unique(np.array(top_k)[:, 1], return_counts = True)
np.argmax (counts) # giving the class which occurs maximum number of times.


# In[53]:


counts # array having number of occurences


# In[54]:


li # array having unique classes


# In[55]:


li[np.argmax(counts)]


# In[57]:


np.array([1,2,3,4,5,6]) == np.array([1,1,1,4,5,2])


# In[58]:


sum(np.array([1,2,3,4,5,6]) == np.array([1,1,1,4,5,2]))


# ## Understanding use of astype :

# The number is quite large in 784 dimensions. calculating distance 784 times is very large number therefore we are storing it in int64,as it is quite large number to accomodate.
# If we don't use this, then they will not have enough space for computational calculations as it is matrix of 20,000 * 784.
