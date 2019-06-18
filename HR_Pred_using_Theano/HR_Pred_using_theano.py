#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import theano
import theano.tensor as T
def error_rate(p, t):
    return np.mean(p!=t)


# In[2]:


df = pd.read_csv("HR_comma_sep.csv")
df.groupby('left').mean()


# In[3]:


Y = df['left']
subdf = df[['satisfaction_level', 'average_montly_hours', 'promotion_last_5years', 'salary']]
subdf.head()


# In[4]:


salary_dummies = pd.get_dummies(subdf.salary, prefix = "salary")
subdf.drop(["salary"], axis = 1, inplace = True)


# In[5]:


df1 = pd.concat([subdf, salary_dummies], axis = 1)


# In[6]:


df1.drop(["average_montly_hours"], axis = 1, inplace = True)


# In[42]:


y = []
for i in Y:
    if i == 1:
        y.append(0)
    else:
        y.append(1)
y = pd.Series(y)

Yr = pd.concat([Y, y], axis = 1)

X_train, X_test, Y_train, Y_test = train_test_split(df1.values, Yr.values, test_size =0.3)
X_train.shape


# In[43]:


train_examples, input_size = X_train.shape
hidden_size = 3
output_size = 2
batch_sz = 100
n_batches = train_examples // batch_sz
reg = 0.01
max_iter = 20
print_period = 10
lr = 0.000006

W1_init = np.random.random([input_size, hidden_size])
b1_init = np.zeros(hidden_size)

W2_init = np.random.random([hidden_size, output_size])
b2_init = np.zeros(output_size)

thX = T.matrix("X")
thT = T.matrix("T")
W1 = theano.shared(W1_init, "W1")
W2 = theano.shared(W2_init, "W2")
b1 = theano.shared(b1_init, "b1")
b2 = theano.shared(b2_init, "b2")

thZ = T.nnet.relu(thX.dot(W1) + b1)

thY = T.nnet.softmax(thZ.dot(W2) + b2)

prediction = T.argmax(thY, axis = 1)

cost = -(thT*T.log(thY)).sum() + reg*((W1*W1).sum() + (b1*b1).sum() + (W2*W2).sum() + (b2*b2).sum())

update_W1 = W1 - lr*T.grad(cost, W1)
update_b1 = b1 - lr*T.grad(cost, b1)
update_W2 = W2 - lr*T.grad(cost, W2)
update_b2 = b2 - lr*T.grad(cost, b2)

train = theano.function([thX, thT], updates = [(W1, update_W1), (W2, update_W2), (b1, update_b1), (b2, update_b2)])

get_prediction = theano.function(inputs = [thX, thT], outputs = [cost, prediction])

costs = []




# In[44]:


for i in range(max_iter):
    for j in range(n_batches):
        Xbatch = X_train[j*batch_sz:(j*batch_sz + batch_sz)]
        Ybatch = Y_train[j*batch_sz:(j*batch_sz + batch_sz)]

        train(Xbatch, Ybatch)
        if j % print_period == 0:
            cost_val, prediction_val = get_prediction(X_test, Y_test)
            err = error_rate(prediction_val, Y_test)
            print("Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j, cost_val, err))
            costs.append(cost_val)


# In[47]:


import matplotlib.pyplot as plt
plt.plot(costs)
plt.show()


# In[ ]:




