#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split


# In[2]:


df = pd.read_csv("HR_comma_sep.csv")
df.groupby("left").mean()


# In[3]:


df1 = df[["satisfaction_level", "average_montly_hours", "Work_accident", "promotion_last_5years"]]
Y = df.left

y = []
for i in Y:
    if i == 1:
        y.append(0)
    else:
        y.append(1)


y = pd.Series(y)
Yr = pd.concat([Y, y], axis = 1)


# In[4]:


X_train, X_test, Y_train, Y_test = train_test_split(df1.values, Yr.values, test_size = 0.2)


# In[42]:


inputs = tf.placeholder(tf.float32, [None, 4])
targets = tf.placeholder(tf.float32, [None, 2])

weights_1 = tf.get_variable("weights_1", [4,3])
biases_1 = tf.get_variable("biases_1", [3])

outputs_1 = tf.nn.relu(tf.matmul(inputs, weights_1) + biases_1)

weights_2 = tf.get_variable("weights_2", [3, 2])
biases_2 = tf.get_variable("biases_2", [2])

outputs = tf.matmul(outputs_1, weights_2 ) + biases_2

loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits = outputs, labels = targets)

mean_loss = tf.reduce_mean(loss)

optimize = tf.train.AdamOptimizer().minimize(mean_loss)

out_equals_targets = tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1))

accuracy = tf.reduce_mean(tf.cast(out_equals_targets, tf.float32))


# In[43]:


sess = tf.InteractiveSession()


# In[44]:


initializer  = tf.global_variables_initializer()
sess.run(initializer)


# In[45]:


N, D = X_train.shape
batch_size = 1
batches_number = N // batch_size
max_epochs = 20


# In[29]:


for epoch_counter in range(max_epochs):
    curr_epoch_loss = 0
    
    for batch_counter in range(batches_number):
        
        
        input_batch = X_train[batch_counter*batch_size: (batch_counter + 1)*batch_size]
        target_batch = Y_train[batch_counter*batch_size: (batch_counter + 1)*batch_size]
        
        _, batch_loss = sess.run([optimize, mean_loss],
                                feed_dict = {inputs : input_batch, targets : target_batch})
        
        curr_epoch_loss += batch_loss
        #print("Batch loss at batch number: %d is %.3f " %(batch_counter, batch_loss))
        
    curr_epoch_loss /= batches_number
    
    
    print('Epoch '+str(epoch_counter+1)+
          '. Mean loss: '+'{0:.3f}'.format(curr_epoch_loss))

print("End Of Training")


# In[30]:


input_batch, target_batch = X_test, Y_test

test_accuracy = sess.run([accuracy], 
    feed_dict={inputs: input_batch, targets: target_batch})

test_accuracy_percent = test_accuracy[0] * 100.

print('Test accuracy: '+'{0:.2f}'.format(test_accuracy_percent)+'%')


# In[ ]:

