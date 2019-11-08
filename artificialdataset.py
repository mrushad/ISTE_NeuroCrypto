#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
def random_bools(size, n):
  
  # creating a n * size array temp with random values 0 and 1
  temp =  np.random.random_integers(low=0, high=1, size=[size, n])
  
  # Convert 0 -> -1 and 1 -> 1 
  temp = temp*2 - 1
  
  return temp.astype(np.float32)
  
def get_dataset(sample_size, TEXT_SIZE, KEY_SIZE):

  m = random_bools(sample_size, TEXT_SIZE)
  k = random_bools(sample_size, KEY_SIZE)
  
  return m, k

