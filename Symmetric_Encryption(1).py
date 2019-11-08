
# coding: utf-8

# In[6]:


def symm_encryp_create(input_message, input_secret_key):
 
  # Alice creates a ciphertext and a secret key from the input message
  Alice_out_ciphertext = model( 'Alice', input_message, input_secret_key)
  
  # Bob takes the ciphertext along with the secret key to decrypt and generate the original message
  Bob_out_message_decrypted      = model( 'Bob',   Alice_out_ciphertext, input_secret_key)

  # Eve takes in just the ciphertext text and tries to decrpyt it without using key
  Eve_out_message_decrypted      = model( 'Eve',   Alice_out_ciphertext)
  
  return Bob_out_message_decrypted, Eve_out_message_decrypted



# In[ ]:

`

