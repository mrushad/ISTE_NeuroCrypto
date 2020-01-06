# Input configuration.
TEXT_SIZE = 16
KEY_SIZE  = 16

# Training parameters.
learning_rate   = 0.0008
batch_size      = 4096
sample_size     = 4096*5
epochs          = 850000
steps_per_epoch = int(sample_size/batch_size)

ITERS_PER_ACTOR = 1
EVE_MULTIPLIER = 2  # Train Eve 2x for every step of Alice/Bob

# Function to return a list of tensor's to be used in training based on the scope of model
def get_collection(collection):
  return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=collection) 

sess = tf.Session()

# Create symmetric cryptosystem in form of adversarial networks
Bob_out, Eve_out = symm_crypto_create(input_message, input_key)

# Loss
eve_loss       = eve_loss(input_message, Eve_out, batch_size)
alice_bob_loss = alice_bob_loss(input_message, Bob_out, Eves_loss, batch_size)

# Collect each model's tensors
alice_vars = get_collection('Alice')
bob_vars   = get_collection('Bob')
eve_vars   = get_collection('Eve')

# optimizers
Eve_opt  = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, epsilon=1e-08).minimize(eve_loss, var_list=[eve_vars])
bob_opt  = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, epsilon=1e-08).minimize(alice_bob_loss, var_list=[alice_vars + bob_vars]) 

# initialize variables
init = tf.global_variables_initializer()
sess.run(init)


# DATASET 
messages, keys = get_dataset(sample_size, TEXT_SIZE, KEY_SIZE)

for i in range(epochs):

  for j in range(steps_per_epoch):
    
    # select the batch from the samples
    batch_messages = messages[j*batch_size: (j+1)*batch_size]
    batch_keys     = keys[j*batch_size: (j+1)*batch_size]

    # Train Alice and Bob's models
    for _ in range(ITERS_PER_ACTOR):
      temp = sess.run([bob_opt, alice_bob_loss],feed_dict={input_message:batch_messages , input_key:batch_keys })
      Alice_bob_loss = temp[1]

    # Train Eve's model
    for _ in range(ITERS_PER_ACTOR*EVE_MULTIPLIER):
      temp = sess.run([Eve_opt, eve_loss], feed_dict={input_message:batch_messages , input_key:batch_keys })
      Eve_loss = temp[1]

  # output Alice-Bob loss and Eve's loss after every 100 iterations  
  if i%100 == 0:
    print(i,'  Alice_bob_loss: ', Alice_bob_loss,'    Eve_loss:', Eve_loss)

sess.close()
