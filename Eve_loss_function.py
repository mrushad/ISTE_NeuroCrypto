def eve_loss(input_message, eve_out_message, batch_size):

  # Mapping -1 to 0 and 1 to 1
  eve_diff = (eve_out_message + 1.0)/2.0 - (input_message + 1.0)/2.0
  
  # Eve's average L1 distance Loss of the given batch
  loss = (1/batch_size)*tf.reduce_sum(tf.abs(eve_diff))
  
  return loss
