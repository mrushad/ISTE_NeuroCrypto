def alice_bob_loss(input_message, bob_out_message, eves_loss, batch_size):
  
  #Part for Alice and Bob in which the number of bits is compared and error is calculated
  ab_diff = (bob_out_message + 1.0)/2.0 - (input_message + 1.0)/2.0
  
  # The average L1 distanceof LAD is calculated between Alice and Bob
  bob_reconstruction_loss = (1/batch_size)*tf.reduce_sum(tf.abs(ab_diff))

  
  # Part of calculation of errror involving the reconstruction of Eve.
  # We take Eve's % of error to be 50% to avoid complication
  # Reason for that is that if Eve gets 100% error(Max error) the next iteration could be fully correct by simply flipping the bits.
  # ((N/2 - EveLoss)^2)/((N/2)^2)
  eve_evadropping_loss = tf.reduce_sum( tf.square(float(TEXT_SIZE) / 2.0 - eves_loss) / (TEXT_SIZE / 2)**2)

  # The error in the number of bits plus the error in the actual ciphertext(Replication of Eve)
  loss = bob_reconstruction_loss + eve_evadropping_loss
  
  return loss
