# Neural Cryptography and Secure Communications

The ISTE NEUROCRYPTO project is an implementation of the paper: <a href="https://arxiv.org/abs/1610.06918">"Learning to Protect Communications with Adversarial Neural Cryptography"</a> by Martín Abadi and David G. Andersen at Google Brain.
It was done using Python and Tensorflow.

### Project Idea
The following is an extract from the paper:
"We ask whether neural networks can learn to use secret keys to protect information from other neural networks. Specifically, we focus on ensuring confidentiality properties in a multiagent system, and we specify those properties in terms of an adversary. Thus, a system may consist of neural networks named Alice and Bob, and we aim to limit what a third neural network named Eve learns from eavesdropping on the communication between Alice and Bob. We do not prescribe specific cryptographic algorithms to these neural networks; instead, we train end-to-end, adversarially. We demonstrate that the neural networks can learn how to perform forms of encryption and decryption, and also how to apply these operations selectively in order to meet confidentiality goals."

### Prereqiusites

To run this project, you'll need to have the following installed:
1. python3
2. matplotlib>=3.1.0
3. Tensorflow (Separate code is provided to run in tf1 and tf2)
4. numpy>=1.16.4

### Running
1. For running in tf1:
```
python3 maincode.py
```

2. For running in tf2:
```
python3 maincodetf2.py
```
