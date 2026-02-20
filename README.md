import numpy as np
np.random.seed(42)
x = np.array([0.6, -0.3])   
W1 = np.random.uniform(-0.5, 0.5, (2, 1)) 
W2 = np.random.uniform(-0.5, 0.5, (1, 1))  

b1 = 0.5
b2 = 0.7

z1 = np.dot(x, W1) + b1
a1 = np.tanh(z1)

z2 = np.dot(a1, W2) + b2
output = np.tanh(z2)

print("W1:\n", W1)
print("W2:\n", W2)
print("Hidden layer output:", a1)
print("Final network output:", output)
