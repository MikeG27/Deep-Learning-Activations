import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
sns.set()

from activations import *

x1 = np.linspace(-8,8)
x2 = np.array([np.linspace(-8,8,30)])

################# SIGMOID ##################

plt.figure(figsize=(15,8))

plt.plot(x1,sigmoid(x1),"r")
plt.scatter(x2,sigmoid(x2))
pylab.text(4, 0.8, r'$\sigma(x)=\frac{1}{1+e^{-x}}$', fontsize=30)
plt.title("Sigmoid function")
plt.xlabel("X axis")
plt.ylabel("Y axis")

plt.savefig("sigmoid.png")

################# TANH  ##################

plt.figure(figsize=(15,8))


plt.plot(x1,tanh(x1),"r")
plt.plot([-8,8],[0,0],"grey") # Horizontal line
plt.scatter(x2,tanh(x2))
plt.title("Tanh function")
plt.xlabel("X axis")
plt.ylabel("Y axis")

plt.savefig("tanh.png")


################# RELU  ##################

plt.figure(figsize=(15,8))
plt.plot(x1,relu(x1),"r")
plt.scatter(x2,relu(x2))
plt.plot([0,0],[0,10],"grey") # Vertical line
plt.plot([-8,8],[0,0],"grey") # Horizontal line
plt.title("Relu function")
plt.xlabel("X axis")
plt.ylabel("Y axis")

plt.savefig("relu.png")


################# SOFTMAX  ##################

#data
a = np.linspace(0,10,8)
a = a.reshape(a.shape[0],1)

plt.figure(figsize=(15,8))

plt.plot(a,softmax(a),linewidth = 2)
plt.scatter(a,softmax(a),c="r",linewidth = 5)
plt.title("Softmax function")
plt.xlabel("X axis")
plt.ylabel("Y axis")

plt.savefig("softmax.png")