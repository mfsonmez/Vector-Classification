from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

class NN():
  def __init__(self, n_samples, n_features, n_outputs, n_hidden = 1):
    self.n_samples = n_samples
    self.n_features = n_features
    self.n_hidden = n_hidden
    self.n_outputs = n_outputs

    self.W_h = np.random.randn(n_features, n_hidden)
    self.b_h = .01 + np.zeros((1, n_hidden))
    self.W_o = np.random.randn(n_hidden, n_outputs)
    self.b_o = .01 + np.zeros((1, n_outputs))

  def sigmoid(self, x):
    return 1/(1 + np.exp(-x))

  def loss(self, y, p_pred):
    return -1/y.shape[0] * (np.sum(y * np.log(p_pred) + (1 - y) * (np.log(1 - p_pred))))

  def predict(self, X):
    return np.squeeze(np.round(self.forward_prop(X)["O"]))

  def forward_prop(self, X):
    # Hidden layer
    A_h = X @ self.W_h + self.b_h
    H = self.sigmoid(A_h)

    # Output layer
    A_o = H @ self.W_o + self.b_o
    O = self.sigmoid(A_o)
    return {
      "A_h": A_h, 
      "H": H, 
      "A_o": A_o, 
      "O": O
    }

  # This is not a true implmentation of backprop
  def backward_prop(self, X, y_, forward):
    one_n = np.ones(self.n_samples)
    y = (y_[np.newaxis]).T # convert to column vector

    dA_o = (y - forward["O"])
    dL_dW_o = 1/self.n_samples * forward["H"].T @ dA_o
    dL_db_o = 1/self.n_samples * one_n.T @ dA_o
    
    dA_h = (dA_o @ self.W_o.T) * (self.sigmoid(forward["A_h"]) * (1 - self.sigmoid(forward["A_h"])))
    dL_dW_h = 1/self.n_samples * X.T @ dA_h
    dL_db_h = 1/self.n_samples * one_n.T @ dA_h

    return {
      "dL_dW_h": dL_dW_h, 
      "dL_db_h": dL_db_h, 
      "dL_dW_o": dL_dW_o, 
      "dL_db_o": dL_db_o
    }

  def train(self, X, y, learning_rate = .5, max_iter = 1001):
    print("Learning Rate:", learning_rate)
    for i in range(0, max_iter):
      forward_prop_dict = self.forward_prop(X)
      G = self.backward_prop(X, y, forward_prop_dict)

      # Gradient step
      self.W_h = self.W_h + learning_rate * G["dL_dW_h"]
      self.b_h = self.b_h + learning_rate * G["dL_db_h"]

      self.W_o = self.W_o + learning_rate * G["dL_dW_o"]
      self.b_o = self.b_o + learning_rate * G["dL_db_o"]

      if i % 100 == 0:
        print(f"Iteration: {i}, Training Loss: {self.loss(y, np.squeeze(forward_prop_dict['O']))}")
        
# Configuration options
blobs_random_seed = 42
centers = [(0,0), (5,5)]
cluster_std = 1
frac_test_split = 0.33
num_features_for_samples = 2
num_samples_total = 2000

try:
    #Load Data
    X_train, X_test, y_train, y_test = np.load('./data2.npy', allow_pickle=True)
    X_total = np.concatenate([X_train, X_test])
except:
    # Generate data
    inputs, targets = make_blobs(n_samples = num_samples_total, centers = centers, n_features = num_features_for_samples, cluster_std = cluster_std)
    X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=frac_test_split, random_state=blobs_random_seed)
    # Save and load temporarily
    np.save('./data2.npy', (X_train, X_test, y_train, y_test))
    X_train, X_test, y_train, y_test = np.load('./data2.npy', allow_pickle=True)
    X_total = np.concatenate([X_train, X_test])

# Show Total Data on plot
plt.scatter(X_total[:,0], X_total[:,1])
plt.title('Linearly separable data')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

fig = plt.figure(figsize=(8,12))
ax = fig.add_subplot(2,1,1)
plt.scatter(X_test[:,0], X_test[:,1], c=y_test)
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Test set without classification")

n_ouputs = 1
n_TRAIN = int(num_samples_total - (frac_test_split* num_samples_total))
nn = NN(n_samples = n_TRAIN, n_features = num_features_for_samples, n_outputs = 1, n_hidden = 10)

learningRates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ]

for learningRate in learningRates:    
    nn.train(X_train, y_train, learningRate)
    y_pred_train = nn.predict(X_train)
    y_pred_test = nn.predict(X_test)

    print("Train accuracy:", 1/X_train.shape[0] * np.sum(y_pred_train == y_train))
    print("Test accuracy:", 1/X_test.shape[0] * np.sum(y_pred_test == y_test))

    ax = fig.add_subplot(2,1,2)
    plt.scatter(X_test[:,0], X_test[:,1], c=y_pred_test)
    plt.xlabel("X1")
    plt.ylabel("X2")
    rate = 'With Learning Rate ' + str(learningRate)
    plt.title("Test set with classification")
    plt.text(-3.5, 8, rate, fontsize = 8, color = 'g')
    plt.show()