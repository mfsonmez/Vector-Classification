# Imports
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

def purity_score(y_true, y_pred):
    # matrix which will hold the majority-voted labels
    y_voted_labels = np.zeros(y_true.shape)
    # Ordering labels
    ## Labels might be missing e.g with set like 0,2 where 1 is missing
    ## First find the unique labels, then map the labels to an ordered set
    ## 0,2 should become 0,1
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that 
    # we count the actual occurence of classes between two consecutive bin
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner
    
    return accuracy_score(y_true, y_voted_labels)

# Configuration options
blobs_random_seed = 42
centers = [(0,0), (5,5)]
cluster_std = 1
frac_test_split = 0.33
num_features_for_samples = 2
num_samples_total = 2000

try:
    #Load Data
    X_train, X_test, y_train, y_test = np.load('./data1.npy', allow_pickle=True)
    X_total = np.concatenate([X_train, X_test])
except:
    # Generate data
    inputs, targets = make_blobs(n_samples = num_samples_total, centers = centers, n_features = num_features_for_samples, cluster_std = cluster_std)
    X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=frac_test_split, random_state=blobs_random_seed)
    # Save and load temporarily
    np.save('./data1.npy', (X_train, X_test, y_train, y_test))
    X_train, X_test, y_train, y_test = np.load('./data1.npy', allow_pickle=True)
    X_total = np.concatenate([X_train, X_test])

# Show Total Data on plot
plt.scatter(X_total[:,0], X_total[:,1])
plt.title('Linearly separable data')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

# Show Total Data on plot
plt.scatter(X_test[:,0], X_test[:,1])
plt.title('Linearly separable test data')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

kmeans = KMeans(n_clusters=2)
kmeans.fit(X_train, y_train)
pred_y = kmeans.predict(X_test)

#Calculate rand index (labels_true, labels_pred)
print("Rand Index:")
print(adjusted_rand_score(y_test, pred_y))

#Calculate purity score (labels_true, labels_pred)
print("Purity Score:")
print(purity_score(y_test, pred_y))

#Calculate classification report
print("Classification Report:")
print(classification_report(y_test, pred_y))

plt.scatter(X_test[:, 0], X_test[:, 1], c=pred_y, s=50, cmap='viridis')
plt.title("K-means Algorithm")
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=50, alpha=0.5);
plt.show()