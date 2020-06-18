import numpy as np
from copy import deepcopy
from time import time
from sys import maxsize
from scipy.spatial.distance import cdist

#threading stuff
import threading

# how close the old and new centroids are allowed to be in kmeans
THRESHOLD = 10
# max number of centroids to try out for kmeans tuned
MAX_K = 5


def kmeans(training, K = 2, PRINT = 0):
    """

        1. Choose K. Choose location of centroid.
        2. Assign each point of input to K.
        3. Update centroid with average of points within centroid range.
        4. repear 2 and 3 until no input point is reassigned.

        parameters:
            training - the training example to find centroids of
            K - number of centroids to find
            PRINT - whether or not to display additional information about the process

        return:
            centroids - the array representing the k many centroids.
                    A data sample belongs to whatever centroid it is closest to,
                    and takes that centroid's label (i.e. 0,1,2,...,k).
    """
    if PRINT:
        print("kmeans:",K," :OUTPUT MODE ENABLED.")
    dimensions = training[0].shape

    t = np.array(training)
    K_num = K # number of centroids
    K_dim = (K_num,) + dimensions # dimensions of K
    centroids = t[np.random.choice(t.shape[0], K_num),:] # start values of centroids
    
    if PRINT:
        print("Training data:\n", t)
        print("Starting centroids:\n", centroids)
        print()

    # list of labels for each training
    # ie label at training_labels[i] is the centroid
    # that training[i] belongs to.
    training_labels = np.zeros(t.shape[0]) * -1

    # centroids from previous iteration 
    OLD_centroids = np.zeros(centroids.shape)


    difference = dist(centroids, OLD_centroids) 
    while difference > THRESHOLD:    
        print("Difference between old and new centroids:", difference)
        # ASSIGN CENTROID LABELS TO TRAINING.
        """
        for i in range(t.shape[0]):
            # list of distances of point training i to 
            # all the centroids
            distances = dists(t[i], centroids)
            # index of the cluster that has the lowest 
            # distance to training[i]
            cluster = np.argmin(distances)
            # assigning the label to the ith training sample
            training_labels[i] = cluster
        """
        # get distance of training from each centroid
        t_k_distance = cdist(t, centroids) 
        # get the closest centroid's number for each training example
        training_labels = np.argmin(t_k_distance, axis=1)

        # assign current centroids to old before update
        OLD_centroids = deepcopy(centroids)
        # UPDATE CENTROIDS with averages

        if PRINT:
            print("Updating centroids...")

        for k in range(K_num):
            if PRINT:
                print("Updating centroid", k, "...")
            kPoints = [training[i] for i in range(t.shape[0]) if training_labels[i] == k]
            if len(kPoints) == 0:
                continue
            kMean = np.mean(kPoints, axis=0)
            centroids[k] = kMean
        difference = dist(centroids, OLD_centroids)

    if PRINT:
        # OUTPUT CENTROIDS
        print("Difference between old and new centroids:", difference)
        print("Final Centroids:", centroids)
        print("Training data:", (t, training_labels))

    return (centroids, training_labels)
            

def dists(point, centroids):
    return [dist(point,c) for c in centroids]

def dist(pa, pb):
    """
        Euclidean distance of point A from point B.
    """
    return np.linalg.norm(np.array(pa)-np.array(pb))


def kmeansTunedThread(training, k, WCSSk, PRINT=0):
    if PRINT:
        print("K: ",k, "started.")

    iterations = 5
    for i in range(iterations):
        print(k, ":", i, "/", iterations)
        clusterI = []
        cent, labels = kmeans(training, PRINT=PRINT, K=k)
        clusterI.append(_calcWCSS(training, labels, cent))
    avg = sum(clusterI)/iterations
    if avg == 0:
        WCSSk[k-1] = maxsize
    else:
        WCSSk[k-1] = avg    

    if PRINT:
        print("K: ",k, "finished.")

def _kmeansTuned(training, PRINT=0):
    """
    Runs kmeans several times and finds the best K value (number of centroids to use).
    """
    if PRINT:
        print("Kmeans Tunded: OUTPUT MODE ENABLED.")
    # run kmeans with different k
    WCSSk = [maxsize] * MAX_K
    threads = [threading.Thread(target=kmeansTunedThread,
        args=(training,k,WCSSk, PRINT)) for k in range(1,MAX_K+1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("WCSS: ",WCSSk)

    # UNTHREAD
    #for i in range(1,MAX_K+1):
    #    kmeansTunedThread(training,i,WCSSk, PRINT=PRINT)

    if PRINT:
        print("WCCS:",WCSSk)
    candidates = [ WCSSk[i-1]-2*WCSSk[i]+WCSSk[i+1] for i in range(1,len(WCSSk)-1)]
    if PRINT:
        print("Candidates: ", (candidates))
    # gets the k of the lowest WCSS
    bestK = np.argmax(np.array(candidates)) + 1 + 1
    return bestK

def runKmeansTuned(training, PRINT=0):
    """
        Finds the optimal number of centroids (between 1 and MAX_K), then runs kmeans with that many centroids.
    """
    if PRINT:
        print("run kmeans tuned: OUTPUT MODE ENABLED.")
    k  = _kmeansTuned(training, PRINT=PRINT)
    print("Number of clusters (k): {}".format(k))
    return kmeans(training, K=k, PRINT=PRINT)

def _calcWCSS(T, labels, C):
    """
        parameters:
            T - training data; input data.
            labels - labels for each training sample. ith label corresponds to ith training sample
            C - the k centroids
    """
    WCSSsum = []
    # calculates the WCSS per cluster.
    for i in range(len(C)):
        WCSSsum.append(sum([dist(T[j], C[i]) for j in range(len(T)) if labels[j]==i]))
    # total WCSS of all cluster
    totalWCSS = np.sum(WCSSsum)
    return totalWCSS



if __name__ == "__main__":
    # simple test.
    test = np.random.randint(0, high=255, size=((100,100,3)))
    print(kmeans(test))

    bestK = _kmeansTuned(test) 
    print("Best K to use:", bestK)
    print(kmeans(test, K=bestK))