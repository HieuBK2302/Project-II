from scipy import io
from sklearn.cluster import KMeans
from sklearn import metrics



f = io.mmread("train.mtx")
label = open("label.txt","r")
label = label.read()
label_true = []
line = label.split("\n")
for l in line:
	if (l != ' '):
		label_true.append(l)
del label_true[-1]
kmeans = KMeans(n_clusters = 20, init = 'k-means++', random_state=0)
test = kmeans.fit(f)
#print(kmeans.cluster_centers_)
label_pre = test.labels_
print(label_true)
print(label_pre)

k = metrics.adjusted_rand_score(label_true, label_pre)
print(k)