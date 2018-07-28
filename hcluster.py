import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

np.random.seed(0)
data = np.random.randn(10, 2)

def compute_pairwise_distances(data, metric):
    return pairwise_distances(X=data, metric=metric, n_jobs=-1)


class Node(object):
    def __init__(self, id, height):
        self.id = id
        self.height = height
        self.child = None

    def __repr__(self):
        return "id: {}\ndepth:{}".format(self.id, self.height)

class Cluster(object):
    def __init__(self, data, dist_metric):
        num_instances = len(data)
        self.remaining_nodes = [Node([i], 0) for i in range(num_instances)]
        self.height = 0
        self.dist = compute_pairwise_distances(data, dist_metric)

    def get_index_of_two_closest_clusters(self):
        np.fill_diagonal(self.dist, np.inf)
        index = np.unravel_index(np.argmin(self.dist, axis=None), self.dist.shape)
        np.fill_diagonal(self.dist, 0)  # revert op
        return index

    def merge(self, merge_id1, merge_id2):
        self.height += 1

        x1 = self.remaining_nodes[merge_id1]
        x2 = self.remaining_nodes[merge_id2]

        merged_x = Node(x1.id + x2.id, self.height)

        self.remaining_nodes.append(merged_x)
        self.update_dist(merge_id1, merge_id2)

    def update_dist(self, remove_id1, remove_id2):
        del self.remaining_nodes[remove_id1]
        del self.remaining_nodes[remove_id2 - 1]

        self.dist = np.delete(self.dist, remove_id1, axis=0)
        self.dist = np.delete(self.dist, remove_id2 - 1, axis=0)
        self.dist = np.delete(self.dist, remove_id1, axis=1)
        self.dist = np.delete(self.dist, remove_id2 - 1, axis=1)

        new_cluster_ids = self.remaining_nodes[-1].id
        dist_to_new_cluster = []
        for i in range(len(self.remaining_nodes) - 1):
            tmp_dist = self.compute_cluster_distance(self.remaining_nodes[i].id, new_cluster_ids, data)
            dist_to_new_cluster.append(tmp_dist)

        row = np.array(dist_to_new_cluster).reshape(1, -1)
        col = np.array(dist_to_new_cluster + [0]).reshape(-1, 1)
        self.dist = np.concatenate((self.dist, row), axis=0)
        self.dist = np.concatenate((self.dist, col), axis=1)

    def compute_cluster_distance(self, c1, c2, data):
        print("dist between indices:", c1, c2)
        x1 = np.take(data, c1, axis=0)
        x2 = np.take(data, c2, axis=0)

        dist_mat = pairwise_distances(x1, x2)
        dist = self.linkage(dist_mat, "single")
        return dist

    def linkage(self, dist_mat, metric):
        if metric == "complete":
            np.fill_diagonal(dist_mat, -1)
            dist = np.max(dist_mat)
        if metric == "single":
            np.fill_diagonal(dist_mat, np.inf)
            dist = np.min(dist_mat)
        return dist

# eval with visualization
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.scatter(data[:,0], data[:,1])
for i in range(len(data)):
    ax.annotate(i, (data[i, 0], data[i, 1]))
plt.savefig('cluster.png')
plt.show()


# Main program
root = Cluster(data, 'euclidean')
for _ in range(10):
    if len(root.remaining_nodes) == 3:
        break
    indices_to_merge = root.get_index_of_two_closest_clusters()
    print("id to merge:", indices_to_merge)
    root.merge(indices_to_merge[0], indices_to_merge[1])
print("remaining:", root.remaining_nodes)







