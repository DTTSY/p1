from util import exm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.neighbors import kneighbors_graph
from sklearn import datasets
# import seaborn as sns
import logging
import time
import os
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.sparse.csgraph import laplacian
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.utils import shuffle
from itertools import combinations


import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', filename='./log.log', filemode='w')
logger = logging.getLogger(__name__)


class alg:
    def __init__(self, data, k, cluster_lable):
        self.data = data
        self.k = k
        self.cluster_lable = cluster_lable
        self.A = None
        self.C = None
        self.W = None

    def main(self):
        self.construct_graph()

    def construct_graph(self):
        self.A = kneighbors_graph(
            self.data, 3, mode='distance', include_self=False, n_jobs=-1)
        # self.A = kneighbors_graph(self.data, self.k, mode='distance')
        a = self.A.toarray()
        print(a.shape)
        print(f"A is {np.array_equal(a, a.T)} symmetric matrix")
        # make matrix symmetrical
        a = (a+a.T)/2
        self.A = a
        # print(f"isolated nodes: {find_isolated_nodes(a)}")
        # nx_graph = nx.from_numpy_array(
        #     a, create_using=nx.DiGraph)
        # draw_graph_with_nx(
        #     nx_graph, node_color=self.cluster_lable.tolist(), node_size=self.A.shape[0])

    def get_pairwise_distance(self):
        pass

    def get_pairwise_similarity(self):
        pass

    def pairwise_constrain(self, node_pair):
        return int(self.cluster[node_pair[0]] == self.cluster[node_pair[1]])

    def get_pairwise_constrain(self, data):
        return np.apply_along_axis(self.pairwise_constrain, 1, data)


class E2CP:
    def __init__(self, data, Z=None):
        self.data = data
        self.A = None
        self.Z = Z

    def set_pairwise_constrain_matrix(self, Z):
        self.Z = Z

    def set_pairwise_constrain_matrix(self, Z):
        self.Z = Z

    def main(self):
        if self.A == None:
            self.construct_graph()

        L = laplacian(self.A, normed=True)
        F_v = self.A.copy()
        F_prime = None
        alpha = 0.1
        c = F_v
        # for _ in range(100):
        err = 1
        iter = 0
        theshold = 1e-3
        while err > theshold:
            F_v = alpha*np.dot(L, F_v)+(1-alpha)*self.Z
            err = np.sum(np.square(c-F_v))/self.A.shape[0]
            c = F_v
            # iter += 1
            # print(f"step1 iter: {iter} err: {err}")
        F_prime = F_v.copy()
        iter = 0
        while err > theshold:
            F_prime = alpha*np.dot(F_prime, L)+(1-alpha)*F_v
            err = np.sum(np.square(c-F_prime))/self.A.shape[0]
            c = F_prime
            # iter += 1
            # print(f"step2 iter: {iter} err: {err}")

        for i in range(F_prime.shape[0]):
            for j in range(F_prime.shape[1]):
                if F_prime[i, j] >= 0:
                    self.A[i, j] = 1-(1-F_prime[i, j])*(1-self.A[i, j])
                else:
                    self.A[i, j] = (1+F_prime[i, j])*self.A[i, j]

    def construct_graph(self):
        # self.A = kneighbors_graph(
        #     self.data, 10, mode='distance', include_self=False, n_jobs=-1)
        # self.A = np.exp(-7.6*pairwise_distances(self.data, metric='euclidean'))
        # 使用高斯kernel构建相似度矩阵
        self.A = np.exp(-7.6*pairwise_distances(self.data,
                        metric='euclidean', n_jobs=-1))
        # 计算log2(n)
        # log2_n = np.log2(self.A.shape[0])
        # self.A = kneighbors_graph(self.data, self.k, mode='distance')
        # a = self.A.toarray()
        a = self.A
        # print(a.shape)
        # print(f"A is {np.array_equal(a, a.T)} symmetric matrix")
        # make matrix symmetrical
        self.A = (a+a.T)/2


class IC:
    def __init__(self, data, C=None):
        self.data = data
        self.A = None
        self.C = C
        self.graph = None
        self.alpha = 0.1

    def set_pairwise_constrain_matrix(self, Z):
        self.Z = Z

    def set_graph(self, graph):
        self.graph = graph

    def construct_graph(self, pnn=5):
        self.A = kneighbors_graph(
            self.data, pnn, mode='distance', include_self=False, n_jobs=-1)
        self.A = (self.A+self.A.T)/2
        self.graph = nx.from_numpy_array(self.A.toarray())
        # print(self.graph.edges(data=True))

    def prob(self, node, t):
        d = self.graph.degree(node)
        return d/(d+self.alpha*t)

    def activate(self, seed_set):
        active = seed_set.copy()
        newly_active = seed_set.copy()

        while newly_active:
            next_active = []
            for node in newly_active:
                neighbors = self.graph.neighbors(node)
                for neighbor in neighbors:
                    ww = self.graph.edges[node, neighbor]['weight']
                    if neighbor not in active and ww < self.prob(neighbor, 1):
                        next_active.append(neighbor)
                        self.graph.edges[node, neighbor]['weight'] = np.sqrt(
                            (ww-self.C[node, neighbor])**2)
            newly_active = next_active
            active.update(newly_active)
        return active

    def main(self):
        self.construct_graph()
        np.savetxt('A.csv', nx.adjacency_matrix(
            self.graph).toarray(), delimiter=',')


def pairwise_constrain(node_pair, cluster):
    return int(cluster[node_pair[0]] == cluster[node_pair[1]])


def get_pairwise_constrain_matrix(datasize, num_constraints, cluster):
    pairs = np.array(list(combinations(range(datasize), 2)))
    pairs = shuffle(pairs, random_state=42)
    node_pair = pairs[:num_constraints]
    pairwise_constrain_matrix = np.zeros((datasize, datasize))

    for i in range(len(node_pair)):
        pairwise_constrain_matrix[node_pair[i][0], node_pair[i][1]] = int(
            cluster[node_pair[i][0]] == cluster[node_pair[i][1]])
        pairwise_constrain_matrix[node_pair[i][1], node_pair[i][0]] = int(
            cluster[node_pair[i][0]] == cluster[node_pair[i][1]])
    return pairwise_constrain_matrix


def preprocess_data(dataset_path: str):
    df = pd.read_csv(dataset_path, header=None)
    # drop the duplicate rows
    df = df.drop_duplicates()
    return df


def get_data(dataset_path: str):
    df = preprocess_data(dataset_path)
    cluster_lable = df.iloc[:, -1].values
    k = np.unique(cluster_lable)
    k = k.shape[0]
    data = df.iloc[:, :-1].values
    return data, cluster_lable, k


def find_isolated_nodes(adjacency_matrix):
    isolated_nodes = []
    n = len(adjacency_matrix)
    for i in range(n):
        # 如果节点 i 的度数为 0，则将其添加到孤立节点列表中
        if np.sum(adjacency_matrix[i]) == 0:
            isolated_nodes.append(i)
    return isolated_nodes


def draw_graph_with_nx(graph, node_color='blue', node_size=100, with_labels=True):
    plt.figure(figsize=(8, 6))
    nx.draw(graph, pos=nx.spring_layout(graph), node_size=node_size, node_color=node_color,
            with_labels=with_labels, cmap=plt.cm.tab10)
    plt.show()


def spectral_clustering(n_clusters, similarity_matrix=None, data=None):
    # 计算相似性矩阵
    # similarity_matrix = pairwise_distances(data, metric='euclidean')

    # 计算拉普拉斯矩阵
    laplacian_matrix = laplacian(similarity_matrix, normed=True)

    # 计算拉普拉斯矩阵的特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(laplacian_matrix)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real

    # 对特征值进行排序，并获取前n_clusters个特征向量
    indices = np.argsort(eigenvalues)[:n_clusters]
    chosen_eigenvectors = eigenvectors[:, indices]

    # 对选定的特征向量进行K-means聚类
    kmeans = KMeans(n_clusters=n_clusters).fit(chosen_eigenvectors)

    return kmeans.labels_


def evaluate_clustering(cluster_labels, true_labels):
    accuracy = np.sum(cluster_labels == true_labels) / len(true_labels)
    ari = adjusted_rand_score(true_labels, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)
    return accuracy, ari, nmi
    # plt.savefig(f'./graph{time.time()}.png')


def run(listdir):
    data_name = None
    for i in listdir:
        print(i)
        # split the file name and the extension
        data_name = os.path.splitext(i)[0]
        # if the extension is .csv

        data, cluster_lable, k = get_data(f'data/{i}')
        print(f"getdata {i}", data.shape, cluster_lable.shape, k)
        # alg = alg(data, k, cluster_lable)
        # alg.construct_graph()
        # labels = spectral_clustering(k, alg.A)
        # accuracy, ari, nmi = evaluate_clustering(labels, cluster_lable)
        # print(f'Accuracy: {accuracy}')
        # print(f'ARI: {ari}')
        # print(f'NMI: {nmi}')
        # node_pair = np.array([[0, 1], [1, 2], [2, 3], [3, 4],
        #                      [4, 5], [5, 6], [6, 7], [7, 8]])
        # print(node_pair.shape)
        # print(get_pairwise_constrain_matrix(
        #     data.shape[0], node_pair, cluster_lable))
        # 生成1-100的数间隔为10
        # a = np.ceil(np.arange(0.1, 1.1, 0.1)*data.shape[0])
        a = np.arange(10, 101, 10)
        aris = []
        nmis = []
        e2cp = E2CP(data)
        A = e2cp.construct_graph()
        for i in a:
            print(f"constraint number: {i}")
            e2cp.A = A
            e2cp.set_pairwise_constrain_matrix(
                get_pairwise_constrain_matrix(data.shape[0], i, cluster_lable))
            e2cp.main()
            labels = spectral_clustering(k, e2cp.A)
            accuracy, ari, nmi = evaluate_clustering(labels, cluster_lable)
            # print(f'Accuracy: {accuracy}')
            # print(f'ARI: {ari}')
            # print(f'NMI: {nmi}')
            aris.append(ari)
            nmis.append(nmi)
        # 画折线图
        plt.clf()
        plt.plot(a, aris, label='ARI')
        plt.plot(a, nmis, label='NMI')
        plt.xlabel('Number of pairwise constraints')
        plt.ylabel('ARI and NMI')
        plt.title(f'ARI and NMI on {data_name}')
        plt.legend()
        plt.savefig(f'./graph/{data_name}.png')

        arange = np.arange(0.01, 0.11, 0.01)
        a = np.ceil(arange * data.shape[0]*(data.shape[0]-1)/2).astype(int)
        aris = []
        nmis = []
        e2cp = E2CP(data)
        for i in a:
            print(f"constraint number: {i}")
            e2cp.A = A
            e2cp.set_pairwise_constrain_matrix(
                get_pairwise_constrain_matrix(data.shape[0], i, cluster_lable))
            e2cp.main()
            labels = spectral_clustering(k, e2cp.A)
            accuracy, ari, nmi = evaluate_clustering(labels, cluster_lable)
            # print(f'Accuracy: {accuracy}')
            # print(f'ARI: {ari}')
            # print(f'NMI: {nmi}')
            aris.append(ari)
            nmis.append(nmi)
        # 画折线图
        plt.clf()
        plt.plot(arange, aris, label='ARI')
        plt.plot(arange, nmis, label='NMI')
        plt.xlabel('percentage of pairwise constraints')
        plt.ylabel('ARI and NMI')
        plt.title(f'ARI and NMI on {data_name}')
        plt.legend()
        plt.savefig(f'./graph/{data_name}_p.png')


if __name__ == '__main__':
    dataset_path = 'data/iris.csv'
    # list all the files in the data directory
    print(os.listdir('data'))
    listdir = ['balance.csv', 'banknote.csv', 'breast.csv', 'dermatology.csv', 'diabetes.csv', 'ecoli.csv', 'glass.csv', 'haberman.csv', 'ionosphere.csv', 'iris.csv',
               'led.csv', 'mfeat_karhunen.csv', 'mfeat_zernike.csv', 'musk.csv', 'pima.csv', 'seeds.csv', 'segment.csv', 'soybean.csv', 'thyroid.csv', 'vehicle.csv', 'wine.csv']
    data_name = None
    data, cluster_lable, k = get_data(dataset_path)
    ic = IC(data)
    ic.main()

    # plt.plot(range(1, 11), wcss)

    # print(alg.A.shape, alg.A[0])
    # print(data.shape)
    # print(cluster.shape)
    # print(k)
