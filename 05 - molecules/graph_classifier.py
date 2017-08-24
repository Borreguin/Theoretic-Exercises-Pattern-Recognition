from lxml import etree
from munkres import Munkres
import timeit
import csv


class node:

    def __init__(self, node_id, symbol):
        self.node_id = node_id
        self.symbol = symbol
        self.edges = []

    def get_id(self):
        return self.node_id

    def get_symbol(self):
        return self.symbol

    def add_edge(self, edge):
        self.edges.append(edge)

    def number_of_edge(self):
        return len(self.edges)


class edge:

    def __init__(self, from_node, to_node, value):
        self.from_node = from_node
        self.to_node = to_node
        self.value = value

    def get_from(self):
        return self.from_node

    def get_to(self):
        return self.to_node

    def get_value(self):
        return self.value


class molecule:

    def __init__(self, graph_id, m_class):
        self.graph_id = graph_id
        self.m_class = m_class

    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.edges

    def get_id(self):
        return self.graph_id

    def set_graph(self, file_name):
        self.nodes = {}
        self.edges = []
        tree = etree.parse(file_name)
        self.set_nodes(tree)
        self.set_edges(tree)

    def set_nodes(self, tree):
        for xml_node in tree.xpath("/gxl/graph/node"):
            node_id = xml_node.get("id")
            node_symbol = xml_node.getchildren()[0].getchildren()[0].text
            self.nodes[node_id] = node(node_id, node_symbol)

    def set_edges(self, tree):
        for xml_edge in tree.xpath("/gxl/graph/edge"):
            from_node = xml_edge.get("from")
            to_node = xml_edge.get("to")
            value = xml_edge.findtext("attr/int")
            new_edge = edge(from_node, to_node, value)
            self.edges.append(new_edge)
            self.nodes[from_node].add_edge(new_edge)
            self.nodes[to_node].add_edge(new_edge)

    def get_class(self):
        return self.m_class

    def show(self):
        print("molecule : ", self.graph_id, " class : ", self.m_class, "\n")
        for node in self.nodes:
            print("Node ", node[0])
            print(node[1], "\n")
        for edge in self.edges:
            print("Edge from : ", edge.get_from())
            print(" to : ", edge.get_to())
            print(" value : ", edge.get_value())


class graph_classifier:

    def __init__(self):
        self.training_set = []
        self.testing_set = []
        self.m = Munkres()

    def train(self, file_name):
        self.file_to_set(file_name, self.training_set)

    def set_testing_set(self, file_name):
        self.testing_set_name = file_name.split('.')[0]
        self.file_to_set(file_name, self.testing_set)

    def file_to_set(self, file_name, set_a):
        print("Load file : "+file_name+" start")
        with open(file_name, "r") as fichier:
            for content in fichier:
                content = content.splitlines()[0].split(" ")
                sample = molecule(content[0], content[1])
                sample.set_graph("gxl/"+content[0]+".gxl")
                set_a.append(sample)
        print("Load file : "+file_name+" end")

    def build_cost_matrix(self, graph_a, graph_b):
        cn = 1  # Node deletion/insertion cost
        ce = 1  # Edge deletion/insertion cost
        cs = 2 * cn  # Node substitution cost ( if symbols ≠, 0 )

        nodes_a = list(graph_a.get_nodes().values())
        nodes_b = list(graph_b.get_nodes().values())

        n = len(nodes_a)
        m = len(nodes_b)
        matrix_size = (n + m)
        matrix = [[0 for x in range(matrix_size)] for y in range(matrix_size)]

        # Upper left
        for i in range(0, n):
            for j in range(0, m):
                if(nodes_a[i].get_symbol() != nodes_b[j].get_symbol()):
                    matrix[i][j] = cs

        # Upper right
        for i in range(0, n):
            for j in range(m, matrix_size):
                if(i == (j - m)):
                    matrix[i][j] = cn + (ce * nodes_a[i].number_of_edge())
                else:
                    matrix[i][j] = float('inf')

        # Lower left
        for i in range(n, matrix_size):
            for j in range(0, m):
                if((i - n) == j):
                    matrix[i][j] = cn + (ce * nodes_b[j].number_of_edge())
                else:
                    matrix[i][j] = float('inf')
        return matrix

    def get_cost(self, matrix, max_cost):
        indexes = self.m.compute(matrix)
        total = 0
        for row, column in indexes:
            value = matrix[row][column]
            total += value
            if(total > max_cost):
                return False
        return total

    def flush_accuracy(self):
        self.correct = 0.0
        self.incorrect = 0.0
        self.accuracy = 0.0

    def show_accuracy(self):
        print("Accuracy of the model : ", self.accuracy*100)

    def classify(self, NN):
        # print("NN : ", NN)
        classes, costs = zip(*NN)
        return most_common(classes)

    def update_accuracy(self, predicted, real):
        # print("Predicted as : ", predicted, " Real class is : ", real)
        if(predicted == real):
            self.correct = self.correct + 1.0
        else:
            self.incorrect = self.incorrect + 1.0
        self.accuracy = (self.correct/(self.correct+self.incorrect))

    def knn(self, k):
        self.flush_accuracy()
        c = csv.writer(open(self.testing_set_name+"_"+str(k)+".csv", "w"), delimiter=',')

        for test_sample in self.testing_set:
            print("Classify sample : ", test_sample.get_id())
            start = timeit.default_timer()
            NN = []
            max_cost = 1000

            for train_sample in self.training_set:
                cost = self.get_cost(self.build_cost_matrix(test_sample, train_sample), max_cost)
                if (len(NN) == 0) or (cost is not False):
                    NN = self.test_n(NN, (train_sample.get_class(), cost), k)
                    max_cost = NN[0][1]
            stop = timeit.default_timer()
            print("Classified in : ", stop - start)

            predicted_class = self.classify(NN)
            c.writerow([test_sample.get_id(), predicted_class])
            if(test_sample.get_class() != '?'):
                self.update_accuracy(predicted_class, test_sample.get_class())
                self.show_accuracy()
        print("Classification finished")

    def test_n(self, nn_list, n, k):
        if len(nn_list) < k:
            nn_list.append(n)
            if (len(nn_list) == k) and k != 1:
                nn_list_sorted = sorted(nn_list, key=getKey, reverse=True)
                return nn_list_sorted
            else:
                return nn_list
        else:
            if n[1] < nn_list[0][1]:
                nn_list[0] = n
                nn_list_sorted = sorted(nn_list, key=getKey, reverse=True)
                return nn_list_sorted
            else:
                return nn_list


# Used for sorting list of tuples
def getKey(item):
    return item[1]


# Return the most common number in a list
def most_common(lst):
    return max(set(lst), key=lst.count)


my_classifier = graph_classifier()
my_classifier.train("train.txt")
my_classifier.set_testing_set("test_15.txt")
my_classifier.knn(3)
