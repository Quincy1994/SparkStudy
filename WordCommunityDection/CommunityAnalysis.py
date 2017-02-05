# coding=utf-8

from igraph import *

""" 设置点的颜色，默认是红色"""
color_dict = {0: "pink", 1: "green",2:"purple",3:"orange",4:"blue",5:"yellow",6:"red",7:"#8B2500",8:"#87CEEB",9:"#707070",
              10:"#FFF68F",11:"#FFEFD5",12:"#FFE4E1",13:"#FFDEAD",14:"#FFC1C1",15:"#FFB90F",16:"#FFA54F",17:"#FF8C00",
              18:"black",19:"#FF6EB4",20:"#FF4500",21:"#FF3030",22:"#F5DEB3",23:"#F0FFFF",24:"#F08080",25:"#EED2EE",26:"#EECFA1",
              27:"#EECBAD",28:"#EEC900",29:"#DDA0DD",30:"#E3E3E3",31:"#DB7093",32:"#D8BFD8",33:"#D2B48C",34:"#CDCDB4",
              35:"#CDAD00",36:"#CD853F",37:"#CD5555",38:"#CAE1FF",39:"#BCEE68",40:"#A0522D",41:"#AEEEEE",42:"#9AFF9A",
              43:"#B03060",44:"#8B6508",45:"#8B475D",46:"#8B1A1A",47:"#836FFF",48:"#7A378B",49:"#76EEC6",50:"#698B69"}


class Node:
    id = None
    name = None
    size = None
    color = None

    def __init__(self):
        return

    def set_id(self, id):
        self.id = id

    def get_id(self):
        return self.id

    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def set_size(self, size):
        self.size = size

    def get_size(self):
        return self.size

    def set_color(self, color):
        self.color = color

    def get_color(self):
        return self.color

class ComminityAnalysis:

    def __init__(self, filename):
        self.vocabulary, self.flen = self.loadData(filename)
        self.theta = self.get_theta()
        self.terms = self.vocabulary.keys()
        self.g = Graph(1)

    def get_theta(self):
        if self.flen < 10:
            theta = 0
        elif self.flen < 100:
            theta = 1
        elif self.flen < 1000:
            theta = self.flen * 0.05
        else:
            theta = 50
        return theta

    def loadData(self, filename):
        sentences = open(filename).readlines()
        flen = sentences.__len__()
        vocabulary = dict()
        for sentence in sentences:
            tokens = sentence.split(':')
            word = tokens[0]
            reverse_index = tokens[1].replace("[", "").replace("]", "").replace(" ", "").replace("\n","").split(",")
            vocabulary[word] = reverse_index
        return vocabulary, flen

    def create_graph(self, graph_dir):
        node_list = []
        colorType = 0
        if len(self.terms) - 1 <= 0:
            f = open(graph_dir + "/node.txt", 'w')
            f.close()
            f = open(graph_dir + "/edge.txt", 'w')
            f.close()
            return
        self.g.add_vertices(len(self.terms) -1)
        self.g.vs["label"] = self.terms
        edges = []
        weights = []
        length = len(self.terms)
        matrix = [[0 for col in range(length)] for row in range(length)]
        for i in range(0, len(self.terms), 1):
            for j in range(0, len(self.terms), 1):
                v1 = self.terms[i]
                v2 = self.terms[j]
                union = set(self.vocabulary[v1]) & set(self.vocabulary[v2])
                mix = union.__len__()
                matrix[i][j] = mix
                # print "create OK"
                if mix > self.theta:
                    edges += [(i, j)]
                    weights.append(mix)
        self.g.add_edges(edges)
        self.g = self.g.simplify()

        result = self.g.community_multilevel(weights)
        print result
        listresult = [0*i for i in range(self.terms.__len__())]
        length = result.__len__()
        if length < 10:
            comunity_theta = 3
        elif length < 20:
            comunity_theta = 4
        elif length < 30:
            comunity_theta = 5
        else:
            comunity_theta = 6

        for i in range(0, result.__len__(), 1):
            if result[i].__len__() <= 1:
                continue
            if result[i].__len__() > 10:
                sublist = result[i]
                subg = Graph(1)
                sublen = result[i].__len__()
                subg.add_vertices(sublen - 1)
                edges = []
                weights = []
                for a in range(0, sublen, 1):
                    for b in range(0, sublen, 1):
                        v1 = sublist[a]
                        v2 = sublist[b]
                        weight = matrix[v1][v2]
                        if weight > self.theta * 1.5:
                            edges += [(a, b)]
                            weights.append(weight)
                subg.add_edges(edges)
                subg = subg.simplify()
                subresult = subg.community_multilevel(weights)
                for a in range(0, subresult.__len__(), 1):
                    if subresult[a].__len__() < comunity_theta:
                        continue
                    for b in range(0, subresult[a].__len__(), 1):
                        node = Node()
                        v = sublist[subresult[a][b]]
                        print self.terms[v], ":", matrix[v][v],
                        node.set_id(v)
                        node.set_name(self.terms[v])
                        node.set_size(matrix[v][v])
                        node.set_color(color_dict[colorType % 50])
                        node_list.append(node)
                    print
                    colorType += 1
                continue

            for j in range(0, result[i].__len__(), 1):
                print self.terms[result[i][j]], ":", matrix[result[i][j]][result[i][j]],
                node = Node()
                node.set_id(result[i][j])
                node.set_name(self.terms[result[i][j]])
                node.set_size(matrix[result[i][j]][result[i][j]])
                node.set_color(color_dict[colorType % 50])
                node_list.append(node)
                listresult[result[i][j]] = i

            colorType += 1
            print

        data = ""
        for node in node_list:
            print node.id, ",", node.name, ",", node.size, ",", node.color
            print
            data += str(node.id) + "," + str(node.name) + "," + str(node.size) + "," + str(node.color) + "\n"
        # print graph_dir
        print data
        f = open(graph_dir + "/node.txt", 'w')
        f.write(data)
        f.close()

        data = ""
        for v1 in range(0, len(node_list)-1 , 1):
            for v2 in range(v1+1, len(node_list), 1):
                    data += str(v1) + "," + str(v2) + "," + str(matrix[v1][v2]) + "\n"

        f = open(graph_dir + "/edge.txt", 'w')
        f.write(data)
        f.close()
        # p = Plot()
        # p.background = "#ffffff"  # 将背景改为白色，默认是灰色网格
        # p.add(self.g,
        #       bbox=(50, 50, 550, 550),  # 设置图占窗体的大小，默认是(0,0,600,600)
        #       layout=layout,  # 图的布局
        #       edge_width=0.5, edge_color="grey",  # 边的宽度和颜色，建议灰色，比较好看
        #       vertex_label_size=5, # 点标签的大小
        # vertex_color = [color_dict[i % 50] for i in listresult])  # 为每个点着色
        # # p.save("/home/quincy/SNA.png")  # 将图保存到特定路径，igraph只支持png和pdf
        # p.show()
        # p.remove(self.g)  # 清除图像


def train(reverseIndexFile):
    # filename = "/home/quincy1994/实验室项目/影迷关注点分析/实验数据/ReverseIndex/56b1c0d571ba2d607f0cb709.txt"
    tokens = str(reverseIndexFile).split("/")
    id = tokens[-1].replace(".txt","")
    print id
    filedir = "/home/quincy1994/实验室项目/影迷关注点分析/实验数据/GraphData/"
    graph_dir = filedir + id

    if os.path.exists(graph_dir):
        return
    if not os.path.exists(graph_dir):
        os.mkdir(graph_dir)
    ca = ComminityAnalysis(reverseIndexFile)
    ca.create_graph(graph_dir)


def main():
    file_dir = "/home/quincy1994/实验室项目/影迷关注点分析/实验数据/ReverseIndex/"
    for parent, dirnames, filenames in os.walk(file_dir):
        for filename in filenames:
            # filename = "5645c95a756a5d75424ca124"
            train(file_dir + filename)
            # break

if __name__ == '__main__':
    main()