# coding: UTF-8
import networkx
import matplotlib.pyplot as plt
import numpy as np


class Marriage:
    def __init__(self, malenum = 0, femalenum = 0):
        self.malenum = malenum
        self.femalenum = femalenum
        self.mprefer = [[None for col in range(femalenum)] for row in range(malenum)]
        self.fprefer = [[None for col in range(malenum)] for row in range(femalenum)]
        self.mname = range(malenum)
        self.fname = range(femalenum)

    def set_num(malenum, femalenum):
        self.malenum = malenum
        self.femalenum = femalenum

    def set_prefer(self, mprefer, fprefer):
        self.mprefer = mprefer
        self.fprefer = fprefer

    def set_name(self, mname, fname):
        self.mname = mname
        self.fname = fname

    def match(self):
        m_single = range(self.malenum)
        f_single = range(self.femalenum)
        self.married = {}
        while len(m_single) >= 1:
            for i in range(self.malenum):
                if i in m_single:
                    for j in range(len(self.mprefer[i])):
                        femaleid = self.mprefer[i][j]
                        femaleallid = self.mprefer[i][j] + self.malenum
                        if femaleid < 0:
                            m_single.remove(i)
                            self.married.update({self.mname[i]: 0})
                            break
                        if femaleid in f_single:
                            if i in self.fprefer[femaleid]:
                                m_single.remove(i)
                                f_single.remove(femaleid)
                                self.married.update({self.fname[femaleid]: self.mname[i]})
                                self.married.update({self.mname[i]: self.fname[femaleid]})
                                break
                        elif i in self.fprefer[femaleid]:
                            if self.fprefer[femaleid].index(self.mname.index(self.married[self.fname[femaleid]])) > self.fprefer[femaleid].index(i):
                                m_single.append(self.mname.index(self.married[self.fname[femaleid]]))
                                m_single.remove(i)
                                self.married.update({self.married[self.fname[femaleid]]: 0})
                                self.married.update({self.fname[femaleid]: self.mname[i]})
                                self.married.update({self.mname[i]: self.fname[femaleid]})
                                break
                    if i in m_single:
                        m_single.remove(i)
                        self.married.update({self.mname[i]: 0})
        for k in range(self.femalenum):
            if k in f_single:
                f_single.remove(k)
                self.married.update({self.fname[k]: 0})
        return  self.married

    def graph(self):
        vector = {}
        m_vector = {}
        f_vector = {}
        pos = {}
        m_pos = {}
        f_pos = {}
        for i in self.mname:
            vector[i] = []
            m_vector[i] = []
            if self.married[i] != 0:
                vector[i].append(self.married[i])
        for i in self.fname:
            vector[i] = []
            f_vector[i] = []
            if self.married[i] != 0:
                vector[i].append(self.married[i])
        counter = 0
        for i in self.mname:
            j = len(self.mname)
            pos[i] = np.array([1, j-counter])
            m_pos[i] = np.array([1, j-counter])
            counter += 1
        counter = 0
        for i in self.fname:
            j = len(self.fname)
            pos[i] = np.array([2, j-counter])
            f_pos[i] = np.array([2, j-counter])
            counter += 1
        graph = networkx.Graph(vector)
        m_graph = networkx.Graph(m_vector)
        f_graph = networkx.Graph(f_vector)
        networkx.draw_networkx_labels(m_graph, m_pos, font_size=28, font_color="b")
        networkx.draw_networkx_labels(f_graph, f_pos, font_size=28, font_color="r")
        networkx.draw_networkx_nodes(graph, pos, node_size=1000, node_color="w")
        networkx.draw_networkx_edges(graph, pos, width=1)
        plt.xticks([])
        plt.yticks([])
        plt.show
