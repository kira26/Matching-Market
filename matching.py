"""
Filename: matching.py
Authors: Yoshimasa Ogawa
LastModified: 16/06/2015

A collection of functions to solve the matching problems.
"""

import networkx
import matplotlib.pyplot as plt
import numpy as np
import random


def acceptable(m_id, f_id, f_prefs):
    """
    Check the partner is acceptable.
    Returns True or False.
    """
    if (-1 in f_prefs[f_id]):
        if f_prefs[f_id].index(m_id) < f_prefs[f_id].index(-1):
            return True
        else:
            return False
    else:
        return True

def compare(m_id, f_id, f_prefs, f_matched):
    """
    Check the partner is better than ever.
    Returns True or False.
    """
    if f_prefs[f_id].index(m_id) < f_prefs[f_id].index(f_matched[f_id]):
        return True
    else:
        return False

def make_prefs(m_num, f_num):
    m_prefs = [range(-1, f_num) for col in range(m_num)]
    f_prefs = [range(-1, m_num) for col in range(f_num)]
    for i in range(m_num):
        random.shuffle(m_prefs[i])
    for i in range(f_num):
        random.shuffle(f_prefs[i])
    return m_prefs, f_prefs

def deferred_acceptance(m_prefs, f_prefs):
    """
    For test_matching.py.
    Return 2 lists that include male, female pair information.
    """
    test = Marriage()
    m_matched, f_matched = test.match(m_prefs, f_prefs)
    return m_matched, f_matched

class Marriage:
    """
    Solve one-to-one matching problems by Deferred Acceptance Algorithm.
    match() returns 2 lists, m_matched and f_matched.
    graph() print a graph about the matching pair made by match().
    """
    def match(self, m_prefs, f_prefs):
        self.m_prefs = m_prefs
        self.f_prefs = f_prefs
        self.m_num = len(m_prefs)
        self.f_num = len(f_prefs)
        self.m_matched = [None for col in range(self.m_num)]
        self.f_matched = [None for col in range(self.f_num)]
        self.m_name = range(self.m_num)
        self.f_name = range(self.f_num)
        m_single = range(self.m_num)
        f_single = range(self.f_num)
        while len(m_single) >= 1:
            for i in range(self.m_num):
                if i in m_single:
                    m_id = i
                    for j in range(len(self.m_prefs[i])):
                        f_id = self.m_prefs[m_id][j]
                        if f_id < 0:
                            m_single.remove(i)
                            self.m_matched[m_id] = -1
                            break
                        elif f_id in f_single:
                            if acceptable(m_id, f_id, f_prefs):
                                m_single.remove(m_id)
                                f_single.remove(f_id)
                                self.m_matched[m_id] = f_id
                                self.f_matched[f_id] = m_id
                                break
                        else:
                            if compare(m_id, f_id, f_prefs, self.f_matched):
                                m_single.append(self.f_matched[f_id])
                                self.m_matched[self.f_matched[f_id]] = -1
                                m_single.remove(i)
                                self.m_matched[m_id] = f_id
                                self.f_matched[f_id] = m_id
                                break
                    if m_id in m_single:
                        m_single.remove(i)
                        self.m_matched[m_id] = -1
        for k in range(self.f_num):
            if k in f_single:
                f_single.remove(k)
                self.f_matched[k] = -1
        return self.m_matched, self.f_matched

    def set_name(self, m_name, f_name):
        self.m_name = m_name
        self.f_name = f_name

    def graph(self):
        """
        Print a graph about the matching pair made by match().
        Lables of male are blue and that of female are red.
        """
        vector = {}
        m_vector = {}
        f_vector = {}
        pos = {}
        m_pos = {}
        f_pos = {}
        for i in range(self.m_num):
            vector["m%s" % i] = []
            m_vector[self.m_name[i]] = []
            pos["m%s" % i] = np.array([1, self.m_num-i])
            m_pos[self.m_name[i]] = np.array([1, self.m_num-i])
            if self.m_matched[i] != -1:
                vector["m%s" % i].append("f%s" % self.m_matched[i])
        for i in range(self.f_num):
            vector["f%s" % i] = []
            f_vector[self.f_name[i]] = []
            pos["f%s" % i] = np.array([2, self.f_num-i])
            f_pos[self.f_name[i]] = np.array([2, self.f_num-i])
            if self.f_matched[i] != -1:
                vector["f%s" % i].append("m%s" % self.f_matched[i])
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
