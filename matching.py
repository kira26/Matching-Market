"""
Filename: matching.py
Authors: Yoshimasa Ogawa
LastModified: 23/06/2015

A collection of functions to solve the matching problems.
"""
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import random


class Matching:

    def __init__(self, prop_prefs, resp_prefs, caps=None):
        """
        Compute matching problems by Deferred Acceptance Algorithm.

        Parameters
        ----------
        prop_prefs : ndarray(int, ndim=2)
            Preference of proposers

        resp_prefs : ndarray(int, ndim=2)
            Preference of respondants

        caps : list
            Capacity of respondants

        """
        self.prop_prefs = np.asarray(prop_prefs)
        self.resp_prefs = np.asarray(resp_prefs)
        self.prop_num = self.prop_prefs.shape[0]
        self.resp_num = self.resp_prefs.shape[0]
        self.prop_unmatched = self.resp_num
        self.resp_unmatched = self.prop_num
        self.resp_ranks = np.argsort(resp_prefs)
        self.switch = 1
        if caps is None:
            self.switch = 0
            caps = [1 for col in range(self.resp_num)]
        self.caps = caps
        self.prop_matched = np.zeros(self.prop_num, dtype=int) + self.prop_unmatched
        self.resp_matched = np.zeros(sum(self.caps), dtype=int) + self.resp_unmatched
        self.prop_name = range(self.prop_num)
        self.resp_name = range(self.resp_num)
        self.indptr = np.zeros(self.resp_num+1, dtype=int)
        self.algo = None
        np.cumsum(caps, out=self.indptr[1:])

    def DA(self):
        """
        Get the stable matching pairs by Deffered Acceptance Algorithm.

        Return
        ------
        One-to-One Matching ( switch is 0 ):

            prop_matched : ndarray(int, ndim=1)
                Stable Matching Pairs in which index is
                the number of proposers

            resp_matched : ndarray(int, ndim=1)
                Stable Matching Pairs in which index is
                the number of respondants


        Many-to-One Matching ( switch is 1 ):

            prop_matched : ndarray(int, ndim=1)
                Stable Matching Pairs in which index is
                the number of proposers

            resp_matched : ndarray(int, ndim=1)
                Stable Matching Pairs

            indptr : ndarray(int, ndim=1)
                Array that shows which respondants match
                with proposers in resp_matched

        """
        self.algo = "Deffered Acceptance"
        self.prop_matched = np.zeros(self.prop_num, dtype=int) + self.prop_unmatched
        self.resp_matched = np.zeros(sum(self.caps), dtype=int) + self.resp_unmatched
        prop_single = range(self.prop_num)
        caps_rest = [i for i in self.caps]

        while len(prop_single) >= 1:
            for i in prop_single:
                prop_id = i
                for j in self.prop_prefs[prop_id]:
                    resp_id = j
                    if resp_id == self.prop_unmatched:
                        prop_single.remove(prop_id)
                        self.prop_matched[prop_id] = self.prop_unmatched
                        break
                    elif caps_rest[resp_id] >= 1:
                        if self.resp_unmatched in self.resp_ranks[resp_id]:
                            if self.resp_ranks[resp_id][prop_id] < self.resp_ranks[resp_id][self.resp_unmatched]:
                                prop_single.remove(prop_id)
                                self.prop_matched[prop_id] = resp_id
                                caps_rest[resp_id] -= 1
                                self.resp_matched[self.indptr[resp_id]+caps_rest[resp_id]] = prop_id
                                break
                        else:
                            prop_single.remove(prop_id)
                            self.prop_matched[prop_id] = resp_id
                            caps_rest[resp_id] -= 1
                            self.resp_matched[self.indptr[resp_id]+caps_rest[resp_id]] = prop_id
                            break
                    else:
                        deffered = self.resp_matched[self.indptr[resp_id]:self.indptr[resp_id+1]]
                        max_rank = max([self.resp_ranks[resp_id][i] for i in deffered])
                        max_id = self.resp_prefs[resp_id][max_rank]
                        if self.resp_ranks[resp_id][prop_id] < max_rank:
                            prop_single.append(max_id)
                            prop_single.remove(prop_id)
                            self.prop_matched[max_id] = self.prop_unmatched
                            self.prop_matched[prop_id] = resp_id
                            self.resp_matched[np.where(self.resp_matched == max_id)[0]] = prop_id
                            break
                if prop_id in prop_single:
                    prop_single.remove(prop_id)
                    self.prop_matched[prop_id] = self.prop_unmatched
        if self.switch == 0:
                return self.prop_matched, self.resp_matched
        else:
            return self.prop_matched, self.resp_matched, self.indptr

    def TTC(self):
        """
        Get the stable matching pairs by Top Trading Cycle Algorithm.
        """
        self.algo = "Top Trading Cycle"
        self.prop_matched = np.zeros(self.prop_num, dtype=int) + self.prop_unmatched
        self.resp_matched = np.zeros(sum(self.caps), dtype=int) + self.resp_unmatched
        prop_single = range(self.prop_num)
        caps_rest = [i for i in self.caps]

        def make_unmatched(prop_id):
            prop_single.remove(prop_id)
            self.prop_matched[prop_id] = self.prop_unmatched

        while len(prop_single) >= 1:
            prop_approach = [0 for i in range(self.prop_num)]
            resp_approach = [0 for i in range(self.resp_num)]
            prop_id = prop_single[0]
            prop_circle = [0 for i in range(self.prop_num)]
            trading_chain = [[],[]]
            trading_chain[0].append(prop_id)
            prop_circle[prop_id] += 1
            while True:
                if prop_approach[prop_id] > len(self.prop_prefs):
                    make_unmatched(prop_id)
                    success = False
                    break
                resp_id = self.prop_prefs[prop_id][prop_approach[prop_id]]
                prop_approach[prop_id] += 1
                if resp_id == self.prop_unmatched:
                    make_unmatched(prop_id)
                    success = False
                    break
                elif caps_rest[resp_id] >= 1:
                    next_prop_id = resp_to_prop(resp_id, resp_approach, self.resp_unmatched, self.resp_prefs, prop_single)
                    if next_prop_id != self.resp_unmatched:
                        prop_id = next_prop_id
                        trading_chain[1].append(resp_id)
                        if prop_circle[prop_id] == 1:
                            success = True
                            break
                        else:
                            trading_chain[0].append(prop_id)
                            prop_circle[prop_id] += 1
                            pass
                    else:
                        pass
                else:
                    pass
            if success:
                ptr = trading_chain[0].index(prop_id)
                trading_chain[0] = trading_chain[0][ptr:]
                trading_chain[1] = trading_chain[1][ptr:]
                for i in range(len(trading_chain[0])):
                    prop_id = trading_chain[0][i]
                    resp_id = trading_chain[1][i]
                    prop_single.remove(prop_id)
                    self.prop_matched[prop_id] = resp_id
                    caps_rest[resp_id] -= 1
                    self.resp_matched[self.indptr[resp_id]+caps_rest[resp_id]] = prop_id
        if self.switch == 0:
            return self.prop_matched, self.resp_matched
        else:
            return self.prop_matched, self.resp_matched, self.indptr



    def summary(self):
        print "-----Initial Setting-----"
        print "The number of proposers : %s" % self.prop_num
        print "The number of respondants : %s" % self.resp_num
        print "Preference of proposers : (unmatched = %s)" % self.prop_unmatched
        print self.prop_prefs
        print "Preference of respondants : (unmatched = %s)" % self.resp_unmatched
        print self.resp_prefs
        if self.switch == 1:
            print "Capacity of respondants : "
            print self.caps
        print "-----Matching Type-----"
        if self.switch == 0:
            print "One-to-One Matching"
        else:
            print "Many-to-One Matching"
        print self.algo
        print "-----Result-----"
        print "Stable matching pairs"
        print "Proposers side : "
        print self.prop_matched
        print "Respondants side : "
        print self.resp_matched
        if self.switch == 1:
            print "indptr : "
            print self.indptr


    def set_name(self, prop_name, resp_name):
        self.prop_name = prop_name
        self.resp_name = resp_name


    def graph(self):
        """
        Print a graph about the matching pair made by match().
        Lables of male are blue and that of female are red.
        """
        # Set the variables
        vector = {}
        prop_vector = {}
        resp_vector = {}
        pos = {}
        prop_pos = {}
        resp_pos = {}
        height = max(self.prop_num, self.resp_num)
        for i in range(self.prop_num):
            vector["m%s" % i] = []
            prop_vector[self.prop_name[i]] = []
            pos["m%s" % i] = np.array([1, height-i])
            prop_pos[self.prop_name[i]] = np.array([1, height-i])
            if self.prop_matched[i] != self.prop_unmatched:
                vector["m%s" % i].append("f%s" % self.prop_matched[i])
        for i in range(self.resp_num):
            vector["f%s" % i] = []
            resp_vector[self.resp_name[i]] = []
            pos["f%s" % i] = np.array([2, height-i])
            resp_pos[self.resp_name[i]] = np.array([2, height-i])
        graph = nx.Graph(vector)
        m_graph = nx.Graph(prop_vector)
        f_graph = nx.Graph(resp_vector)
        nx.draw_networkx_labels(m_graph, prop_pos, font_size=28, font_color="b")
        nx.draw_networkx_labels(f_graph, resp_pos, font_size=28, font_color="r")
        nx.draw_networkx_nodes(graph, pos, node_size=1000, node_color="w")
        nx.draw_networkx_edges(graph, pos, width=1)
        plt.xticks([])
        plt.yticks([])
        plt.show


def resp_to_prop(resp_id, resp_approach, resp_unmatched, resp_prefs, prop_single):
    while True:
        if resp_approach[resp_id] > len(resp_prefs):
            return resp_unmatched
        prop_id = resp_prefs[resp_id][resp_approach[resp_id]]
        if prop_id == resp_unmatched:
            return resp_unmatched
        elif prop_id in prop_single:
            return prop_id
        else:
            resp_approach[resp_id] += 1
            pass

def deferred_acceptance(prop_prefs, resp_prefs, caps=None):
    test = Matching(prop_prefs, resp_prefs, caps)
    if caps is None:
        prop_matched, resp_matched = test.get_pairs()
        return prop_matched, resp_matched
    else:
        prop_matched, resp_matched, indptr = test.get_pairs()
        return prop_matched, resp_matched, indptr
