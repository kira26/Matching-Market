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

    def __init__(self, prop_prefs, resp_prefs, resp_caps=None, prop_caps=None):
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


        Algorithm Functions
        Return
        ------
        One-to-One Matching ( switch is 'oto' ):

            prop_matched : ndarray(int, ndim=1)
                Stable Matching Pairs in which index is
                the number of proposers

            resp_matched : ndarray(int, ndim=1)
                Stable Matching Pairs in which index is
                the number of respondants

        Many-to-One Matching ( switch is 'mtm' ):

            prop_matched : ndarray(int, ndim=1)
                Stable Matching Pairs in which index is
                the number of proposers

            resp_matched : ndarray(int, ndim=1)
                Stable Matching Pairs

            resp_indptr : ndarray(int, ndim=1)
                Array that shows which respondants match
                with proposers in resp_matched
        """
        self.prop_prefs = np.asarray(prop_prefs)
        self.resp_prefs = np.asarray(resp_prefs)
        self.prop_num = self.prop_prefs.shape[0]
        self.resp_num = self.resp_prefs.shape[0]
        self.prop_unmatched = self.resp_num
        self.resp_unmatched = self.prop_num
        self.resp_ranks = np.argsort(self.resp_prefs)
        self.switch = 'mtm'
        if prop_caps is None:
            self.switch = 'mto'
            prop_caps = [1 for col in range(self.prop_num)]
            if resp_caps is None:
                self.switch = 'oto'
                resp_caps = [1 for col in range(self.resp_num)]
        self.resp_caps = resp_caps
        self.prop_caps = prop_caps
        self.prop_matched = np.zeros(self.prop_num, dtype=int) + self.prop_unmatched
        self.resp_matched = np.zeros(sum(self.resp_caps), dtype=int) + self.resp_unmatched
        self.prop_name = range(self.prop_num)
        self.resp_name = range(self.resp_num)
        #self.indptr = np.zeros(self.resp_num+1, dtype=int)
        self.prop_indptr = np.zeros(self.prop_num+1, dtype=int)
        self.resp_indptr = np.zeros(self.resp_num+1, dtype=int)
        self.algo = None
        np.cumsum(self.prop_caps, out=self.prop_indptr[1:])
        np.cumsum(resp_caps, out=self.resp_indptr[1:])

    def DA(self):
        """
        Get the stable matching pairs by Deffered Acceptance Algorithm.
        """
        self.algo = "Deffered Acceptance"
        self.prop_matched = np.zeros(sum(self.prop_caps), dtype=int) + self.prop_unmatched
        self.resp_matched = np.zeros(sum(self.resp_caps), dtype=int) + self.resp_unmatched
        propcaps_rest = [i for i in self.prop_caps]
        respcaps_rest = [i for i in self.resp_caps]
        prop_single = range(self.prop_num)
        prop_counter = [0 for i in range(self.prop_num)]

        while len(prop_single) >= 1:
            prop_single_copy = [i for i in prop_single]
            for prop_id in prop_single_copy:
                if prop_counter[prop_id] == self.prop_unmatched:
                    prop_single.remove(prop_id)
                    break
                resp_id = self.prop_prefs[prop_id][prop_counter[prop_id]]
                prop_counter[prop_id] += 1
                if respcaps_rest[resp_id] >= 1:
                    propcaps_rest[prop_id] -= 1
                    respcaps_rest[resp_id] -= 1
                    self.prop_matched[self.prop_indptr[prop_id]+propcaps_rest[prop_id]] = resp_id
                    self.resp_matched[self.resp_indptr[resp_id]+respcaps_rest[resp_id]] = prop_id
                    if propcaps_rest[prop_id] == 0:
                        prop_single.remove(prop_id)
                else:
                    deffered = self.resp_matched[self.resp_indptr[resp_id]:self.resp_indptr[resp_id+1]]
                    max_rank = max([self.resp_ranks[resp_id][i] for i in deffered])
                    max_id = self.resp_prefs[resp_id][max_rank]
                    if self.resp_ranks[resp_id][prop_id] < max_rank:
                        prop_single.append(max_id)
                        propcaps_rest[max_id] += 1
                        propcaps_rest[prop_id] -= 1
                        self.prop_matched[np.where(self.prop_matched[self.prop_indptr[max_id]:self.prop_indptr[max_id+1]] == resp_id)[0] + self.prop_indptr[max_id]] = self.prop_unmatched
                        self.prop_matched[self.prop_indptr[prop_id]+propcaps_rest[prop_id]] = resp_id
                        self.resp_matched[np.where(self.resp_matched[self.resp_indptr[resp_id]:self.resp_indptr[resp_id+1]] == max_id)[0] + self.resp_indptr[resp_id]] = prop_id
                        if propcaps_rest[prop_id] == 0:
                            prop_single.remove(prop_id)

        if self.switch == 'oto':
            return self.prop_matched, self.resp_matched
        elif self.switch == 'mto':
            return self.prop_matched, self.resp_matched, self.resp_indptr
        else:
            return self.prop_matched, self.resp_matched, self.prop_indptr, self.resp_indptr


    def NM(self):
        """
        Get the stable matching pairs by Normal Matching.
        """
        self.algo = "Normal Matching"
        self.prop_matched = np.zeros(sum(self.prop_caps), dtype=int) + self.prop_unmatched
        self.resp_matched = np.zeros(sum(self.resp_caps), dtype=int) + self.resp_unmatched
        propcaps_rest = [i for i in self.prop_caps]
        respcaps_rest = [i for i in self.resp_caps]
        prop_single = range(self.prop_num)
        prop_counter = [0 for i in range(self.prop_num)]
        choice_num = 0

        while len(prop_single) >= 1:
            prop_single_copy = [i for i in prop_single]
            if choice_num == self.prop_prefs.shape[1]:
                break
            choices = np.histogram(self.prop_prefs[prop_single, choice_num], range(self.resp_num+1))[0]
            for resp_id in range(self.resp_num):
                if respcaps_rest[resp_id] >= choices[resp_id]:
                    for prop_id in prop_single_copy:
                        if self.prop_prefs[prop_id][choice_num] == resp_id:
                            propcaps_rest[prop_id] -= 1
                            respcaps_rest[resp_id] -= 1
                            self.prop_matched[self.prop_indptr[prop_id]+propcaps_rest[prop_id]] = resp_id
                            self.resp_matched[self.resp_indptr[resp_id]+respcaps_rest[resp_id]] = prop_id
                            if propcaps_rest[prop_id] == 0:
                                prop_single.remove(prop_id)
                elif respcaps_rest[resp_id] != 0:
                    applicants = [i for i in prop_single_copy if self.prop_prefs[i][choice_num] == resp_id]
                    for k in range(self.prop_num):
                        prop_id = self.resp_prefs[resp_id][k]
                        if prop_id in applicants:
                            propcaps_rest[prop_id] -= 1
                            respcaps_rest[resp_id] -= 1
                            self.prop_matched[self.prop_indptr[prop_id]+propcaps_rest[prop_id]] = resp_id
                            self.resp_matched[self.resp_indptr[resp_id]+respcaps_rest[resp_id]] = prop_id
                            if propcaps_rest[prop_id] == 0:
                                prop_single.remove(prop_id)
                        if respcaps_rest[resp_id] == 0:
                            break
            choice_num += 1

        if self.switch == 'oto':
            return self.prop_matched, self.resp_matched
        elif self.switch == 'mto':
            return self.prop_matched, self.resp_matched, self.resp_indptr
        else:
            return self.prop_matched, self.resp_matched, self.prop_indptr, self.resp_indptr


    def TTC(self):
        """
        Get the stable matching pairs by Top Trading Cycle.
        """
        self.algo = "Top Trading Cycle"
        self.prop_matched = np.zeros(self.prop_num, dtype=int) + self.prop_unmatched
        self.resp_matched = np.zeros(sum(self.resp_caps), dtype=int) + self.resp_unmatched
        prop_single = range(self.prop_num)
        caps_rest = [i for i in self.resp_caps]

        def make_unmatched(prop_id):
            prop_single.remove(prop_id)
            self.prop_matched[prop_id] = self.prop_unmatched

        while len(prop_single) >= 1:
            prop_approach = [0 for i in range(self.prop_num)]
            resp_approach = [0 for i in range(self.resp_num)]
            prop_circle = [0 for i in range(self.prop_num)]
            prop_id = prop_single[0]
            trading_chain = [[],[]]
            trading_chain[0].append(prop_id)
            prop_circle[prop_id] += 1
            while True:
                if prop_approach[prop_id] >= self.prop_prefs.shape[1]:
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
                    success2 = False
                    while True:
                        if resp_approach[resp_id] >= self.resp_prefs.shape[1]:
                            next_prop_id = resp_unmatched
                            success2 = True
                            break
                        prop_id = self.resp_prefs[resp_id][resp_approach[resp_id]]
                        resp_approach[resp_id] += 1
                        if prop_id in prop_single:
                            trading_chain[1].append(resp_id)
                            if prop_circle[prop_id] == 1:
                                success = True
                                success2 = True
                                break
                            else:
                                trading_chain[0].append(prop_id)
                                prop_circle[prop_id] += 1
                                break
                    if success2:
                        break
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
                    self.resp_matched[self.resp_indptr[resp_id]+caps_rest[resp_id]] = prop_id
        if self.switch == 0:
            return self.prop_matched, self.resp_matched
        else:
            return self.prop_matched, self.resp_matched, self.resp_indptr


    def summary(self):
        print "-----Initial Setting-----"
        print "The number of proposers : %s" % self.prop_num
        print "The number of respondants : %s" % self.resp_num
        print "Preference of proposers : (unmatched = %s)" % self.prop_unmatched
        print self.prop_prefs
        print "Preference of respondants : (unmatched = %s)" % self.resp_unmatched
        print self.resp_prefs
        if self.switch == 'mto':
            print "Capacity of respondants : "
            print self.resp_caps
        if self.switch == 'mtm':
            print "Capacity of proposers : "
            print self.prop_caps
            print "Capacity of respondants : "
            print self.resp_caps
        print "-----Matching Type-----"
        if self.switch == 'oto':
            print "One-to-One Matching"
        elif self.switch == 'mto':
            print "Many-to-One Matching"
        else:
            print "Many-to-Many Matching"
        print self.algo
        print "-----Result-----"
        print "Stable matching pairs"
        print "Proposers side : "
        print self.prop_matched
        print "Respondants side : "
        print self.resp_matched
        if self.switch == 1:
            print "resp_indptr : "
            print self.resp_indptr


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
            for j in range(self.prop_caps[i]):
                if self.prop_matched[self.prop_indptr[i]+j] != self.prop_unmatched:
                    vector["m%s" % i].append("f%s" % self.prop_matched[self.prop_indptr[i]+j])
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


def deferred_acceptance(prop_prefs, resp_prefs, caps=None):
    test = Matching(prop_prefs, resp_prefs, caps)
    if caps is None:
        prop_matched, resp_matched = test.get_pairs()
        return prop_matched, resp_matched
    else:
        prop_matched, resp_matched, indptr = test.get_pairs()
        return prop_matched, resp_matched, indptr
