"""
Filename: matching.py
Authors: Yoshimasa Ogawa
LastModified: 23/06/2015

A collection of functions to solve the matching problems.
"""
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import random
from matching import *

def make_prop_prefs(prop_num=100, resp_num=20, theta=0, alpha=0.8):
    indicater = np.tile(np.identity(resp_num), (prop_num/resp_num, 1))
    common_value = np.random.rand(resp_num)
    private_value = np.random.rand(prop_num, resp_num)
    utility = np.zeros((prop_num, resp_num))
    prop_prefs = np.zeros((prop_num, resp_num), dtype=int)

    for prop_id in range(prop_num):
        for resp_id in range(resp_num):
            utility[prop_id][resp_id] = indicater[prop_id][resp_id] * theta + alpha * common_value[resp_id] + (1 - alpha) * private_value[prop_id][resp_id]
    for prop_id in range(prop_num):
        for resp_id in range(resp_num):
            prop_prefs[prop_id][resp_id] = np.where(np.sort(utility[prop_id])[::-1] == utility[prop_id][resp_id])[0][0]

    return prop_prefs

def make_resp_prefs(prop_num=100, resp_num=20):
    random_prefs = np.random.rand(resp_num, prop_num)
    resp_prefs = np.zeros((resp_num, prop_num), dtype=int)

    for resp_id in range(resp_num):
        for prop_id in range(prop_num):
            resp_prefs[resp_id][prop_id] = np.where(np.sort(random_prefs[resp_id])[::-1] == random_prefs[resp_id][prop_id])[0][0]

    return resp_prefs

def ave_rank(prop_prefs, resp_prefs, mtype, caps=None):
    prop_num = len(prop_prefs)
    resp_num = len(resp_prefs)
    if caps is None:
        caps = [5 for col in range(resp_num)]
    M = Matching(prop_prefs, resp_prefs, caps)
    if mtype == 'DA':
        prop_matched, resp_matched, indptr = M.DA()
    elif mtype == 'TTC':
        prop_matched, resp_matched, indptr = M.TTC()
    elif mtype == 'BS':
        prop_matched, resp_matched, indptr = M.BS()
    rank = 0
    for prop_id in range(prop_num):
        rank += np.where(prop_prefs[prop_id] == prop_matched[prop_id])[0][0]+1
    ave_rank = rank / prop_num

    return ave_rank

def simulate(caps_list, mtype, prop_num=100, resp_num=20, ts=30):
    aves = [None for col in range(len(caps_list))]
    for i in range(len(caps_list)):
        ave = 0
        caps = [caps_list[i] for col in range(resp_num)]
        for j in range(ts):
            prop_prefs = make_prop_prefs(prop_num, resp_num)
            for k in range(100):
                resp_prefs = make_resp_prefs(prop_num, resp_num)
                ave += ave_rank(prop_prefs, resp_prefs, mtype, caps)
        ave /= ts * 100
        aves[i] = ave
    return aves
