"""
Filename: matchfuncs.py
Author: Yoshiasa Ogawa
Functions for matching algorithms.
"""
import numpy as np
import itertools
import gambit
from collections import defaultdict

def DA(prop_prefs, resp_prefs, resp_caps=None, prop_caps=None, list_length=None):
    """
    Deffered Acceptance Algorithm

    Parameters
    ---
    prop_prefs : ndarray(int, ndim=2)
      Preference of proposers
    resp_prefs : ndarray(int, ndim=2)
      Preference of respondants
    prop_caps : list
      Capacity of proposers
    resp_caps : list
      Capacity of respondants

    Returns
    ---
    prop_matched : ndarray(int, ndim=1)
        Matching Pairs for proposers
    resp_matched : ndarray(int, ndim=1)
        Matching Pairs for respondants
    prop_indptr : ndarray(int, ndim=1)
        Index Pointer for proposers
    resp_indptr : ndarray(int, ndim=1)
        Index Pointer for respondants

    """
    prop_prefs = np.asarray(prop_prefs)
    resp_prefs = np.asarray(resp_prefs)
    prop_num = prop_prefs.shape[0]
    resp_num = resp_prefs.shape[0]
    prop_unmatched = resp_num
    resp_unmatched = prop_num
    resp_ranks = np.argsort(resp_prefs)
    if prop_caps is None:
        p_caps = [1 for col in range(prop_num)]
    else:
        p_caps = prop_caps
    if resp_caps is None:
        r_caps = [1 for col in range(resp_num)]
    else:
        r_caps = resp_caps
    prop_matched = np.zeros(sum(p_caps), dtype=int) + prop_unmatched
    resp_matched = np.zeros(sum(r_caps), dtype=int) + resp_unmatched
    prop_indptr = np.zeros(prop_num+1, dtype=int)
    resp_indptr = np.zeros(resp_num+1, dtype=int)
    np.cumsum(p_caps, out=prop_indptr[1:])
    np.cumsum(r_caps, out=resp_indptr[1:])
    propcaps_rest = [i for i in p_caps]
    respcaps_rest = [i for i in r_caps]
    prop_single = range(prop_num)
    prop_counter = [0 for i in range(prop_num)]
    if list_length is None:
        l_length = prop_prefs.shape[1]
    else:
        l_length = list_length

    while len(prop_single) >= 1:
        prop_single_roop = [i for i in prop_single]
        for prop_id in prop_single_roop:
            if prop_counter[prop_id] == l_length:
                prop_single.remove(prop_id)
                break
            resp_id = prop_prefs[prop_id][prop_counter[prop_id]]
            prop_counter[prop_id] += 1
            if resp_id == prop_unmatched:
                prop_single.remove(prop_id)
            elif respcaps_rest[resp_id] >= 1:
                propcaps_rest[prop_id] -= 1
                respcaps_rest[resp_id] -= 1
                prop_matched[prop_indptr[prop_id]+propcaps_rest[prop_id]] = resp_id
                resp_matched[resp_indptr[resp_id]+respcaps_rest[resp_id]] = prop_id
                if propcaps_rest[prop_id] == 0:
                    prop_single.remove(prop_id)
            else:
                deffered = resp_matched[resp_indptr[resp_id]:resp_indptr[resp_id+1]]
                max_rank = max([resp_ranks[resp_id][i] for i in deffered])
                max_id = resp_prefs[resp_id][max_rank]
                if resp_ranks[resp_id][prop_id] < max_rank:
                    prop_single.append(max_id)
                    propcaps_rest[max_id] += 1
                    propcaps_rest[prop_id] -= 1
                    prop_matched[np.where(prop_matched[prop_indptr[max_id]:prop_indptr[max_id+1]] == resp_id)[0] + prop_indptr[max_id]] = prop_unmatched
                    prop_matched[prop_indptr[prop_id]+propcaps_rest[prop_id]] = resp_id
                    resp_matched[np.where(resp_matched[resp_indptr[resp_id]:resp_indptr[resp_id+1]] == max_id)[0] + resp_indptr[resp_id]] = prop_id
                    if propcaps_rest[prop_id] == 0:
                        prop_single.remove(prop_id)

    if prop_caps is None and resp_caps is None:
        return prop_matched, resp_matched
    elif prop_caps is None:
        return prop_matched, resp_matched, resp_indptr
    else:
        return prop_matched, resp_matched, prop_indptr, resp_indptr

def BOS(prop_prefs, resp_prefs, resp_caps=None, prop_caps=None, list_length=None):
    """
    Boston School Algorithm

    Parameters
    ---
    prop_prefs : ndarray(int, ndim=2)
      Preference of proposers
    resp_prefs : ndarray(int, ndim=2)
      Preference of respondants
    prop_caps : list
      Capacity of proposers
    resp_caps : list
      Capacity of respondants

    Returns
    ---
    prop_matched : ndarray(int, ndim=1)
        Matching Pairs for proposers
    resp_matched : ndarray(int, ndim=1)
        Matching Pairs for respondants
    prop_indptr : ndarray(int, ndim=1)
        Index Pointer for proposers
    resp_indptr : ndarray(int, ndim=1)
        Index Pointer for respondants

    """
    prop_prefs = np.asarray(prop_prefs)
    resp_prefs = np.asarray(resp_prefs)
    prop_num = prop_prefs.shape[0]
    resp_num = resp_prefs.shape[0]
    prop_unmatched = resp_num
    resp_unmatched = prop_num
    resp_ranks = np.argsort(resp_prefs)
    if prop_caps is None:
        p_caps = [1 for col in range(prop_num)]
    else:
        p_caps = prop_caps
    if resp_caps is None:
        r_caps = [1 for col in range(resp_num)]
    else:
        r_caps = resp_caps
    prop_matched = np.zeros(sum(p_caps), dtype=int) + prop_unmatched
    resp_matched = np.zeros(sum(r_caps), dtype=int) + resp_unmatched
    prop_indptr = np.zeros(prop_num+1, dtype=int)
    resp_indptr = np.zeros(resp_num+1, dtype=int)
    np.cumsum(p_caps, out=prop_indptr[1:])
    np.cumsum(r_caps, out=resp_indptr[1:])
    propcaps_rest = [i for i in p_caps]
    respcaps_rest = [i for i in r_caps]
    prop_single = range(prop_num)
    prop_counter = [0 for i in range(prop_num)]
    if list_length is None:
        l_length = prop_prefs.shape[1]
    else:
        l_length = list_length

    while len(prop_single) >= 1:
        prop_single_copy = [i for i in prop_single]
        approach = np.zeros(resp_num)
        for prop_id in prop_single_copy:
            if prop_counter[prop_id] == l_length:
                prop_single.remove(prop_id)
                continue
            resp_id = prop_prefs[prop_id][prop_counter[prop_id]]
            prop_counter[prop_id] += 1
            if resp_id == prop_unmatched:
                prop_single.remove(prop_id)
            else:
                approach[resp_id] += 1
        prop_single_copy = [i for i in prop_single]
        for resp_id in range(resp_num):
            if respcaps_rest[resp_id] >= approach[resp_id]:
                for prop_id in prop_single_copy:
                    if prop_prefs[prop_id][prop_counter[prop_id]-1] == resp_id:
                        propcaps_rest[prop_id] -= 1
                        respcaps_rest[resp_id] -= 1
                        prop_matched[prop_indptr[prop_id]+propcaps_rest[prop_id]] = resp_id
                        resp_matched[resp_indptr[resp_id]+respcaps_rest[resp_id]] = prop_id
                        if propcaps_rest[prop_id] == 0:
                            prop_single.remove(prop_id)
            elif respcaps_rest[resp_id] != 0:
                applicants = [i for i in prop_single_copy if prop_prefs[i][prop_counter[prop_id]-1] == resp_id]
                for k in range(resp_prefs.shape[1]):
                    prop_id = resp_prefs[resp_id][k]
                    if prop_id in applicants:
                        propcaps_rest[prop_id] -= 1
                        respcaps_rest[resp_id] -= 1
                        prop_matched[prop_indptr[prop_id]+propcaps_rest[prop_id]] = resp_id
                        resp_matched[resp_indptr[resp_id]+respcaps_rest[resp_id]] = prop_id
                        if propcaps_rest[prop_id] == 0:
                            prop_single.remove(prop_id)
                    if respcaps_rest[resp_id] == 0:
                        break

    if prop_caps is None and resp_caps is None:
        return prop_matched, resp_matched
    elif prop_caps is None:
        return prop_matched, resp_matched, resp_indptr
    else:
        return prop_matched, resp_matched, prop_indptr, resp_indptr

def Nash(algo, prop_prefs, resp_prefs, resp_caps=None, prop_caps=None, list_length=None):
    """
    Nash equilibria in pure strategy.

    Parameters
    ---
    algo : string
        'DA'  : Deffered Acceptance Algorithm
        'BOS' : Boston School Choice
    prop_prefs : ndarray(int, ndim=2)
        Preference of proposers
    resp_prefs : ndarray(int, ndim=2)
        Preference of respondants
    prop_caps : list
        Capacity of proposers
    resp_caps : list
        Capacity of respondants

    Returns
    ---
    Nash : ndarray(int, ndim=2)
        Nash equilibria in pure strategy

    """
    prop_prefs = np.asarray(prop_prefs)
    resp_prefs = np.asarray(resp_prefs)
    prop_num = prop_prefs.shape[0]
    resp_num = resp_prefs.shape[0]
    resp_ranks = np.argsort(resp_prefs)
    iter_prefs = np.zeros((prop_num, resp_num), dtype=int)
    l = range(resp_num)
    behavior = defaultdict()
    counter = 0
    for element in itertools.permutations(l, len(l)):
        behavior[counter] = element
        counter += 1
    g = gambit.Game.new_table([counter, counter, counter])
    for profile in g.contingencies:
        for prop_id, behave_num in zip(range(prop_num), profile):
            iter_prefs[prop_id] = behavior[behave_num]
        if algo == 'DA':
            prop_matched, resp_matched = DA(iter_prefs, resp_prefs, resp_caps, prop_caps, list_length)
        elif algo == 'BOS':
            prop_matched, resp_matched = BOS(iter_prefs, resp_prefs, resp_caps, prop_caps, list_length)
        for i in range(prop_num):
            g[profile][i] = resp_num - resp_ranks[i][prop_matched[i]]
    solver = gambit.nash.ExternalEnumPureSolver()
    nash = solver.solve(g)
    return nash, behavior

def SubmitList(nash, behavior):
    """
    Make a submit preference from one of the nash equilibria

    Parameters
    ---
    nash : ndarray(int, ndim=1)
        one of the nash equilibria
    behavior : dict
    """
    nash = np.asarray(nash, dtype=int)
    comb = len(behavior)
    resp_num = len(behavior[0])
    prop_num = len(nash) / comb
    nash = nash.reshape(prop_num, comb)
    submit_pref = np.zeros((prop_num, resp_num), dtype=int)
    for prop_id in range(prop_num):
        submit_pref[prop_id] = behavior[np.where(nash[prop_id] == 1)[0][0]]
    return submit_pref
