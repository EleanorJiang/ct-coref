from itertools import permutations
import math
import random
import numpy as np


def whether_in_the_list(name, epochs):
    # false to skip (continue), true to do analysis
    for epoch in epochs:
        if f"_{epoch}." in f"_{name}.":
            return True
        if epoch == "best" and name =="best":
            return True
    return False


def find_span(head):
    def dfs(token):
        nonlocal start, end
        is_empty = True
        for elem in token.children:
            is_empty = False
        if is_empty:
            if token.i < start:
                start = token.i
            if token.i > end:
                end = token.i
            return
        for child in token.children:
            dfs(child)
    start, end = head.i, head.i
    dfs(head)
    return start, end


def decide_gram_role(gram_role, token):
    """
    decide the gram role of token and add it to gram_role
    Noth that it highly depends on the spacy version you use.
    gram_role: `Dict[str, List[Span]`.
            The keys are: "subj" (subject) and "obj" (object).
            The values are lists of spans.
    It should be chosen from the following list:
                csubj: clausal subject
                csubj: clausal subject
                csubjpass: clausal subject (passive)
                nsubj: nominal subject
                nsubjpass: nominal subject (passive)
                obj: object
                dobj: direct object
                pobj: object of preposition
                (root, used to decide the importance)
    new feature:
        now we also find the head of this token, along with its position and span cover
            gram_role: a dict where the keys are ['sbj', 'obj'] and each value is a tuple of span and its head span
                Dict[str, Tuple[DepSpan, DepSpan]], where DepSpan = Tuple[int, int, int] , i.e. start, root, end
    """
    tags = ['nsubj', 'nsubjpass', 'csubj', 'csubjpass', 'obj', 'dobj', 'pobj' ]
    if 'subj' in token.dep_:
        start, end = find_span(token)
        hstart, hend = find_span(token.head)
        # gram_role['subj'].append(((start, token.i, end), (hstart, token.head.i, hend)))
        gram_role['subj'].append((start, end))
    if 'obj' in token.dep_:
        start, end = find_span(token)
        hstart, hend = find_span(token.head)
        # gram_role['obj'].append(((start, token.i, end), (hstart, token.head.i, hend)))
        gram_role['obj'].append((start, end))
    return gram_role


def get_perm_id_lists(num_sent, search_space_size):
    """
    :param num_sent:
    :param search_space_size:
    :return: a list (length of search_space_size-1) of id list (length of num_sent)
    """
    # if num_sent is small, then return all permutation of [0,1,2,3,4,5]
    if math.factorial(num_sent) <= search_space_size:
        return list(permutations(range(num_sent)))[1:]
    perm_id_lists = []
    for i in range(search_space_size - 1):
        random.seed(i)
        perm_id_lists.append(np.random.permutation(num_sent))
    return perm_id_lists


