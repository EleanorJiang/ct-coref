from enum import Enum
import collections
from typing import Dict, List, Optional, Tuple, DefaultDict
from itertools import permutations
import pandas as pd
import numpy as np
import math
import random
from collections import Counter
from heapq import nlargest
from ordered_set import OrderedSet

TypedSpan = Tuple[int, Tuple[int, int]]
DepSpan = Tuple[int, int, int]  # start, root, end


class Transition(Enum):
    NA = 0
    RETAIN = 1
    CONTIUNE = 2
    S_SHIFT = 3
    R_SHIFT = 4
    NOCB = 5


class ConvertedSent:
    def __init__(self, sentence_id, document_id, words, coref_spans=None, pos_tags=None, gram_role=None, srl=None,
                 top_spans=[], clusters=[], offset=None) -> None:
        self.document_id = document_id
        self.sentence_id = sentence_id
        self.words = words
        self.pos_tags = pos_tags
        self.gram_role = gram_role
        self.srl = srl
        self.top_spans = top_spans
        self.clusters = clusters
        self.offset = offset
        self.coref_spans = coref_spans


class CenteringUtterance:
    """
    Usage:
     creating a CenteringUtterance object by:
        centeringUtterance =  CenteringUtterance(convertSent, candidate="clusters", ranking="grl")
        the init function automatically setup all the utterance-level properties,
                e.g. create the CF_list with the correct ranking for you.
        However, the discourse-level properties need to be set manually.
    Utterance-level properties:
        document_id: int
        sentence_id: int
        words: List[str]
        gram_role:  the spans of different grammatical roles in this uttererance,
                    a dict  where the keys are ['pro-subj', 'pro-obj', 'pro-other', 'subj', 'obj', 'others']
                    and each value is a tuple of span and its head span
                    Dict[str, Tuple[DepSpan, DepSpan]], where DepSpan = Tuple[int, int, int] , i.e. start, root, end
        srl:  the spans of different semantic roles in this uttererance,
                    a dict  where the keys are ['PRP-ARG0', 'PRP-ARG1', 'PRP-other', 'ARG0', 'ARG1', 'others']
                    and each value is a tuple of span and its head span
                    Dict[str, Tuple[DepSpan, DepSpan]], where DepSpan = Tuple[int, int, int] , i.e. start, root, end
        ranking: str, either "grl or "srl"
        candidate_mentions: OrderedSet[TypedSpan], where each typedSpan is in the format of (entity_id, (start, end)).
        CF_list: List[TypedSpan], where each typedSpan is in the format of (entity_id, (start, end)).
    Discourse-level properties:
        CB_weights: Dict[int, float], where keys are entity_ids and values are their weights
        CB: int, entity_id. The highest ranked entity in the CB list
        first_CP: int, entity_id. The first mentioned entity in the entire discourse
        transition: Transition
        cheapness: bool, Cb(Un) = Cp(Un-1)
        coherence: bool, Cb(Un) = Cb(Un-1)
        salience: book, Cb(Un) = Cp(Un)
    """
    def __init__(self, sentence: ConvertedSent, candidate, ranking) -> None:
        self.document_id = sentence.document_id
        self.sentence_id = sentence.sentence_id
        self.words = sentence.words
        self.gram_role = sentence.gram_role
        self.srl = sentence.srl
        self.ranking = ranking
        if candidate == "coref_spans":
            self.candidate_mentions = self._set_candidate_mentions(sentence.coref_spans)
        elif candidate == "top_spans":
            self.candidate_mentions = self._set_candidate_mentions(sentence.top_spans)
        elif candidate == "clusters":
            self.candidate_mentions = self._set_candidate_mentions(sentence.clusters)
        else:
            raise AssertionError("the CT parameter `candidate` can only be "
                                 "'coref_spans', 'top_spans' or 'clusters' ")
        self.CF_list = []
        if ranking == "grl":
            self._set_CF_list(gram_role=sentence.gram_role, pos_tags=sentence.pos_tags)
        elif ranking == "srl":
            self._set_CF_list_by_srl(srl=sentence.srl, pos_tags=sentence.pos_tags)
        else:
            raise AssertionError("the CT parameter `candidate` can only be "
                                 "'grl', 'srl' or 'recency' ")
        self.CF_weights, self.CP = self._set_CF_weights()
        self.CB_weights = None
        self.first_CP = None
        self.CB = None
        self.transition = Transition.NA
        self.cheapness = None
        self.coherence = None
        self.salience = None


    def _set_candidate_mentions(self, someTypeSpanLst): # coref_spans or clusters or top_spans
        return OrderedSet(someTypeSpanLst)


    def _set_CF_list(self, gram_role, pos_tags):
        entities: DefaultDict[str, List[TypedSpan]] = collections.defaultdict(list)
        subj_spans = [(span_pair[0][0], span_pair[0][-1]) for span_pair in gram_role["subj"]]
        obj_spans = [(span_pair[0][0], span_pair[0][-1]) for span_pair in gram_role["subj"]]
        for typed_span in self.candidate_mentions:
            entity_id, span = typed_span
            if len(span) == 1 and ("PRP" in pos_tags[span[0]]):
                if span in subj_spans:
                    entities["pro-subj"].append(typed_span)
                elif span in obj_spans:
                    entities["pro-obj"].append(typed_span)
                else:
                    entities["pro-other"].append(typed_span)
            elif span in subj_spans:
                entities["subj"].append(typed_span)
            elif span in obj_spans:
                entities["obj"].append(typed_span)
            else:
                entities["others"].append(typed_span)
        for key in ['pro-subj', 'pro-obj', 'pro-other', 'subj', 'obj', 'others']:
            self.CF_list.extend(entities[key])


    def _set_CF_list_by_srl(self, srl, pos_tags):
        entities: DefaultDict[str, List[TypedSpan]] = collections.defaultdict(list)
        for typed_span in self.candidate_mentions:
            entity_id, span = typed_span
            if len(span) == 1 and ("PRP" in pos_tags[span[0]]):
                if span in srl["ARG0"]:
                    entities["PRP-ARG0"].append(typed_span)
                elif span in srl["ARG1"]:
                    entities["PRP-ARG1"].append(typed_span)
                else:
                    entities["PRP-other"].append(typed_span)
                continue
            if span in srl["ARG0"]:
                entities["ARG0"].append(typed_span)
                continue
            if span in srl["ARG1"]:
                entities["ARG1"].append(typed_span)
                continue
            else:
                entities["others"].append(typed_span)
        for key in ['PRP-ARG0', 'PRP-ARG1', 'PRP-other', 'ARG0', 'ARG1', 'others']:
            self.CF_list.extend(entities[key])


    def _set_CF_weights(self):
        CF_weights = {}
        i = 1
        entity_id = None
        for typed_span in reversed(self.CF_list):
            entity_id, span = typed_span
            CF_weights[entity_id] = i
            i += 1
        CP = entity_id
        return CF_weights, CP

    def set_CB_and_cheapness_coherence(self, centering_document):
        '''
        Please use set_CB_weights(centering_document, gamma=0, big_gamma=0) instead

        Unlike the original version of CT,
        we decide to compare Un to the nearest previous Utterance with a non-empty Cf-list
        :param centering_document:
        :return:
        '''
        # Seeking the nearest valid sentence (we are skipping utterances with empty candidate_mentions, e.g. "Uh-huh.")
        for i in range(self.sentence_id - 1, -1, -1):
            Uprev = centering_document[i]
            if len(Uprev.CF_list):
                break
        # if no previous sentences are valid (candidate_mentions/CF_list is not empty),
        # we return with CB and cheapness&coherence being None.
        if len(Uprev.CF_list) == 0 or len(self.candidate_mentions) == 0:
            return
        # set CB as the top ranking entity in Uprev.CF_list which exists in current candidate_mentions
        tmp_list = [entity[0] for entity in self.candidate_mentions]
        if Uprev.CF_list[0][0] in tmp_list:
            self.CB = Uprev.CF_list[0][0]
            self.cheapness = True
            if Uprev.CB and self.CB != Uprev.CB:
                self.coherence = False
            else:
                self.coherence = True
            return
        # if CB is not equal to the top element in Uprev.CF_list (CP_{n-1})
        self.cheapness = False
        # continue searching other elements in Uprev.CF_list
        for typed_span in Uprev.CF_list:
            if typed_span[0] in tmp_list:
                self.CB = typed_span[0]
                if Uprev.CB and self.CB != Uprev.CB:
                    self.coherence = False
                else:
                    self.coherence = True
                return
        # if candidate_mentions is not empty and Uprev.CF_list is not empty but they are disjoint, then set CB to NOCB
        self.coherence = False

    def set_CB_weights(self, Uprev, paras=None):
        '''
        Modify from set_CB_and_cheapness_coherence:
            changed the way to choose CB
            now it's the highest ranked entity in CB_weights
        CB, cheapness and coherence are also set here
        paras: [paras, gate_escape_factor]
              W[Cb(Un)] <- W[Cb(Uprev)] * decay_factor +
                            gate_escape_factor * [ Cf(Uprev) - Cf(Un) ] * Uprev.CF_weights[Cf(Uprev)] +  [Cf(Un) ^ Cf(Uprev)] * Uprev.CF_weights[Cf(Uprev)]

        '''
        if paras is None:
            gamma, big_gamma, gate_escape_factor = 0, 0, 0
        else:
            gamma, big_gamma, gate_escape_factor = paras
        self.CB_weights = Uprev.CB_weights
        # set the weight of entities in the CF_list to be the reversed ranks of them
        # now set up self.CB_weights
        if Uprev.CB_weights is None:
            # Uprev is the first utterance U_0, set Cb(U_1) <- Cf(U_0)
            self.CB_weights = Uprev.CF_weights
            self.first_CP = Uprev.CF_list[0][0]
        else:
            # Cb(Un) <- Cb(Uprev) + Cf(Un)
            # W[Cb(Un)] <- W[Cb(Uprev)] + weight[Cf(Uprev)] * gate(Cf(Un), Cf(Uprev))
            self.CB_weights = {}
            previous_CBs = set(Uprev.CB_weights.keys())
            previous_CFs = set(Uprev.CF_weights.keys())
            CBs = previous_CBs.union(previous_CFs)
            for entity in CBs:
                self.CB_weights[entity] = 0
                decay_factor = gamma
                if entity == Uprev.first_CP:
                    decay_factor = big_gamma
                if entity in Uprev.CB_weights.keys():
                    # entity that in previous CB list
                    self.CB_weights[entity] += Uprev.CB_weights[entity] * decay_factor
                if entity in Uprev.CF_weights.keys():  # entity in self.CF_weights.keys() and
                    # this entity is in current CF list and prvious CF list
                    if entity in self.CF_weights.keys():
                        self.CB_weights[entity] += Uprev.CF_weights[entity]
                    else:
                        self.CB_weights[entity] += Uprev.CF_weights[entity] * gate_escape_factor
                # if gamma != 0 and self.CB_weights[entity] == 0:
                #     print("entity: ", entity)
                #     print("Uprev.CF_weights: ", Uprev.CF_weights)
                #     print("Uprev.CB_weights", Uprev.CB_weights)
                #     print("CF_weights: ", self.CF_weights)
                #     print("CB_weights", Uprev.CB_weights)
                #     raise AssertionError("entity must be either in prev CF or CB")
            # print("CB_weights", self.CB_weights)


        self.CB = nlargest(1, self.CB_weights, key=self.CB_weights.get)[0]
        # print(self.CB)
        if self.CB_weights[self.CB] == 0:
            self.CB = None
        self.cheapness = self.set_cheapness(Uprev)
        self.coherence = self.set_coherence(Uprev)


    def set_cheapness(self, Uprev):
        """
        Uprev.CF_list[0][0] is the entity id of CP, the top one entity in the CF_list
        """
        return self.CB == Uprev.CP

    def set_coherence(self, Uprev):
        """
        Uprev.CF_list[0][0] is the entity id of CP, the top one entity in the CF_list
        """
        return self.CB == Uprev.CB

    def set_salience(self):
        if self.CF_list != []:
            if not self.CB:
                self.salience = False
            else:
                self.salience = (self.CB == self.CF_list[0][0])

    def set_transition(self):
        if self.coherence == True and self.salience == True:
            self.transition = Transition.CONTIUNE
        elif self.coherence == True and self.salience == False:
            self.transition = Transition.RETAIN
        elif self.coherence == False and self.salience == True:
            self.transition = Transition.S_SHIFT
        elif self.coherence == False and self.salience == False:
            self.transition = Transition.R_SHIFT
        # if self.salience is None and self.coherence is not None:
        #      print("Error in set_transition:", self.transition, self.coherence, self.salience, self.CF_list, self.candidate_mentions)


"""
The following functions are for generating statistics of centering_documents, namely Table 1
"""
def converted_to_centering(converted_document, paras=None,
                           candidate="cluster", ranking="grl") -> List[List[CenteringUtterance]]:
    '''
    Get centering_documents: Convert ConvertedSent to CenteringUtterance
    :param converted_documents:
    :param candidate: "top_spans" or "cluster"
    :param ranking: "grl" or "srl"
    :param gamma: [0,1]
    :param big_gamma: [0,1], or None
    :return: centering_documents: List[List[CenteringUtterance]], a list of centering_document,
    where centering_document is a list of centering_utterance
    '''
    centering_document = []
    curUtterance = None
    for j in range(0, len(converted_document)):
        tmpUtterance = CenteringUtterance(converted_document[j], candidate=candidate, ranking=ranking)
        if len(tmpUtterance.CF_list) == 0:
            continue
        if not curUtterance:
            curUtterance = tmpUtterance
            continue
        prviousUtterance = curUtterance
        curUtterance = tmpUtterance
        curUtterance.set_CB_weights(prviousUtterance, paras=paras)
        # print("CF_weights: ", curUtterance.CF_weights)
        # print("CB_weights", curUtterance.CB_weights)
        curUtterance.set_salience()
        curUtterance.set_transition()
        centering_document.append(curUtterance)
    return centering_document


def statistics(centering_document, doc_id):
    """
    return df: a dataframe of one row, 11 column
                corresponds to one line in Table_1_per_document
    """
    cnt = Counter()
    cnt["doc_id"] = doc_id
    cnt["num_all_u"] = len(centering_document)
    cnt["num_valid_u"], cnt["num_nocb"], cnt["num_cheapness"], cnt["num_coherence"], cnt["num_salience"], cnt[
        "num_continue"], cnt["num_retain"], cnt["num_s_shift"], cnt["num_r_shift"] = 0, 0, 0, 0, 0, 0, 0, 0, 0
    for centeringUtterance in centering_document[1:]:
        if len(centeringUtterance.candidate_mentions) == 0:
            continue
        # print(centeringUtterance.transition, centeringUtterance.transition is Transition.CONTIUNE)
        cnt["num_valid_u"] += 1
        if centeringUtterance.CB==None:
            cnt["num_nocb"] += 1
        if centeringUtterance.cheapness:
            cnt["num_cheapness"] += 1
        if centeringUtterance.coherence:
            cnt["num_coherence"] += 1
        if centeringUtterance.salience:
            cnt["num_salience"] += 1
        if centeringUtterance.transition == Transition.CONTIUNE:
            # print("num_continue++")
            cnt["num_continue"] += 1
        elif centeringUtterance.transition == Transition.RETAIN:
            cnt["num_retain"] += 1
        elif centeringUtterance.transition == Transition.S_SHIFT:
            cnt["num_s_shift"] += 1
        elif centeringUtterance.transition == Transition.R_SHIFT:
            cnt["num_r_shift"] += 1
    df = pd.DataFrame([cnt.values()], columns=cnt.keys())
    return df


def more_stastics(df):
    '''
    calculate the percentage of utterances that do not violate a certain CT constraint in the entire document.
    for all the scores, higher == more coherent
    '''
    df['num_kp'] = (df['num_valid_u'] - df['num_nocb'] + df['num_cheapness'] + df['num_coherence'] + df['num_salience']) / 4
    df['%valid_u'] = df['num_valid_u']/df['num_all_u']*100
    df['%not_nocb'] = (1-df['num_nocb']/df['num_valid_u'])*100
    df['%cheapness'] = df['num_cheapness']/df['num_valid_u']*100
    df['%coherence'] = df['num_coherence']/df['num_valid_u']*100
    df['%salience'] = df['num_salience']/df['num_valid_u']*100
    df['%continue'] = df['num_continue']/(df['num_valid_u'])*100
    df['%retain'] = df['num_retain']/(df['num_valid_u'])*100
    df['%s_shift'] = df['num_s_shift']/(df['num_valid_u'])*100
    df['%r_shift'] = df['num_r_shift']/(df['num_valid_u'])*100
    df['%kp'] = (df['%not_nocb'] + df['%cheapness']+ df['%coherence'] + df['%salience']) /4


"""
The following functions are for computing centering scores, namely Table 2
"""
def get_perm_id_lists(num_sent, search_space_size):
    """
    :param num_sent:
    :param search_space_size:
    :return: a list (length of search_space_size) of id list (length of num_sent)
    """
    # if num_sent is small, then return all permutation of [0,1,2,3,4,5]
    if math.factorial(num_sent) <= search_space_size:
        return list(permutations(range(num_sent)))[1:]
    perm_id_lists = []
    for i in range(search_space_size - 1):
        random.seed(i)
        perm_id_lists.append(np.random.permutation(num_sent))
    return perm_id_lists


def get_centering_search_spaces(documents, search_space_size, candidate, ranking, paras):
    centering_search_spaces = []
    for i, document in enumerate(documents):
        perm_id_lists = get_perm_id_lists(len(document), search_space_size)
        original_centerDoc = converted_to_centering(document,
                                    paras=paras, candidate=candidate, ranking=ranking)
        centering_search_space = [original_centerDoc]
        # for ids in random_documents_id_list:
        #     permutated_document = [document[i] for i in ids]
        #     centering_search_spaces.append(doc2CenterDoc(permutated_document, candidate=candidate, ranking=ranking))
        centering_search_space.extend(
            [converted_to_centering([document[i] for i in ids],
                    paras=paras, candidate=candidate, ranking=ranking) for ids in perm_id_lists]
        )
        # for k in range(1, search_space_size, 1):
        #     random_document = shuffle_list(doc_search_spaces)
        #     doc_search_spaces.append(random_document)
        #     centering_search_space.append(doc2CenterDoc(random_document, candidate=candidate, ranking=ranking))
        centering_search_spaces.append(centering_search_space)
    return centering_search_spaces


def compute_CT_score(doc_id, centering_search_space, search_space_size, M: str):
    '''
    :param centering_search_space:  a list of random sampled centering documents from one original document
    :param search_space_size: len(centering_search_space)
    :param M: string, the metric, '%not_nocb' or '%cheapness' or 'transition'
                                  'kp'(sum up '%not_nocb'&'%cheapness'&'%coherence'&'%salience')
    :return: float [0,1], the classification rate, the higher, the better
    '''
    original_doc_score = centering_search_space[0]
    df = pd.concat([statistics(centering_document, doc_id) for centering_document in centering_search_space], ignore_index=True,
                   sort=False)
    more_stastics(df)
    if M == 'transition':
        return compute_CT_score_by_transition(df,search_space_size)
    worse = df[df[M] < df[M].get(0)].count()[0]
    equal = df[df[M] == df[M].get(0)].count()[0]
    classification_rate = (worse + equal / 2) / search_space_size
    return classification_rate


def compute_CT_score_by_transition(df,search_space_size):
    worse = df[(df['%continue'] < df['%continue'].get(0))
               | ((df['%continue'] == df['%continue'].get(0)) & (df['%retain'] < df['%retain'].get(0))) |
               ((df['%continue'] == df['%continue'].get(0)) & (df['%retain'] == df['%retain'].get(0)) & (
                           df['%s_shift'] < df['%s_shift'].get(0)))
               ].count()[0]
    equal = df[((df['%continue'] == df['%continue'].get(0)) & (df['%retain'] == df['%retain'].get(0)) & (
                df['%s_shift'] == df['%s_shift'].get(0)))
    ].count()[0]
    classification_rate = (worse + equal / 2) / search_space_size
    return classification_rate
