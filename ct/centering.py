from enum import Enum
import collections
from typing import Dict, List, Optional, Tuple, DefaultDict
import pandas as pd
import numpy as np
import json
from collections import Counter
from heapq import nlargest
from ontonotes.conll_util import string2list, string2grldict, string2clusters
from ct.ct_util import get_perm_id_lists
import copy

TypedSpan = Tuple[int, Tuple[int, int]]
DepSpan = Tuple[int, int, int]  # start, root, end
DEFAULT_WEIGHT = 0

class Transition(Enum):
    NA = 0
    NOCB = 1
    R_SHIFT = 2
    S_SHIFT = 3
    RETAIN = 4
    CONTIUNE = 5


class ConvertedSent:
    """
    A class representing the annotations available for a single CONLL formatted sentence
    or a sentenece with coref predictions.
    # Parameters：
        - document_id: `int`.
        - line_id: `int`. The true sentence id within the document.
        - words: `List[str]`. A list of tokens corresponding to this sentence.
                    The Onotonotes tokenization, need to be mapped.
        - clusters: `Dict[int, List[Tuple[int, int]]]`.
        - pos_tags: `List[str]`. The pos annotation of each word.
        - named_entities: `List[str]`. The BIO tags for named entities in the sentence.
        - gram_roles: `Dict[str, List[Tuple[int, int]]]`. The keys are 'subj', 'obj'.
                        The values are lists of spans.
        - semantic_roles:  the spans of different semantic roles in this uttererance,
            a dict  where the keys are 'ARG0', 'ARG1'.
    """
    def __init__(self, document_id, line_id, words, clusters=None, pos_tags=None,
                 gram_roles=None, semantic_roles=None, named_entities=None) -> None:
        self.document_id = document_id
        self.line_id = line_id
        self.words = words
        self.clusters = clusters
        self.pos_tags = pos_tags
        self.gram_roles = gram_roles
        self.semantic_roles = semantic_roles
        self.named_entities = named_entities


class ConvertedDoc:
    """
    A class representing the annotations for a `CONLL` formatted document.
    # Parameters
        - document_id: `int`.
        - sentences: `List[ConvertedSent]`.
        - entity_ids: `set[int]`. A set of entity ids that appear in this documents
                according to the `clusters` in all the `convertedSent`s.
    """
    def __init__(self, document_id=None) -> None:
        self.document_id = document_id
        self.sentences = []
        self.entity_ids = set()

    def get_entity_ids_from_ontonotes(self):
        for sentence in self.sentences:
            for entity_id in sentence.clusters.keys():
                self.entity_ids.add(entity_id)

    def update_clusters(self, allennlp_clusters, map):
        """
        Update the clusters attribute in each ConvertedSent object from ontonotes GT annotations to
        the predicted dicts from allennlp_csv. The entity_id's remains the same.
        :param allennlp_clusters: `List[List[Span]]`. The token ids are document-level ids.
                        Example: [[[42, 44], [68, 71]], ... ].
        :param map: `Dict[int, List[Tuple[int, int, int]]`
                    348 key-value pairs，the key is “doc_id”,
                    and the value is a list of 3-tuples Tuple[int, int, int] of length len(allennlp_tokens),
                    where the 3-tuple is (line_id, start, end).
        """
        self.entity_ids = set()
        doc_id = str(self.document_id)
        for line_id in range(len(self.sentences)):
            self.sentences[line_id].clusters = collections.defaultdict(list)
        for entity_id, cluster in enumerate(allennlp_clusters):
            self.entity_ids.add(entity_id)
            for span in cluster:
                line_id = map[doc_id][span[0]][0]
                new_span = (map[doc_id][span[0]][1], map[doc_id][span[1]][2])
                # print(line_id, span, new_span)
                # print(self.sentences[line_id].words[new_span[0]:new_span[1]+1])
                self.sentences[line_id].clusters[entity_id].append(new_span)



class CenteringUtterance:
    """
    # Usage:
     creating a CenteringUtterance object by:
        centeringUtterance =  CenteringUtterance(convertSent, candidate="clusters", ranking="grl")
        the init function automatically setup all the utterance-level properties,
                e.g. create the CF_list with the correct ranking for you.
        However, the discourse-level properties need to be set manually.

    # Parameters:
    Ontonotes Annotations:
        - document_id: `int`.
        - line_id: `int`. The true sentence id within the document.
        - words: `List[str]`. A list of tokens corresponding to this sentence.
                    The Onotonotes tokenization, need to be mapped.
        - clusters: `Dict[int, List[Tuple[int, int]]]`.
        - pos_tags: `List[str]`. The pos annotation of each word.
        - named_entities: `List[str]`. The BIO tags for named entities in the sentence.
        - gram_roles: `Dict[str, List[Tuple[int, int]]]`. The keys are 'subj', 'obj'.
                        The values are lists of spans.
        - semantic_roles:  the spans of different semantic roles in this uttererance,
                    a dict  where the keys are 'ARG0', 'ARG1'.
                    The values are lists of spans.

    Utterance-level properties:
        - ranking: `str`. either `grl` or `srl`.
        - CF_list: `List[int]`.
        - CF_weights: `Dict[int, float]`.
                The keys are entity id's and the values are their corresponding weights.
        - CP: `int`. The highest ranked element in the CF_list.

    Discourse-level properties:
        - CB_list: `List[int]`. A list of `entity_id`s which are the CB candidates in this utterance.
        - CB_weights: `Dict[int, float]`. The keys are `entity_id`s in `CB_list` and the values are their weights.
        - CB: `int`. The highest ranked entity in the `CB_list`.
        - first_CP: `int`. The first mentioned entity in the utterance.
        - transition: `Transition`
        - cheapness: `bool`. Cb(Un) = Cp(Un-1)
        - coherence: `bool`. Cb(Un) = Cb(Un-1)
        - salience: `bool`. Cb(Un) = Cp(Un)
        - nocb: `bool`. The `CB_list` is empty.
    """
    def __init__(self, sentence: ConvertedSent, ranking) -> None:
        self.document_id = sentence.document_id
        self.line_id = sentence.line_id
        self.words = sentence.words
        self.clusters = sentence.clusters
        self.pos_tags = sentence.pos_tags
        self.gram_roles = sentence.gram_roles
        if 'subj' not in self.gram_roles.keys():
            self.gram_roles['subj'] = []
        if 'obj' not in self.gram_roles.keys():
            self.gram_roles['obj'] = []
        self.semantic_roles = None  # todo: add a semantic_roles attribute for ConvertedSent
        self.ranking = ranking
        self.CP = None
        self.CF_weights = None
        self.CF_list = None
        self.CB_weights = None
        self.first_CP = None
        self.CB = None
        self.transition = Transition.NA
        self.cheapness = None
        self.coherence = None
        self.salience = None
        self._set_utterance_properties()

    @classmethod
    def get_sorted_entity_list(self, entity_weights):
        # sort entity_ids by entity_weights
        entity_weights = sorted(entity_weights.items(), key=lambda item: item[1], reverse=True)
        entity_list = []
        for entity_id, weight in entity_weights:
            if weight == 0:
                break
            entity_list.append(entity_id)
        return entity_list

    @classmethod
    def _add_key_value_once(self, dict, key, value):
        if key not in dict.keys():
            dict[key] = value

    def _set_CF_weights(self):
        """
        CF_weights: `Dict[int, float]`. The keys are `entity_id`s and the values are their weights.
        TODO: add srl
        """
        CF_weights: Dict[int, float] = {}
        for entity_id, spanList in self.clusters.items():
            for span in spanList:
                if len(span) == 1 and ("PRP" in self.pos_tags[span[0]]):
                    if span in self.gram_roles['subj']:
                        self._add_key_value_once(CF_weights, entity_id, 6)
                    elif span in self.gram_roles['obj']:
                        self._add_key_value_once(CF_weights, entity_id, 5)
                    else:
                        self._add_key_value_once(CF_weights, entity_id, 4)
                elif span in self.gram_roles['subj']:
                    self._add_key_value_once(CF_weights, entity_id, 3)
                elif span in self.gram_roles['subj']:
                    self._add_key_value_once(CF_weights, entity_id, 2)
                else:
                    self._add_key_value_once(CF_weights, entity_id, 1)
        self.CF_weights = dict(CF_weights)

    def _set_utterance_properties(self):
        if self.clusters == {}:
            return
        self._set_CF_weights()
        self.CF_list = self.get_sorted_entity_list(self.CF_weights)
        self.CP = self.CF_list[0]

    def set_transition(self):
        if self.coherence == True and self.salience == True:
            self.transition = Transition.CONTIUNE
        elif self.coherence == True and self.salience == False:
            self.transition = Transition.RETAIN
        elif self.coherence == False and self.salience == True:
            self.transition = Transition.S_SHIFT
        elif self.coherence == False and self.salience == False:
            self.transition = Transition.R_SHIFT


class CenteringDiscourse:
    """
    A class representing a discourse with centering properties.
    # Parameters：
        - document_id: `int`.
        - utterances: `List[CenteringUtterance]`.
        - first_CP: `int`. The first mentioned entity in the entire discourse.
        - ranking: `str`. either `grl` or `srl`.
        - len: `int`. The number of utterances in this discourse.
        - salience: the ratio of salient utterances to the total number of utterances.
        - coherence: the ratio of coherent transitions to all transitions (`len-1`).
        - cheapness: the ratio of cheap transitions to all transitions (`len-1`).
        - nocb: the ratio of utterances with nocb to the total number of utterances.
    """
    def __init__(self, converted_document, ranking, recency_win=None) -> None:
        self.document_id = converted_document.document_id
        self.ranking = ranking
        self.recency_win = recency_win
        self.salience = 0
        self.coherence = 0
        self.cheapness = 0
        self.nocb = 0
        self.transition = 0
        self.utterances = [CenteringUtterance(converted_sent, ranking) for converted_sent in converted_document.sentences]
        self._remove_empty_utterance()
        self.valid_len = len(self.utterances)
        self.reset_discourse_properties()

    def _remove_empty_utterance(self):
        utterances = []
        for utterance in self.utterances:
            if utterance.clusters != {}:
                utterances.append(utterance)
        self.utterances = utterances

    def _set_CB_weights(self, i):
        """
        Set CB_weights for self.utterances[i].
        CB_weights: `Dict[int, float]`. The keys are `entity_id`s and the values are their weights.
        """
        CB_weights: Dict[int, float] = {}
        for entity_id in self.utterances[i-1].CF_list:
            # Same as self.utterances[i-1].CF_list, but remove those who does not appears in self.utterances[i].CF_list
            if entity_id in self.utterances[i].CF_list:
                CB_weights[entity_id] = self.utterances[i-1].CF_weights[entity_id]
        return CB_weights

    def reset_discourse_properties(self):
        """
        CB_list, CB_weights, CB, first_CP, transition, cheapness, coherence, salience, nocb
        """
        for i in range(1, len(self.utterances)):
            self.utterances[i].CB_weights = self._set_CB_weights(i)
            self.utterances[i].CB_list = CenteringUtterance.get_sorted_entity_list(self.utterances[i].CB_weights)
            self.utterances[i].nocb = (len(self.utterances[i].CB_list) == 0)
            if not self.utterances[i].nocb:
                self.utterances[i].CB = self.utterances[i].CB_list[0]
            self.utterances[i].cheapness = (self.utterances[i].CB == self.utterances[i-1].CP)
            self.utterances[i].coherence = (self.utterances[i].CB == self.utterances[i - 1].CB)
            self.utterances[i].salience = (self.utterances[i].CB == self.utterances[i].CP)
            self.utterances[i].set_transition()

    def comput_CT_scores(self):
        """
        :return: a dict of CT scores.
        """
        if self.valid_len > 1:
            self.nocb = -sum([int(self.utterances[i].nocb) for i in range(1, len(self.utterances))]) / (self.valid_len -1)
            self.salience = sum([int(self.utterances[i].salience) for i in range(1, len(self.utterances))]) / (self.valid_len -1)
            self.coherence = sum([int(self.utterances[i].coherence) for i in range(1, len(self.utterances))]) / (self.valid_len -1)
            self.cheapness = sum([int(self.utterances[i].cheapness) for i in range(1, len(self.utterances))]) / (self.valid_len -1)
            self.transition = sum([int(self.utterances[i].transition.value) for i in range(1, len(self.utterances))]) / ((self.valid_len -1) * 5)
        return {"nocb": self.nocb, "salience": self.salience,
                "coherence": self.coherence, "cheapness": self.cheapness,
                "transition": self.transition,
                "kp": (self.nocb+self.salience+self.coherence)/3}



def calculate_permutation_scores(centeringDiscourse, search_space_size=100):
    """
    :param centeringDiscourse: the centeringDiscourse object with its utterances in the original order.
    :param search_space_size: the size of the search space. Default: 100.
    :return:
    """
    final_CT_scores = {"nocb": 0, "salience": 0, "coherence": 0, "cheapness": 0, "transition": 0, "kp": 0}
    orginal_utterances = copy.deepcopy(centeringDiscourse.utterances)
    unnormalized_CT_scores = centeringDiscourse.comput_CT_scores()
    perm_id_lists = get_perm_id_lists(len(orginal_utterances), search_space_size)
    for perm_id_list in perm_id_lists:
        centeringDiscourse.utterances = [orginal_utterances[i] for i in perm_id_list]
        centeringDiscourse.reset_discourse_properties()
        cur_CT_scores = centeringDiscourse.comput_CT_scores()
        for key in final_CT_scores.keys():
            if unnormalized_CT_scores[key] > cur_CT_scores[key]:
                final_CT_scores[key] += 1
            elif unnormalized_CT_scores[key] == cur_CT_scores[key]:
                final_CT_scores[key] += 0.5
    # normalization
    for key in final_CT_scores.keys():
        final_CT_scores[key] = 100 * final_CT_scores[key] / (len(perm_id_lists) + 1)
    return final_CT_scores, unnormalized_CT_scores

