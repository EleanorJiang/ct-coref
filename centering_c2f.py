from enum import Enum
import collections
from allennlp_models.common.ontonotes import OntonotesSentence
from allennlp.data.dataset_readers.dataset_utils.span_utils import TypedSpan
from typing import Dict, List, Optional, Tuple, DefaultDict

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

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
    def __init__(self, sentence: ConvertedSent, candidate, ranking,
                 CB=None, transition=Transition.NA, cheapness=None, coherence=None, salience=None) -> None:
        self.document_id = sentence.document_id
        self.sentence_id = sentence.sentence_id
        self.words = sentence.words
        self.pos_tags = sentence.pos_tags
        self.gram_role = sentence.gram_role
        self.srl = sentence.srl
        ## TODO: Here to modify ``candidate_entities''
        if candidate == "coref_spans":
            self.candidate_entities = self._set_candidate_entities(sentence.coref_spans)
        elif candidate == "top_spans":
            self.candidate_entities = self._set_candidate_entities(sentence.top_spans)
        elif candidate == "clusters":
            self.candidate_entities = self._set_candidate_entities(sentence.clusters)
        ## TODO: Here to modify ``ranking''
        if ranking == "grl":
            self.CF_list = self._set_CF_list(sentence.pos_tags, gram_role=sentence.gram_role)
        elif ranking == "srl":
            self.CF_list = self._set_CF_list_by_srl(sentence.pos_tags, srl=sentence.srl)
        self.CB = CB
        self.transition = transition
        self.cheapness = cheapness
        self.coherence = coherence
        self.salience = salience

    def _set_candidate_entities(self, someTypeSpanLst): # coref_spans or clusters or top_spans
        return set(someTypeSpanLst)


    def _set_CF_list(self, postag, gram_role=None):
        entities: DefaultDict[str, List[TypedSpan]] = collections.defaultdict(list)
        for typed_span in self.candidate_entities:
            span_id, span = typed_span
            if len(span) == 1 and ("PRP" in postag[span[0]]):
                entities["pronuons"].append(typed_span)
                continue
            if span in gram_role["subj"]:
                entities["subj"].append(typed_span)
                continue
            if span in gram_role["dobj"]:
                entities["dobj"].append(typed_span)
                continue
            if span in gram_role["iobj"]:
                entities["iobj"].append(typed_span)
                continue
            if span in gram_role["pobj"]:
                entities["pobj"].append(typed_span)
                continue
            else:
                entities["others"].append(typed_span)
        return entities["pronuons"] + entities["subj"] + entities["dobj"] + entities["iobj"] + entities["pobj"] + \
               entities["others"]

    def _set_CF_list_by_srl(self, postag, srl=None):
        entities: DefaultDict[str, List[TypedSpan]] = collections.defaultdict(list)
        for typed_span in self.candidate_entities:
            span_id, span = typed_span
            if len(span) == 1 and ("PRP" in postag[span[0]]):
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
        return entities["PRP-ARG0"] + entities["PRP-ARG1"] + entities["PRP-other"] + entities["ARG0"] + entities["ARG1"] + \
               entities["others"]

    def set_CB_and_cheapness_coherence(self, centering_document):
        '''
        Unlike the original version of CT,
        we decide to compare Un to the nearest previous Utterance with a non-empty Cf-list
        :param centering_document:
        :return:
        '''
        # Seeking the nearest valid sentence (we are skipping utterances with empty candidate_entities, e.g. "Uh-huh.")
        for i in range(self.sentence_id - 1, -1, -1):
            previousU = centering_document[i]
            if len(previousU.CF_list):
                break
        # if no previous sentences are valid (candidate_entities/CF_list is not empty),
        # we return with CB and cheapness&coherence being None.
        if len(previousU.CF_list) == 0 or len(self.candidate_entities) == 0:
            return
        # set CB as the top ranking entity in previousU.CF_list which exists in current candidate_entities
        tmp_list = [entity[0] for entity in self.candidate_entities]
        if previousU.CF_list[0][0] in tmp_list:
            self.CB = previousU.CF_list[0]
            self.cheapness = True
            if previousU.CB and self.CB[0] != previousU.CB[0]:
                self.coherence = False
            else:
                self.coherence = True
            return
        # if CB is not equal to the top element in previousU.CF_list (CP_{n-1})
        self.cheapness = False
        # continue searching other elements in previousU.CF_list
        for typed_span in previousU.CF_list:
            if typed_span[0] in tmp_list:
                self.CB = typed_span
                if previousU.CB and self.CB[0] != previousU.CB[0]:
                    self.coherence = False
                else:
                    self.coherence = True
                return
        # if candidate_entities is not empty and previousU.CF_list is not empty but they are disjoint, then set CB to NOCB
        self.coherence = False

    def set_salience(self):
        if self.CF_list != []:
            if not self.CB:
                self.salience = False
            else:
                self.salience = (self.CB[0] == self.CF_list[0][0])

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
        #      print("Error in set_transition:", self.transition, self.coherence, self.salience, self.CF_list, self.candidate_entities)


def OntoSentence2ConvertedSent(sentence: OntonotesSentence, sentence_id, gram_role=None, srl=None):
    return ConvertedSent(document_id=sentence.document_id, sentence_id=sentence_id,
                         words=sentence.words, coref_spans=sentence.coref_spans, pos_tags=sentence.pos_tags,
                         gram_role=gram_role, srl=srl)


from collections import Counter
def statistics(centering_document):
    cnt = Counter()
    cnt["num_all_u"] = len(centering_document)
    cnt["num_valid_u"], cnt["num_nocb"], cnt["num_cheapness"], cnt["num_coherence"], cnt["num_salience"], cnt[
        "num_continue"], cnt["num_retain"], cnt["num_s_shift"], cnt["num_r_shift"] = 0,0,0,0,0,0,0,0,0
    for centeringUtterance in centering_document:
        if len(centeringUtterance.candidate_entities) == 0:
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
    for all the scores, higher == more coherence
    '''
    df['num_kp'] = (df['num_valid_u'] - df['num_nocb'] + df['num_cheapness'] + df['num_coherence'] + df['num_salience']) / 4
    df['%valid_u'] = df['num_valid_u']/df['num_all_u']*100
    df['%not_nocb'] = (1-df['num_nocb']/df['num_valid_u'])*100
    df['%cheapness'] = df['num_cheapness']/df['num_valid_u']*100
    df['%coherence'] = df['num_coherence']/df['num_valid_u']*100
    df['%salience'] = df['num_salience']/df['num_valid_u']*100
    df['%continue'] = df['num_continue']/(df['num_valid_u']-1)*100
    df['%retain'] = df['num_retain']/(df['num_valid_u']-1)*100
    df['%s_shift'] = df['num_s_shift']/(df['num_valid_u']-1)*100
    df['%r_shift'] = df['num_r_shift']/(df['num_valid_u']-1)*100
    df['%kp'] = (df['%not_nocb'] + df['%cheapness']+ df['%coherence'] + df['%salience']) /4

# 2.1.4
import random



def shuffle_list(some_lists):
    randomized_list = some_lists[-1][:]
    flag = False
    i = 0
    while not flag:
        i+=1
        if(i>1):
            print("try {} time".format(i))
        random.seed()
        random.shuffle(randomized_list)
        for some_list in some_lists:
            if randomized_list == some_list:
                break
            flag = True
    return randomized_list



def Doc2CenterDoc(document, candidate, ranking):
    centeringUtterance = CenteringUtterance(document[0],  candidate=candidate, ranking=ranking)
    centering_document = [centeringUtterance]
    for j in range(1, len(document)):
        prviousUtterance = centeringUtterance
        centeringUtterance = CenteringUtterance(document[j],candidate=candidate, ranking=ranking)
        centeringUtterance.sentence_id = j
        centeringUtterance.set_CB_and_cheapness_coherence(centering_document)
        centeringUtterance.set_salience()
        centeringUtterance.set_transition()
        centering_document.append(centeringUtterance)
    return centering_document


def get_centering_search_spaces(documents, search_space_size, candidate, ranking):
    centering_search_spaces = []
    for i, document in enumerate(documents):
        print("generate random sample space for the {}-th document".format(i))
        centering_search_space = [
            Doc2CenterDoc(document, candidate=candidate, ranking=ranking)]  # the centered original_document as centering_search_space[0]
        doc_search_spaces = [document]
        for k in range(1, search_space_size, 1):
            random_document = shuffle_list(doc_search_spaces)
            doc_search_spaces.append(random_document)
            centering_search_space.append(Doc2CenterDoc(random_document, candidate=candidate, ranking=ranking))
        centering_search_spaces.append(centering_search_space)
    return centering_search_spaces


def compute_classification_rate(centering_search_space, search_space_size, M: str):
    '''
    :param centering_search_space:  a list of random sampled centering documents from one original document
    :param search_space_size: len(centering_search_space)
    :param M: string, the metric, '%not_nocb' or '%cheapness' or 'transition'
                                  'kp'(sum up '%not_nocb'&'%cheapness'&'%coherence'&'%salience')
    :return: float [0,1], the classification rate, the higher, the better
    '''
    original_doc_score = centering_search_space[0]
    df = pd.concat([statistics(centering_document) for centering_document in centering_search_space], ignore_index=True,
                   sort=False)
    more_stastics(df)
    if M == 'transition':
        return compute_classification_rate_by_transition(df,search_space_size)
    worse = df[df[M] < df[M].get(0)].count()[0]
    equal = df[df[M] == df[M].get(0)].count()[0]
    classification_rate = (worse + equal / 2) / search_space_size
    return classification_rate


def compute_classification_rate_by_transition(df,search_space_size):
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

