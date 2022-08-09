import spacy, os, pickle
from ontonotes import Ontonotes, OntonotesSentence, TypedSpan
import collections
from typing import Dict, List, Optional, Tuple, DefaultDict, NamedTuple, Union, Dict, Any, Optional
from conll_util import find_span


def readOntoNotes_full(file_path):
    '''
    :param file_path:
    :return: documents: List[List[OntonotesSentence]]
            gram_roles: List[List[DefaultDict[str, Span]]]
                        for each sentence, we have DefaultDict[str, Span]
                        e.g.  {'subj': [(16, 16), (19, 23)],
                               'dobj': [(25, 27), (57, 59)],
                                'iobj': [(25, 27), (57, 59)]
                                'pobj': [(25, 27)]}
            mention_masks: tokens being part of a mention are maksed with entity_id
                          List[List[list[int]]]     [[None,116,None,None,None, ...],[],[], ...]
            document_lens: the length of each document
                            List[int] [8,12,23,24,...]
            doc_ids: the document-level token_id
                        List[List[List[int]]]
                        e.g. [[0,1,2,3,...],[14,15,16,...],[], ...]
            clusters_info: a document
                        spans are document-level indexed
                        List[DefaultDict[List[Span]]]]
                        where Span = Tuple[int,int],
                        i.e. entity_id:  [(start, end), (start, end), (start, end)]
                        e.g. [{42: [(16, 16), (19, 23)], 32: [(25, 27), (57, 59)],...},{},{}]
    '''
    spacy.load('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')
    ontonotes_reader = Ontonotes()
    documents = []
    clusters_info = []
    document_lens = []
    gram_roles = []
    mention_masks = []
    doc_ids = []
    for sentences in ontonotes_reader.dataset_document_iterator(file_path):
        clusters: DefaultDict[int, List[Tuple[int, int]]] = collections.defaultdict(list)
        total_tokens = 0
        gram_role_doc = []
        mention_mask_doc = []
        doc_id_doc = []
        for sentence in sentences:
            gram_role: DefaultDict[str, List[Tuple[int, int]]] = collections.defaultdict(list)
            mention_mask = [None] * len(sentence.words)
            doc_id = [0] * len(sentence.words)
            for i, token in enumerate(sentence.words):
                doc_id[i] = total_tokens + i
            for typed_span in sentence.coref_spans:  # TypedSpan = Tuple[int, Tuple[int, int]]
                # Coref annotations are on a _per sentence_
                # basis, so we need to adjust them to be relative
                # to the length of the document.
                entity_id, (start, end) = typed_span
                clusters[entity_id].append((start + total_tokens, end + total_tokens))
                for i, token in enumerate(sentence.words):
                    if start <= i <= end:
                        mention_mask[i] = entity_id
            total_tokens += len(sentence.words)
            doc = nlp(' '.join(sentence.words))
            tokens = [t.text for t in doc]
            token_id, cursor  = [], 0
            for i,token in enumerate(tokens):
                if token in sentence.words[cursor]:
                    token_id.append(token)
                else:
                    print("impossible")
                if tokens[i+1] not in sentence.words[cursor]:
                    cursor += 1
            for token in doc:
                if 'subj' in token.dep_:
                    start, end = find_span(token)
                    gram_role['subj'].append((start, end))
                if 'dobj' in token.dep_:
                    start, end = find_span(token)
                    gram_role['dobj'].append((start, end))
                if 'iobj' in token.dep_:
                    start, end = find_span(token)
                    gram_role['iobj'].append((start, end))
                if 'pobj' in token.dep_:
                    start, end = find_span(token)
                    gram_role['pobj'].append((start, end))

            ## If the tokenization got from spacy is not the same with the original annotation, we do a mapping here ##
            tokens = [t.text for t in doc]
            if len(tokens) != len(sentence.words):
                token_id, cursor = [-1] * len(tokens), 0
                if tokens[0] not in sentence.words[0]:
                    print(tokens, sentence.words)
                for k, token in enumerate(tokens):
                    while token not in sentence.words[cursor] or (token == '.' and sentence.words[cursor] == 'a.m'):
                        cursor += 1
                        if cursor == len(sentence.words):
                            print(k, token_id, tokens, sentence.words)
                    token_id[k] = cursor
                # Sanity Check: Every ``entence.word'' should be covered && Every ``token_id'' should be assigned.
                if cursor != len(sentence.words) - 1:
                    print(len(tokens) == len(sentence.words), tokens, sentence.words)
                for s in token_id:
                    if s == -1:
                        raise AssertionError("Sanity check fails: Every ``entence.word'' should be covered && "
                                             "Every ``token_id'' should be assigned.")
                for key, value in gram_role.items():
                    for l, (start, end) in enumerate(value):
                        gram_role[key][l] = (token_id[start], token_id[end])
            ######################################################
            gram_role_doc.append(gram_role)
            mention_mask_doc.append(mention_mask)
            doc_id_doc.append(doc_id)
        gram_roles.append(gram_role_doc)
        mention_masks.append(mention_mask_doc)
        document_lens.append(total_tokens)
        doc_ids.append(doc_id_doc)
        clusters_info.append(clusters)
        documents.append(sentences)
    return documents, gram_roles, mention_masks, document_lens, doc_ids, clusters_info


def readOntoNotes_with_srl_anotation(documents):
    '''
    :param documents: 2-d list, the documents got by ``prepare_data_full``, or loaded from 'documents.data'
    :return documents_with_srl_anotation:  a 2-d list of OntoNoteSentence, the same structure with ``documents``
            srls: a 2-d list of DefaultDict[str, List[Tuple[int, int]]], {'ARG0': [], 'ARG1':[] }
            chosen_ids: a list of int, corresponding to the indices of chosen documents in the full ``documents''
    '''
    doc_ids = collections.defaultdict(int)
    new_documents = []
    for i, document in enumerate(documents):
        for j, sentence in enumerate(document):
            if len(sentence.coref_spans) and not sentence.srl_frames:
                doc_ids[i] += 1
    # Filter out those documents without srl annotation
    chosen_ids = []
    for i in range(348):
        if i not in doc_ids.keys():
            chosen_ids.append(i)
            new_documents.append(documents[i])

    # Get srl.data
    srls = []
    for i, document in enumerate(new_documents):
        srl_doc = []
        for j, sentence in enumerate(document):
            srl: DefaultDict[str, List[Tuple[int, int]]] = collections.defaultdict(list)
            for srl_frame in sentence.srl_frames:
                label = srl_frame[1]
                for start in range(len(label)):
                    if label[start] == 'B-ARG0':
                        for end in range(start + 1, len(label) + 1, 1):
                            if end == len(label) or label[end] == 'O':
                                while not (label[end - 1] == 'I-ARG0' or label[end - 1] == 'B-ARG0'):
                                    end -= 1
                                srl['ARG0'].append((start, end - 1))
                                break
                for start in range(len(label)):
                    if label[start] == 'B-ARG1':
                        for end in range(start + 1, len(label) + 1, 1):
                            if end == len(label) or label[end] == 'O':
                                while not (label[end - 1] == 'I-ARG1' or label[end - 1] == 'B-ARG1'):
                                    end -= 1
                                srl['ARG1'].append((start, end - 1))
                                break
            #         if not len(srl) and len(sentence.coref_spans):
            #             print(i,j,sentence.words, srl, sentence.srl_frames, sentence.coref_spans)
            srl_doc.append(srl)
        srls.append(srl_doc)
    return new_documents, srls, chosen_ids


def prepare_data_full(dataset_path, save_path):
    """
    1. Prepare all .data
    """
    documents, gram_roles, mention_masks, document_lens, doc_ids, clusters_info = readOntoNotes_full(dataset_path)
    with open(os.path.join(save_path, 'documents.data'), 'wb') as filehandle:
        pickle.dump(documents, filehandle)
    with open(os.path.join(save_path, 'gram_roles.data'), 'wb') as filehandle:
        pickle.dump(gram_roles, filehandle)
    return documents, gram_roles


def prepare_data_with_srl_anotation(dataset_path, save_path):
    if not os.path.exists(os.path.join(save_path, 'documents.data')):
        documents, srls, chosen_ids = readOntoNotes_with_srl_anotation(dataset_path)
    else:
        with open(os.path.join(save_path, 'documents.data'), 'rb') as filehandle:
            documents = pickle.load(filehandle)
    documents_with_srl_anotation, srls, chosen_ids = readOntoNotes_with_srl_anotation(documents)
    with open(os.path.join(save_path, 'documents_with_srl_anotation.data'), 'wb') as filehandle:
        pickle.dump(documents_with_srl_anotation, filehandle)
    with open(os.path.join(save_path, 'srls.data'), 'wb') as filehandle:
        pickle.dump(srls, filehandle)