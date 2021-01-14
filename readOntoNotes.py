import spacy
from allennlp_models.common.ontonotes import Ontonotes
import collections
from typing import Dict, List, Optional, Tuple, DefaultDict, NamedTuple, Union, Dict, Any, Optional
from pathlib import Path
from allennlp.common import plugins
from allennlp.models.archival import Archive, load_archive
from allennlp.predictors.predictor import Predictor
from allennlp.data.dataset_readers.dataset_utils.span_utils import TypedSpan


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

def readOntoNotes_full(file_path):
    '''
    :param file_path:
    :return: documents,
            gram_roles, list[list[DefaultDict(str:tuple(int,int))]]
            mention_masks, list[list[list[int]]]     [[None,116,None,None,None, ...],[],[], ...]
            document_lens, list[int] [8,12,23,24,...]
            doc_ids, list[list[list[int]]]     [[0,1,2,3,...],[14,15,16,...],[], ...]
            clusters_info, list[list[sentence]]
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
    num_utterance_with_nosrl = 0
    for sentences in ontonotes_reader.dataset_document_iterator(file_path):
        clusters: DefaultDict[int, List[Tuple[int, int]]] = collections.defaultdict(list)
        total_tokens = 0
        #     if len(sentences[0].srl_frames) == 0:
        #         #assert len(sentences[1].srl_frames) == 0
        #         num_utterance_with_nosrl +=1
        #         continue
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
                span_id, (start, end) = typed_span
                clusters[span_id].append((start + total_tokens, end + total_tokens))
                for i, token in enumerate(sentence.words):
                    if start <= i <= end:
                        mention_mask[i] = span_id
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
                        print('bad!')
                for key, value in gram_role.items():
                    for l, (start, end) in enumerate(value):
                        gram_role[key][l] = (token_id[start], token_id[end])
            ######################################################
            gram_role_doc.append(gram_role)
            mention_mask_doc.append(mention_mask)
            doc_id_doc.append(doc_id)
        gram_roles.append(gram_role_doc)  # list[list[DefaultDict(str:tuple(int,int))]]
        mention_masks.append(mention_mask_doc)  # list[list[list[int]]]     [[None,116,None,None,None, ...],[],[], ...]
        document_lens.append(total_tokens)  # list[int] [8,12,23,24,...]
        doc_ids.append(doc_id_doc)  # list[list[list[int]]]     [[0,1,2,3,...],[14,15,16,...],[], ...]
        clusters_info.append(
            clusters)  # list[DefaultDict(int:tuple(int,int))]  [{42: [(16, 16), (19, 23)], 32: [(25, 27), (57, 59)],...},{},{}]
        documents.append(sentences)  # list[list[sentence]]
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
    for i, document in enumerate(documents):
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


# 1.2 Get Prediction of c2f-coref Model

# predictor is a CorefPredictor

def get_prediction_from_model(documents, predictor):
    '''
    :param documents:
    :return: predicted_dicts, [#Doc, #Sent] of dict('top_spans', 'antecedent_indices', 'predicted_antecedents', 'document', 'clusters')
            'top_spans': list of pairs (start,end)
            'antecedent_indices'
            'predicted_antecedents': list of int, len(predicted_dict['predicted_antecedents']) == len(predicted_dict['top_spans']),
                                     -1 is no_antecedent, or the index of its antecedent in top_spans
            'document': list of string (words)
            'clusters: list of pairs (start,end), a subset of 'top_spans' (with 'predicted_antecedents' being 1)
                        predicted_dict['top_spans'][predicted_dict['predicted_antecedents']] is illegal.
    '''
    predicted_dicts = []
    for document in documents:
        texts = [' '.join(map(str, sentence.words)) for sentence in document]
        text = ' '.join(map(str, texts))
        predicted_dict = predictor.predict(document=text)
        predicted_dicts.append(predicted_dict)
    return predicted_dicts

def load_predictor_from_th(
        archive_path: Union[str, Path],
        weights_file: str,
        predictor_name: str = None,
        cuda_device: int = 0,
        dataset_reader_to_load: str = "validation",
        frozen: bool = True,
        import_plugins: bool = True,
        overrides: Union[str, Dict[str, Any]] = "",
) -> "Predictor":
    if import_plugins:
        plugins.import_plugins()
    return Predictor.from_archive(
        load_archive(archive_path, cuda_device=cuda_device, overrides=overrides, weights_file=weights_file),
        predictor_name,
        dataset_reader_to_load=dataset_reader_to_load,
        frozen=frozen,
    )