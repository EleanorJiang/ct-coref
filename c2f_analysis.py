import pickle, os, collections, spacy
from typing import List, Tuple, DefaultDict
os.chdir('/cluster/work/cotterell/ct/code/centering_exp')
from readOntoNotes import find_span
from ct_util import *

# archive_path = "/cluster/project/infk/cotterell/ct/data/centering_exp/coref-spanbert-base-2020.12.20.10"
spacy.load('en_core_web_sm')
nlp = spacy.load('en_core_web_sm')

def c2f_analysis(archive_path):
    for file in os.listdir(archive_path):
        if not file.endswith("_predicted_dicts.data"):
            continue
        name = file.split("_predicted_dicts")[0]
        if os.path.isfile(os.path.join(archive_path, "{}_Table_2".format(name))):
            continue
        c2f_analysis_for_predicted_dicts(archive_path, file)


def c2f_analysis_for_predicted_dicts(archive_path, predicted_dict_file, search_space_size,
                                     candidate="clusters"):

    name = predicted_dict_file.split("_predicted_dicts")[0]
    converted_documents_for_c2f = predict_dicts_to_converted_documents(
                                    archive_path, predicted_dict_file, name, save=False)
    # Get Table 1
    get_Table_1(converted_documents_for_c2f, name, archive_path,
                candidate=candidate, ranking="grl")

    # Get Table 2
    get_Table_2(converted_documents_for_c2f, name, archive_path,
                candidate=candidate, ranking="grl", search_space_size=search_space_size)


def predict_dicts_to_converted_documents(archive_path, predicted_dict_file, name, save=False):

    # 1. Load the ``predicted_dicts.data'' in ``archive_path''
    assert os.path.exists(os.path.join(archive_path, "predicted_dicts.data")), \
        "If there is no ``predicted_dicts.data''，please produce ``predicted_dicts.data'' first!!!!!"
    with open(os.path.join(archive_path, predicted_dict_file), 'rb') as filehandle:
        predicted_dicts = pickle.load(filehandle)
    print(predicted_dict_file, len(predicted_dicts))
    if len(predicted_dicts[0]['document']) == 1:
        print("Unsqueezing 'documents'")

    # 2. Before feeding predicted_dicts[i]['document'] into CenteringUtterance(),
    # first let's convert predicted_dicts[i]['document'] into predicted_dicts[i][sent_id] of tokens:
    # flat to 2d, add one more layer ``sentence'' to the nested list
    converted_documents_for_c2f = []
    for i in range(len(predicted_dicts)):
        converted_document = []
        flat_document = []
        text = ""
        # for word in predicted_dicts[i]['document'][0]:
        #     if word.startswith("##"):
        #         word = word
        if len(predicted_dicts[i]['document']) == 1:
            predicted_dicts[i]['document'] = predicted_dicts[i]['document'][0]
        doc = nlp(' '.join(predicted_dicts[i]['document']))
        for j, sentence in enumerate(doc.sents):  # doc.sents: list of strings, sent: a string
            words, pos_tags = [], []  # list of tokens or pos_tag
            gram_role: DefaultDict[str, List[Tuple[int, int]]] = collections.defaultdict(list)
            srl: DefaultDict[str, List[Tuple[int, int]]] = collections.defaultdict(list)
            tmp_doc = nlp(sentence.text)
            for token in tmp_doc:
                words.append(token.text)
                pos_tags.append(token.tag_)
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
            converted_sent = centering_c2f.ConvertedSent(document_id=i, sentence_id=j, words=words, pos_tags=pos_tags,
                                                         gram_role=gram_role, srl=srl)
            converted_document.append(converted_sent)
            flat_document += words
        converted_documents_for_c2f.append(converted_document)
        # Sanity Check: Check whether the total #tokens in one document is the same after getting grl by spacy
        if len(flat_document) != len(predicted_dicts[i]['document']):
            print("#token of the {}-th flat document is {}, while the original is {}".format(i, len(flat_document), len(
                predicted_dicts[i]['document'])))

    # 3. Get ``converted_sent.clusters'' (those mention spans with coref links) and ``converted_sent.top_spans''
    # (singletons + ``clusters'', that is, all detected mention spans).
    # Both ``converted_sent.clusters'' and ``converted_sent.top_spans'' are lists of typed_span (entity_id, (start, end))
    for i, converted_document in enumerate(converted_documents_for_c2f):
        # First step:
        enity_list_from_c2f, enity_list_from_c2f_only_singleton = [], []
        k=-1 # Deal with situations that nothing in ``clusters'' but something in ``top_spans''
        for k, cluster in enumerate(predicted_dicts[i]['clusters']):
            enity_list_from_c2f.extend([(k, tuple(span)) for span in cluster])
        # get singletons from 'top_spans' and add them to enity_list_from_c2f_only_singleton
        for span in predicted_dicts[i]['top_spans']:
            if tuple(span) not in [typeSpan[-1] for typeSpan in enity_list_from_c2f]:
                k += 1
                enity_list_from_c2f_only_singleton.append((k, tuple(span)))
        #############################################################
        first_token = 0
        span_cursor, span_cursor_2 = 0, 0
        for j, converted_sent in enumerate(converted_document):
            converted_sent.offset = first_token
            converted_sent.clusters, converted_sent.top_spans = [], []
            next_first_token = first_token + len(converted_sent.words)
            for typed_span in enity_list_from_c2f:
                span_id, (start, end) = typed_span
                if start >= first_token and end < next_first_token:
                    converted_sent.clusters.append((span_id, (start - first_token, end - first_token)))
            while span_cursor_2 < len(enity_list_from_c2f_only_singleton) and \
                    enity_list_from_c2f_only_singleton[span_cursor_2][-1][0] >= first_token and \
                    enity_list_from_c2f_only_singleton[span_cursor_2][-1][1] < next_first_token:
                span_id, (start, end) = enity_list_from_c2f_only_singleton[span_cursor_2]
                converted_sent.top_spans.append((span_id, (start - first_token, end - first_token)))
                span_cursor_2 += 1
            first_token = next_first_token
            converted_sent.top_spans.extend(converted_sent.clusters)
    print("converted_documents_for_c2f down for {}!".format(predicted_dict_file))
    if save:
        with open(os.path.join(archive_path,'{}_converted_documents_for_c2f.data'.format(name)), 'wb') as filehandle:
            pickle.dump(converted_documents_for_c2f, filehandle)
    return converted_documents_for_c2f