"""
Usage:
 1. python c2f_analysis.py

 2. Jupyter Notebook:
    import os
    import c2f_analysis
    c2f_analysis.c2f_analysis_for_predicted_dicts(archive_path, file, search_space_size=100,
                candidate="clusters",repeating_times=10)
    where:
        archive_path = "/cluster/work/cotterell/ct/data/centering_exp/coref-spanbert-base-2021.1.10.100"
        th_name = "model_state_epoch_33"
        file = os.path.join(archive_path, th_name, "_predicted_dicts.data")

  3. Batch operation without repeating producing existing Table_2 files: c2f_analysis.c2f_analysis(archive_path).
"""

from ct_util import *
import argparse
import pickle, os, collections, spacy
from typing import List, Tuple, DefaultDict

nlp = spacy.load('en_core_web_sm')


def c2f_analysis(archive_path, result_dir, override=True,
                 candidate="clusters", ranking="grl", paras=None,
                 skip_t1=False, skip_t2=False, repeating_times=1, epochs=None):
    for file in os.listdir(archive_path):
        if not file.endswith("_predicted_dicts.data"):
            continue
        name = file.split("_predicted_dicts")[0]
        # if os.path.isfile(os.path.join(archive_path, "{}_Table_1".format(name))):
        #     continue
        if epochs is not None and whether_in_the_list(name, epochs) is False:
            continue
        c2f_analysis_for_predicted_dicts(archive_path, result_dir, file, search_space_size=100,
                                     candidate=candidate, ranking=ranking,
                                         paras=paras,
                                         repeating_times=repeating_times,
                                     override=override, skip_t1=skip_t1, skip_t2=skip_t2)


def c2f_analysis_for_predicted_dicts(archive_path, result_dir, predicted_dict_file, search_space_size=100,
                                     candidate="clusters", ranking="grl",
                                     paras=None,
                                     repeating_times=1,
                                     override=True, skip_t1=False, skip_t2=False):
    """
    This fuction provides a pipeline of
        1. obtaining "model_state_epoch_{x}_converted_documents_for_c2f.data"
        2. producing "model_state_epoch_{x}_Table_1.csv"
        3. producing "model_state_epoch_{x}_Sample100_id{x}_Table_2" for repeating_times, e,g, 10
    Usage example: See above.
    """
    # 0. Logging: the begining of this program
    # th_name is somthing like "model_state_epoch_33"
    th_name = predicted_dict_file.split("_predicted_dicts")[0]  # "model_state_epoch_x"

    print("Running `c2f_analysis` for", archive_path, th_name, "\tSample ", search_space_size)

    # 1. Reproduce converted_documents (lines in else)
    # Or use existing one, e.g. "model_state_epoch_6_converted_documents_for_c2f.data" (lines in if)
    if not os.path.exists(os.path.join(archive_path, "{}_predicted_dicts.data".format(th_name))):
        converted_documents_for_c2f = predict_dicts_to_converted_documents(
                                        archive_path, predicted_dict_file, th_name, save=True)
    else:
        with open(os.path.join(archive_path, '{}_converted_documents_for_c2f.data'.format(th_name)), 'rb') as filehandle:
            converted_documents_for_c2f = pickle.load(filehandle)
    # 2. Get Table 1
    # If you are producing a huge batch of Table 1 and can make sure the program is 100% correct, use the following lines
    # to avoid rerunning.
    #         if os.path.isfile(os.path.join(archive_path, "{}_{}_{}_Table_1".format(th_name, candidate, ranking))):
    #             print(os.path.join(archive_path, "{}_{}_{}_Table_1".format(th_name, candidate, ranking)), " exists")
    #         else:
    if not skip_t1:
        get_Table_1(converted_documents_for_c2f, th_name, archive_path, result_dir,
                    candidate=candidate, ranking=ranking, paras=paras, override=override)

    # 3. Get Table 2
    if not skip_t2:
        for id in range(repeating_times):
            get_Table_2(converted_documents_for_c2f, th_name, archive_path, result_dir,
                    candidate=candidate, ranking=ranking, paras=paras,
                        search_space_size=search_space_size, id=id, override=override)




def predict_dicts_to_converted_documents(archive_path, predicted_dict_file, name, save=True):

    # 1. Load the ``predicted_dicts.data'' in ``save_path''
    assert os.path.exists(os.path.join(archive_path, "{}_predicted_dicts.data".format(name))), \
        "If there is no ``predicted_dicts.data''，please produce ``predicted_dicts.data'' first!!!!!"
    with open(os.path.join(archive_path, predicted_dict_file), 'rb') as filehandle:
        predicted_dicts = pickle.load(filehandle)
    print(predicted_dict_file, len(predicted_dicts))

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
            gram_role: DefaultDict[str, List[Tuple[Tuple[int, int], Tuple[int, int]]]] = collections.defaultdict(list)
            srl: DefaultDict[str, List[Tuple[int, int]]] = collections.defaultdict(list)
            tmp_doc = nlp(sentence.text)
            for token in tmp_doc:
                words.append(token.text)
                pos_tags.append(token.tag_)
                decide_gram_role(gram_role, token)
            converted_sent = centering.ConvertedSent(document_id=i, sentence_id=j, words=words, pos_tags=pos_tags,
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
    print("Converted documents to centering documents for {}: done".format(predicted_dict_file))
    if save:
        with open(os.path.join(archive_path,'{}_converted_documents_for_c2f.data'.format(name)), 'wb') as filehandle:
            pickle.dump(converted_documents_for_c2f, filehandle)
    return converted_documents_for_c2f


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", help="data_dir", type=str, default="/cluster/work/cotterell/ct/data/centering_exp"
    )
    parser.add_argument("-r",
        "--result-dir", help="the name that the result dir in each experiment dir should be"
                             " e.g. result_220627"
                             " Default: result_default", type=str, default="result_default"
    )
    parser.add_argument("-e",
        "--experiment-ids", nargs="+", default=None, type=str,
                        help="the experiment ids, being EMBEDDING-DATA, "
                             "e.g.: naive-2021.6.6 no-embedding-2021.5.6 coref-spanbert-base-2021.5.17"
    )
    parser.add_argument( "-dp", "--data-perc",
                         nargs="+", default=None, type=int,
                             help="the amount of data used in the experiment, being a percentage, i.g. 10 20 ... 100"
                                  "Default: all from 10 to 100"
    )
    parser.add_argument(
        "--epoch", nargs="+", default=None, type=str,
                        help="the epoch, being either a number in [0, 40] or best"
    )
    parser.add_argument("-p",
        "--paras", nargs="+", default=None, type=float,
                        help="gamma, big+gamma, gate_escape_factor"
    )
    args = parser.parse_args()
    archive_paths = []
    if args.data_perc is None:
        data_percentages = [10*i for i in range(1, 11)]
    else:
        data_percentages = args.data_perc
    for exp_id in args.experiment_ids:
        archive_paths += [os.path.join(args.data_dir, f"{exp_id}.{data}") for data in data_percentages]
        #  "/cluster/work/cotterell/ct/data/centering_exp/naive-2021.6.6.100"
    if args.epoch is None:
        epochs = None
    else:
        epochs = args.epoch



    for archive_path in archive_paths:
        c2f_analysis(archive_path, args.result_dir, epochs=epochs, paras=args.paras)

