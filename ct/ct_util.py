import os, pickle
import centering
import pandas as pd


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
    gram role should be chosen from the following list:
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
        gram_role['subj'].append(((start, token.i, end), (hstart, token.head.i, hend)))
    if 'obj' in token.dep_:
        start, end = find_span(token)
        hstart, hend = find_span(token.head)
        gram_role['obj'].append(((start, token.i, end), (hstart, token.head.i, hend)))


def get_Table_1(converted_documents, name, archive_path, result_dir, paras=(0, None, 0),
                candidate="top_spans", ranking="grl", override=True):
    # First, check whether both Table 1 and csv file (394 rows) exists!!!
    data_dir = '/'.join(archive_path.split("/")[:-2])
    experiment_id = archive_path.split("/")[-1]
    save_path = os.path.join(data_dir, result_dir, experiment_id)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    if (not override) \
        and os.path.exists(os.path.join(save_path, "{}_{}_{}_Table_1_per_document.csv".format(name, candidate, ranking))) \
        and os.path.exists(os.path.join(save_path, "{}_{}_{}_Table_1.csv".format(name, candidate, ranking))):
        print("both {} and {} exist, no need to run get_Table_1() anynmore!".format(
            os.path.join(save_path, "{}_{}_{}_Table_1_per_document.csv".format(name, candidate, ranking)),
            os.path.join(save_path, "{}_{}_{}_Table_1".format(name, candidate, ranking))
        ))
        return

    # Then checked whether we have spacy-processed `converted documents`
    if os.path.isfile(os.path.join(save_path, f'{name}_documents.data')):
        with open(os.path.join(save_path, f'{name}_documents.data'), 'rb') as filehandle:
            centering_documents = pickle.load(filehandle)
    else:
        centering_documents = [centering.converted_to_centering(converted_document,  paras=paras,
                                candidate=candidate, ranking=ranking) for converted_document in converted_documents]
        with open(os.path.join(save_path, f'{name}_documents.data'), 'wb') as filehandle:
            pickle.dump(centering_documents, filehandle)

    # Let the format of all percentages to be xx.xx%
    pd.options.display.float_format = '{:.2f}'.format
    # Concatante all documents together, so we will have 384 rows, each row is the statistics of one document (see statistics and more statistics for the names of columns)
    result = pd.concat(
        [centering.statistics(centering_document, doc_id) for doc_id, centering_document in enumerate(centering_documents)],
        ignore_index=True, sort=False)
    centering.more_stastics(result)
    print("Table 1 of {} {}:".format(archive_path, name))
    print(result.describe())
    result.describe().to_csv(os.path.join(save_path, "{}_{}_{}_Table_1.csv".format(name, candidate, ranking)), sep=",", header=True, index=True)
    # save the classification rate of every documents (i.g. save the result table)
    result.to_csv(os.path.join(save_path, "{}_{}_{}_Table_1_per_document.csv".format(name, candidate, ranking)), sep=",", header=True, index=True)


def get_Table_2(converted_documents, name, archive_path, result_dir, paras=(0, None, 0),
                candidate="clusters", ranking="grl", search_space_size=100, id=0, override=True):
    # First, check whether both Sample{}_id{}_Table_2 and Sample{}_id{}_Table_2_per_document exists!!!
    data_dir = '/'.join(archive_path.split("/")[:-2])
    experiment_id = archive_path.split("/")[-1]
    save_path = os.path.join(data_dir, result_dir, experiment_id)
    if (not override) \
            and os.path.exists(os.path.join(save_path, "{}_{}_{}_Sample{}_id{}_Table_2.csv"
            .format(name, candidate, ranking, search_space_size, id))) \
            and os.path.exists(os.path.join(save_path, "{}_{}_{}_Sample{}_id{}_Table_2_per_document.csv"
            .format(name, candidate, ranking, search_space_size, id))):
        print("both {} and {} exist, no need to run get_Table_2() anynmore!".format(
            os.path.join(save_path, "{}_{}_{}_Sample{}_id{}_Table_2"
                         .format(name, candidate, ranking, search_space_size, id)),
            os.path.join(save_path, "{}_{}_{}_Sample{}_id{}_Table_2_per_document"
                         .format(name, candidate, ranking, search_space_size, id))
        ))
        return
    centering_search_spaces = centering.get_centering_search_spaces(converted_documents, search_space_size,
                           paras=paras, candidate=candidate, ranking=ranking)
    # 6 CT-based scores:
    metrics = ['%not_nocb', '%cheapness', '%coherence', '%salience', '%kp', 'transition']
    aver_classif_rates = dict()
    # classification_rates_list is a dict, where the keys are 6 centering-based metrics,
    # the value is a list (of length 348) of classification_rates
    # e.g. classification_rates_list["%not_nocb"][i] = 0.889
    classification_rates_list = dict()
    for M in metrics:
        classification_rates = [centering.compute_CT_score(doc_id, centering_search_space, search_space_size, M)
                                for doc_id, centering_search_space in enumerate(centering_search_spaces)]
        aver_classif_rate = sum(classification_rates) / len(classification_rates) * 100
        aver_classif_rates[M] = aver_classif_rate
        classification_rates_list[M] = classification_rates

    df_aver_classif_rates = pd.DataFrame.from_dict(aver_classif_rates, orient='index').reset_index()
    df_classification_rates_per_dococument = pd.DataFrame.from_dict(classification_rates_list)
    # print Table 2 into the file
    print("Table 2 of {} {}:".format(archive_path, name))
    print(df_aver_classif_rates)
    df_aver_classif_rates.to_csv(os.path.join(save_path, "{}_{}_{}_Sample{}_id{}_Table_2.csv"
            .format(name, candidate, ranking, search_space_size, id)), sep=",", header=True, index=True)
    df_classification_rates_per_dococument.to_csv(os.path.join(save_path, "{}_{}_{}_Sample{}_id{}_Table_2_per_document.csv"
            .format(name, candidate, ranking, search_space_size, id)), sep=",", header=True, index=True)