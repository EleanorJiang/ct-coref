import os, pickle
import centering_c2f
import pandas as pd

def converted_to_centering(converted_documents,
                           candidate="top_spans", ranking="grl",
                           save=False, archive_path=None):
    '''
    Get centering_documents: Convert ConvertedSent to CenteringUtterance
    :param converted_documents:
    :param candidate:
    :param ranking:
    :param save:
    :param archive_path:
    :return: centering_documents
    '''
    centering_documents = []
    for i, document in enumerate(converted_documents):
        centeringUtterance = centering_c2f.CenteringUtterance(document[0], candidate=candidate, ranking=ranking)
        centering_document = [centeringUtterance]
        for j in range(1, len(document)):
            prviousUtterance = centeringUtterance
            centeringUtterance = centering_c2f.CenteringUtterance(document[j], candidate=candidate, ranking=ranking)
            centeringUtterance.set_CB_and_cheapness_coherence(centering_document)
            centeringUtterance.set_salience()
            centeringUtterance.set_transition()
            centering_document.append(centeringUtterance)
        centering_documents.append(centering_document)
    print("centering_documents_for_c2f down!")
    if save:
        with open(os.path.join(archive_path, 'centering_documents.data'), 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(centering_documents, filehandle)
    return centering_documents


def get_Table_1(converted_documents, name, archive_path,
                candidate="top_spans", ranking="grl"):
    centering_documents = converted_to_centering(converted_documents,
                                                 candidate=candidate, ranking=ranking,
                                                 save=False, archive_path=archive_path)
    pd.options.display.float_format = '{:.2f}'.format
    result = pd.concat(
        [centering_c2f.statistics(centering_document) for centering_document in centering_documents],
        ignore_index=True, sort=False)
    centering_c2f.more_stastics(result)
    print("Table 1 of {} {}:".format(archive_path, name))
    print(result.describe())
    with open(os.path.join(archive_path, "{}_{}_{}_Table_1".format(name, candidate, ranking)), 'w') as f:
        print("Table 1 of {}:".format(name), file=f)
        print(result.describe(), file=f)


def get_Table_2(converted_documents, name, archive_path,
                candidate="top_spans", ranking="grl", search_space_size=100):
    centering_search_spaces = centering_c2f.get_centering_search_spaces(converted_documents, search_space_size,
                                                                        candidate=candidate, ranking=ranking)
    metrics = ['%not_nocb', '%cheapness', '%coherence', '%salience', '%kp', 'transition']
    aver_classif_rates = dict()
    for M in metrics:
        classification_rates = [centering_c2f.compute_classification_rate(centering_search_space, search_space_size, M)
                                for centering_search_space in centering_search_spaces]
        aver_classif_rate = sum(classification_rates) / len(classification_rates) * 100
        aver_classif_rates[M] = aver_classif_rate
        print("{} is done".format(M))

    df_aver_classif_rates = pd.DataFrame.from_dict(aver_classif_rates, orient='index').reset_index()
    print("Table 2 of {} {}:".format(archive_path, name))
    print(df_aver_classif_rates)
    with open(os.path.join(archive_path, "{}_{}_{}_Sample_{}_Table_2"
            .format(name, candidate, ranking, search_space_size)), 'w') as f:
        print("Table 2 of {} with search_space_size {}:".format(name, search_space_size), file=f)
        print(df_aver_classif_rates, file=f)