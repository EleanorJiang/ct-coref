import os, pickle
import argparse
from ontonotes.ontonotes import OntonotesSentence
from ontonotes.read_ontonotes import readOntoNotes_full, readOntoNotes_with_srl_anotation
from .ct_util import *
from .centering import ConvertedSent


def OntoSentence2ConvertedSent(sentence: OntonotesSentence, sentence_id, gram_role=None, srl=None):
    return ConvertedSent(document_id=sentence.document_id, sentence_id=sentence_id,
                         words=sentence.words, coref_spans=sentence.coref_spans, pos_tags=sentence.pos_tags,
                         gram_role=gram_role, srl=srl)


# 1. Prepare all .data
def prepare_data_full(dataset_path, archive_path):
    if os.path.exists(os.path.join(archive_path, 'documents.data')) and \
            os.path.exists(os.path.join(archive_path, 'gram_roles.data')):
        with open(os.path.join(archive_path, 'documents.data'), 'rb') as filehandle:
            documents = pickle.load(filehandle)
        with open(os.path.join(archive_path, 'gram_roles.data'), 'rb') as filehandle:
            gram_roles = pickle.load(filehandle)
    else:
        documents, gram_roles, mention_masks, document_lens, doc_ids, clusters_info = readOntoNotes_full(dataset_path)
        with open(os.path.join(archive_path, 'documents.data'), 'wb') as filehandle:
            pickle.dump(documents, filehandle)
        with open(os.path.join(archive_path, 'gram_roles.data'), 'wb') as filehandle:
            pickle.dump(gram_roles, filehandle)
    return documents, gram_roles


def prepare_data_with_srl_anotation(dataset_path,archive_path):
    if not os.path.exists(os.path.join(archive_path, 'documents.data')):
        documents, srls, chosen_ids = readOntoNotes_with_srl_anotation(dataset_path)
    else:
        with open(os.path.join(archive_path, 'documents.data'), 'rb') as filehandle:
            documents = pickle.load(filehandle)
    documents_with_srl_anotation, srls, chosen_ids = readOntoNotes_with_srl_anotation(documents)
    with open(os.path.join(archive_path, 'documents_with_srl_anotation.data'), 'wb') as filehandle:
        pickle.dump(documents_with_srl_anotation, filehandle)
    with open(os.path.join(archive_path, 'srls.data'), 'wb') as filehandle:
        pickle.dump(srls, filehandle)


def corpus_analysis(documents, archive_path, ranking="grl", candidate="coref_spans",
                    gram_roles=None, srls=None, search_space_size=100, t2_repeat=5):
    '''
    :param documents: a 2-d list of OntoNoteSentence
    :param archive_path: somewhere to save all the results
    :param candidate: should be ``coref_spans`` for Ontonotes corpus
    :param ranking: ``grl`` or ``srl``
    :param gram_roles: a 2-d list of DefaultDict[str, List[Tuple[int, int]]],
                        e.g. {'subj': [], 'dobj':[],'iobj': [], 'pobj':[] }
    :param srls: a 2-d list of DefaultDict[str, List[Tuple[int, int]]],
                    e.g. {'ARG0': [], 'ARG1':[] }
    :param search_space_size: int
    :return:
    '''
    assert not (gram_roles is None and srls is None), "Should provide either valid gram_roles or srls"
    if srls is None:
        srls = gram_roles
    else:
        gram_roles = srls
    # 3. Get converted_documents: Convert OntoSentence to ConvertedSent
    converted_documents = []
    for i, document in enumerate(documents):
        converted_sent = OntoSentence2ConvertedSent(document[0], sentence_id=0, gram_role=gram_roles[i][0], srl=srls[i][0])
        converted_document = [converted_sent]
        for j in range(1,len(document)):
            if ranking == "grl":
                converted_sent = OntoSentence2ConvertedSent(document[j], sentence_id=j, gram_role=gram_roles[i][j],
                                                                      srl=None)
            else:
                converted_sent = OntoSentence2ConvertedSent(document[j], sentence_id=j, gram_role=None,
                                                                      srl=srls[i][j])
            converted_document.append(converted_sent)
        converted_documents.append(converted_document)

    # 4. Get Table 1 and Table 2
    name = "best"
    get_Table_1(converted_documents, name, archive_path, result_dir,
                candidate=candidate, ranking=ranking)
    for id in range(t2_repeat):
        get_Table_2(converted_documents, name, archive_path, result_dir,
                    candidate=candidate, ranking=ranking, search_space_size=search_space_size, id=id)

result_dir = None

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", help="data-dir", type=str, default="/cluster/work/cotterell/ct/data/ontonotes/test.english.v4_gold_conll"
    )
    parser.add_argument(
        "--archive-dir", help="archive-dir", type=str, default="/cluster/work/cotterell/ct/data/centering_exp/gold-mention-2021.8.2.100"
    )
    parser.add_argument("-r",
        "--result-dir", help="the name that the result dir in each experiment dir should be"
                             " e.g. result_220627"
                             " Default: result_default", type=str, default="result_default"
    )
    args = parser.parse_args()
    archive_path = args.archive_dir
    result_dir = args.archive_dir
    if not os.path.isdir(archive_path):
        os.mkdir(archive_path)
    documents, gram_roles = prepare_data_full(args.data_dir, archive_path)
    corpus_analysis(documents, archive_path, ranking="grl", candidate="coref_spans",
                    gram_roles=gram_roles, srls=None, search_space_size=100, t2_repeat=5)