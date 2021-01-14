import os, pickle
from corpus_analysis import *


# Here specify the dataset location and output dictionary
dataset_path = "/cluster/work/cotterell/ct/data/ontonotes/test.english.v4_gold_conll"
archive_path = "/cluster/work/cotterell/ct/data/centering_exp/gold"


# 1. It you do not the required ``documents.data`` and ``documents_with_srl_anotation.data``,
# along with the ``gram_roles.data`` and ``srls.data`` yet,  please uncomment this two line

# prepare_data_full(dataset_path,archive_path)
# prepare_data_with_srl_anotation(dataset_path,archive_path)


# 2. Corpus analysis with gram-role-based CF ranking
with open(os.path.join(archive_path, 'documents.data'), 'rb') as filehandle:
    documents = pickle.load(filehandle)
with open(os.path.join(archive_path, 'gram_roles.data'), 'rb') as filehandle:
    gram_roles = pickle.load(filehandle)
print("readOntoNotes down!")

corpus_analysis(documents, archive_path, ranking="grl", candidate="coref_spans",
                    gram_roles=gram_roles, srls=None, search_space_size=100)


# 3. Corpus analysis with semantic-role-based CF ranking
with open(os.path.join(archive_path, 'documents_with_srl_anotation.data'), 'rb') as filehandle:
    documents_with_srl_anotation = pickle.load(filehandle)
with open(os.path.join(archive_path, 'srls.data'), 'rb') as filehandle:
    srls = pickle.load(filehandle)
corpus_analysis(documents, archive_path, ranking="srl", candidate="coref_spans",
                    gram_roles=None, srls=srls, search_space_size=100)

