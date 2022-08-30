import spacy, os, pickle, json, math
from .ontonotes import Ontonotes, OntonotesSentence, TypedSpan
import collections
from typing import Dict, List, Optional, Tuple, DefaultDict, NamedTuple, Union, Dict, Any, Optional


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


def string2list(string):
    """
    convert a printed list (str) to a List of token
    :param string: '[\'--\', \'basically\', \',\', \'it\', \'was\', \'unanimously\', \'agreed\', \'upon\', \'by\', \'the\', \'various\', \'relevant\', \'parties\', \'.\']'
    :return: tokens: `List[str]`.
    """
    org_string = string
    string = string.replace("', '", "[SEP]").replace('\", \"', "[SEP]")
    string = string.replace('\", \'', "[SEP]").replace('\', \"', "[SEP]")
    string = string.replace(" ", "").replace("\\", "").replace("/", "")
    string = string.replace(',', "[COMMA]")
    string = string[2:-2].replace("'", "[PRIME]")
    tokens = string.split('[SEP]')
    tokens = [token.replace("[PRIME]", "'") for token in tokens]
    tokens = [token.replace("[COMMA]", ",") for token in tokens]
    return tokens


def string2grldict(string):
    if type(string) != str:
        return {}
    gram_roles = {}
    if string.find('obj') > string.find('subj'):
        string = string.replace(", 'obj': ", "<SEP>obj:")
    else:
        string = string.replace(", 'subj': ", "<SEP>subj:")
    string = string.replace("'", "").replace(" ", "")
    lists = string[1:-1].split('<SEP>')
    for list in lists:
        kv = list.split(':')
        (key, value) = kv
        value_list = value[2:-2].split("),(")
        span_list = [(int(value.split(",")[0]), int(value.split(",")[1])) for value in value_list]
        gram_roles[key] = span_list
    return gram_roles


def string2clusters(string):
    """
    :param string: "[[[3, 3], [16, 16], [19, 23]], [[25, 27], [42, 44], [57, 59]], [[65, 66], [82, 82]]]"
    :return: cluster: `List[List[Span]]`.
            e.g.  [[(3, 3), (16, 16), (19, 23)], ... ]
    """
    if string == "[]":
        return []
    clusters = []
    list_strings = string.replace(" ", "")[3:-3].split("]],[[")
    for list_str in list_strings:  # list_str = '27,27],[59,60],[72,73'
        lst = list_str.split("],[")  # lst = ['27,27','59,60','72,73']
        cluster = [(int(pair.split(",")[0]), int(pair.split(",")[1])) for pair in lst]
        clusters.append(cluster)
    return clusters

def string2ontonotesClusters(string):
    """
    :param string: "{42: [(2, 2), (5, 9)], 32: [(11, 13)]}"
    :return: cluster: `Dict[int, List[Tuple[int, int]]]`.  {42: [(2, 2), (5, 9)], 32: [(11, 13)]}
    """
    if string == "{}":
        return {}
    clusters: Dict[int, List[Tuple[int, int]]] = {}
    kv_strings = string[1:-3].split(")], ")
    for kv_string in kv_strings:
        kv = kv_string.split(": [(")
        entity_id = int(kv[0])
        span_strings = kv[1].split("), (")   # ["2, 2", "5, 9"]
        span_List = [(int(span_str.split(", ")[0]), int(span_str.split(", ")[1])) for span_str in span_strings]
        clusters[entity_id] = span_List
    return clusters


def ontoNotes2spacy(ontonotes_file, nlp, save_path) -> List[Dict[str, List[List[Any]]]]:
    '''
    :param ontonotes_file: where the "ontonotes/dev.english.v4_gold_conll" file is.
    :param predicted_dict_file: `best_predicted_dicts.data` for reference

    :return: predicted_dicts: List[List[Dict[str, List[List[Any]]]]]
                             a list of predicted dict, each predicted dict corresponds to one sentence in the document
            Example:
            text = "Marry believes in herself but Henry does n’t believe her.”
                     Note that we use " " to connect multiple sentences.
            predicted_dict: {
                         'document': ['Marry', 'believes',  'in', 'herself','but', ‘Henry’, 'does' , ”n’t”, 'believe', 'her', '.'],
                         'clusters': [[[0, 0], [3, 3], [9, 9]]],   # Note that singleton [9,9] is excluded.
                         'pos_tags':  ['SUBJ', 'VERB',  'in', 'herself','but', ‘Henry’, 'does' , ”n’t”, 'believe', 'her', 'PUNCT'],
                         }
    '''
    ontonotes_reader = Ontonotes()
    # with open(predicted_dict_file, 'rb') as filehandle:
    #     orginal_predicted_dicts = pickle.load(filehandle)
    nested_predicted_dicts = []
    for i, sentences in enumerate(ontonotes_reader.dataset_document_iterator(ontonotes_file)):
        predicted_dict_doc = []
        for sentence in sentences:
            predicted_dict = collections.defaultdict(list)
            doc = nlp(' '.join(sentence.words))
            tokens = [t.text for t in doc]
            """
            doc_ids: the document-level token_id
                List[List[int]]  
                e.g. [0,1,2,3,...],[14,15,16,...],[], ...
            """
            # If the tokenization got from spacy is not the same with the original annotation, we do a mapping here
            if len(tokens) != len(sentence.words):
                #print("id:", i)
                #print("ontonotes: ", sentence.words)
                #print("spacy:", tokens)
                """
                token_ids: the token id in sentence.words of the spacy token in `tokens`
                        List[List[int]]  
                        e.g. [0,1,2,3,...],[0,0,1,...],[], ...
                        We assume `tokens` is longer than `sentence.words` as we have spaces between all words, 
                            (By doc = nlp(' '.join(sentence.words)) )
                """
                assert len(tokens) > len(sentence.words)
                word_cover = [[-1, -1] for word in sentence.words]
                cursor = 0
                word_cover[0][0], current_word = 0, sentence.words[0].strip(" ")
                for k, token in enumerate(tokens):
                    token = token.strip(" ")
                    if not current_word.startswith(token):  # token not in current_word anymore
                        word_cover[cursor][1] = k-1
                        word_cover[cursor+1][0] = k
                        cursor += 1
                        current_word = sentence.words[cursor].strip(" ")
                    assert current_word.startswith(token)
                    current_word = current_word[len(token):].strip(" ")
                word_cover[cursor][1] = k - 1
                # Sanity Check: Every ``sentence.word'' should be covered && Every ``token_ids'' should be assigned.
                assert cursor == len(sentence.words) - 1
                assert k == len(tokens)
                for span in word_cover:
                    if span[0] == -1 or span[1] == -1:
                        print(word_cover)
                        raise AssertionError("Sanity check fails: Every ``sentence.word'' should be covered && "
                                             "Every ``token_ids'' should be assigned.")
            predicted_dict['document'] = [token for token in tokens]
            # now we add predicted_dict['clusters']: List[List[Span]]
            # first we collect  clusters: Dict[str, List[Span]]
            clusters = collections.defaultdict(list)
            for typed_span in sentence.coref_spans:
                # TypedSpan = Tuple[int, Tuple[int, int]]
                # Coref annotations are on a _per sentence_
                # basis, so we need to adjust them to be relative
                # to the length of the document.
                entity_id, (start, end) = typed_span
                if len(tokens) != len(sentence.words):
                    clusters[entity_id].append((word_cover[start][0], word_cover[end][1]))
                else:
                    clusters[entity_id].append((start, end))
            predicted_dict['clusters'] = [list(span_list) for span_list in clusters.values()]
        predicted_dict_doc.append(predicted_dict)
    nested_predicted_dicts.append(predicted_dict_doc)
    with open(os.path.join(save_path, "ontonotes_nested_predicted_dict.json"), 'wb') as filehandle:
        json.dump(nested_predicted_dicts, filehandle)

    return nested_predicted_dicts





if __name__=="__main__":
    nlp = spacy.load('en_core_web_sm')
    ontonotes_file = '/cluster/work/cotterell/ct/data/ontonotes/dev.english.v4_gold_conll'
    predicted_dict_file = '/cluster/work/cotterell/ct/data/centering_exp/coref-spanbert-base-2021.5.17/best_predicted_dicts.data'
    ontoNotes2spacy(ontonotes_file, nlp, save_path="/cluster/work/cotterell/ct/data/result_220627")