import spacy, os, pickle, json
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


def ontoNotes2spacy(ontonotes_file, nlp, save_path) -> List[Dict[str, List[List[Any]]]]:
    '''
    :param ontonotes_file: where the "ontonotes/dev.english.v4_gold_conll" file is.
    :param predicted_dict_file: `best_predicted_dicts.data` for reference

    :return: predicted_dicts: List[List[Dict[str, List[List[Any]]]]]
            Example:
            text = "Marry believes in herself but Henry does n’t believe her.”
            predicted_dict: {
                         'document': ['Marry', 'believes',  'in', 'herself','but', ‘Henry’, 'does' , ”n’t”, 'believe', 'her', '.'],
                         'clusters': [[[0, 0], [3, 3], [9, 9]]],   # Note that singleton [9,9] is excluded.
                         'pos_tags':  ['SUBJ', 'VERB',  'in', 'herself','but', ‘Henry’, 'does' , ”n’t”, 'believe', 'her', 'PUNCT'],
                         'srl': ['Marry', 'believes',  'in', 'herself','but', ‘Henry’, 'does' , ”n’t”, 'believe', 'her', '.'],
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
                        raise AssertionError("Sanity check fails: Every ``entence.word'' should be covered && "
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