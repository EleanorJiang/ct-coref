import os, json, collections
from typing import Dict, List, Optional, Tuple, DefaultDict
import pandas as pd
from ontonotes.conll_util import find_span, string2list, string2grldict, string2clusters, string2ontonotesClusters
from ontonotes.align_subwords import align_bpe, find_bpe
from ontonotes.ontonotes import Ontonotes
from ct.centering import *
import argparse

Span = Tuple[int, int]
EXCLUDED_DOCS = [26, 71, 244, 107, 116, 123, 130, 131, 143, 146, 192, 218, 230, 237, 281]


class CTOntonotes:
    """
    This `DatasetReader` is designed to get CT scores for the English OntoNotes v5.0.
    """
    def __init__(self,
            save_path: str = None,
            ontonotes_csv: str = None,
            allennlp_csv: str = None,
            map_json: str = None,
            experiment_id: str = None,
            epoch: str = None,
    ) -> None:
        self.save_path = save_path
        self.ontonotes_csv = ontonotes_csv
        self.allennlp_csv = allennlp_csv
        self.map_json = map_json
        self.experiment_id = experiment_id
        self.epoch = epoch
        self.metrics = ["nocb", "cheapness", "coherence", "salience", "transition", "kp", "valid_len"]
        self.mean_df = pd.DataFrame(columns=["experiment_id", "epoch"] + self.metrics)


    def process_ontonotes(self, ontonotes_file):
        """
        This function processes the ontonotes file `test.english.v4_gold_conll` to `ontonotes_test.csv`, where the columns are:
        - document_id: `int`. 0 - 347
        - document_file_name: `str`. This is a variation on the document filename.
                                It corresponds to the original document_id in the OntonotesSentence class.
        - sentence_id: `int`. The sentence id within the document
        - clusters: `Dict[int, List[Tuple[int, int]]]`.
        - words: `List[str]`. A list of tokens corresponding to this sentence.
                            The Onotonotes tokenization, need to be mapped.
        - pos_tags: `List[str]`. The pos annotation of each word.
        - srl_frames: `List[Tuple[str, List[str]]]`.
                        A dictionary keyed by the verb in the sentence for the given
                        Propbank frame labels, in a BIO format.
        - named_entities: `List[str]`. The BIO tags for named entities in the sentence.
        """
        ontonotes_reader = Ontonotes()
        df = pd.DataFrame(columns=["document_id", "document_file_name", "sentence_id",
                                   "clusters", "words", "pos_tags", "srl_frames"])
        for i, sentences in enumerate(ontonotes_reader.dataset_document_iterator(ontonotes_file)):
            document_id = i
            for sentence in sentences:
                document_file_name = sentence.document_id
                df.loc[len(df.index)] = [document_id, document_file_name, sentence.sentence_id,
                                         sentence.clusters, sentence.words, sentence.pos_tags, sentence.srl_frames]

        df.to_csv(os.path.join(self.save_path, "ontonotes_test.csv"), index=False, sep='\t')


    def preocess_allennlp(self, predicted_dicts=None):
        """
        This function processes the `predicted_dicts.data` files to `ontonotes_allennlp_cluster.csv`,
        where the columns are:
        - document_id: `int`. 0 - 347
        - experiment_id: `str`. E.g. "coref-spanbert-base-2021.1.5.100".
        - embedding: `str`. Chosen from ["SpanBERT", "GloVe", "One-hot"]
        - epoch: `str`. Either epoch id, or "best".
        - %data: `int`
        - "cluster": `List[List[Span]]`. [[[42, 44], [68, 71]], ... ]. The token ids are document-level ids.
        The processed csv is very similar to the `all_data_points.csv`, except that it provides per-document results.
        """
        pass


    def add_grl_from_spacy(self, ontonotes_spacy_csv):
        """
        This function does two things:
        1) map the tokenization in `ontonotes_spacy.csv` to `ontonotes_test.csv`
        2) add the re-indexized `gram_roles` dict and add it into `ontonotes_test.csv`.
        The resulting csv is called `ontonotes_test_grl.csv`.
        """
        ontonotes_test = os.path.join(self.save_path, "ontonotes_test.csv")
        source_df = pd.read_csv(ontonotes_test, sep='\t')
        spacy_df = pd.read_csv(ontonotes_spacy_csv, sep='\t')
        source_df = source_df.reset_index()  # make sure indexes pair with number of rows
        spacy_df = spacy_df.reset_index()  # make sure indexes pair with number of rows
        source_df.insert(source_df.shape[1], 'gram_roles', "")

        for index, row in source_df.iterrows():
            gram_roles_str = spacy_df.loc[index, 'gram_roles']
            if gram_roles_str == "{}":
                # if the gram_roles dict is empty. we skip this sentence
                continue
            # gram_roles = json.loads(gram_roles_str[1:-1])
            gram_roles = string2grldict(gram_roles_str)
            words = string2list(row['words'])
            spacy_tokens = string2list(spacy_df.loc[index, 'tokens'])
            if len(words) != len(spacy_tokens):
                # the two tokenizations are the same, then we do align before adding gram_roles
                # to the original file
                start_alignment, end_alignment = align_bpe(words, spacy_tokens)
                for key, spanlist in gram_roles.items():
                    for i, span in enumerate(spanlist):
                        new_span = start_alignment[span[0]], end_alignment[span[1]]
                        gram_roles[key][i] = new_span
            source_df.loc[index, 'gram_roles'] = str(gram_roles)

        source_df.to_csv(os.path.join(self.save_path, "ontonotes_test_grl.csv"), index=False, sep='\t')


    @staticmethod
    def _add_line_id_to_ontonotes_test(in_file, out_file):
        df = pd.read_csv(in_file, sep='\t', index_col=0)
        df = df.reset_index()
        df.insert(1, 'line_id', -1)
        document_id, line_id = -1, -1
        for index, row in df.iterrows():
            if document_id == row["document_id"]:
                # The same document, we increase the line_id
                line_id += 1
            else:
                # The next document, we reset the line_id, and increase the document_id
                line_id = 0
                document_id += 1
            df.loc[index, 'line_id'] = line_id
        df.to_csv(out_file, index=False, sep='\t')


    def allennlp2ontonotes(self):
        """
        This script does two things:
        1) read `ontonotes_allennlp_tokenization.csv` and `ontonotes_test.csv`, map the indices.
        2）generate a json file `map_allennlp2ontonotes.json`
            map: Dict[int, List[Tuple[int, int, int]]
                348 key-value pairs，the key is “doc_id”,
                and the value is a list of 3-tuples Tuple[int, int, int] of length len(allennlp_tokens),
                where the 3-tuple is (line_id, start, end).
        3) For each span (start, end) in `ontonotes_allennlp_cluster_{*}.csv`,
            the new span in ontonotes_test.csv is `map[doc_id][span[0]][2], map[doc_id][span[1]][3]`
        """
        ontonotes_test = os.path.join(self.save_path, "ontonotes_test.csv")
        source_df = pd.read_csv(ontonotes_test, sep='\t')
        allenlp_df = pd.read_csv(os.path.join(self.save_path, "ontonotes_allennlp_tokenization.csv"), sep='\t')
        source_df = source_df.reset_index()  # make sure indexes pair with number of rows
        allenlp_df = allenlp_df.reset_index()  # make sure indexes pair with number of rows

        idx, document_id = 0, -1
        map: DefaultDict[int, List[Tuple[int, int, int]]] = collections.defaultdict(list)
        for row_id, row in allenlp_df.iterrows():
            tokens = string2list(row['tokens'])
            # start a new document!
            document_id += 1
            sentences = []
            while idx < source_df.shape[0] and source_df.loc[idx, "document_id"] == document_id:
                sentence = string2list(source_df.loc[idx, "words"])
                sentences.append(sentence)
                idx += 1
            # Now we reach the first line of the new document and the line_id is 0,
            # It's time to do the mapping
            remain_tokens = tokens
            for i, sentence in enumerate(sentences):
                # whitening sentence first
                sentence = [word.replace("/", "") for word in sentence]
                (index_start, index_end) = find_bpe(sentence, remain_tokens)
                assert index_start == 0
                tmp_tokens = remain_tokens[index_start: index_end]
                remain_tokens = remain_tokens[index_end:]
                start_alignment, end_alignment = align_bpe(sentence, tmp_tokens)
                tuple_list = [(i, start_id, end_id) for start_id, end_id in zip(start_alignment, end_alignment)]
                map[document_id].extend(tuple_list)
            assert len(map[document_id]) == len(tokens)

        with open(os.path.join(self.save_path, 'map_allennlp2ontonotes.json'), 'w') as json_file:
            json.dump(dict(map), json_file)


    def load_converted_documents_from_csv(self):
        """
        Load the ontonotes test set from `ontonotes_csv`.
        Update the `entity_id` and `clusters` if `experiment_id` and `allennlp_csv` is not None.
        :param converted_document: `ConvertedDoc`.
        :param ontonotes_csv:
        :param map_json:
        :param allennlp_csv:
        :return: converted_documents: `List[ConvertedDoc]`. A list of ConvertedDoc, each ConvertedDoc
                 corresponds to a document in the ontonotes test set (348 documents in total).
                 each ConvertedDoc consists of a list of convertedSent and a document_id.
        """
        ontonotes_df = pd.read_csv(self.ontonotes_csv, sep='\t', dtype=str).reset_index()
        if self.allennlp_csv is not None:
            allennlp_df = pd.read_csv(self.allennlp_csv, sep='\t', dtype=str).reset_index()
            with open(self.map_json, "r") as json_file:
                map = json.load(json_file)
        # construct a list of ConvertedDoc from the ontonotes_csv - `converted_documents`,
        # where each convertedDoc consists of a list of convertedSent
        converted_documents: List[ConvertedDoc] = []
        convertedDoc = ConvertedDoc(document_id=0)
        for index, row in ontonotes_df.iterrows():
            # the same document
            document_id = int(row["document_id"])
            line_id = int(row["line_id"])
            words = string2list(row["words"])
            pos_tags = string2list(row["pos_tags"])
            gram_roles = string2grldict(row["gram_roles"])
            if self.allennlp_csv is None:
                clusters = string2ontonotesClusters(row["clusters"])
            else:
                clusters = None
            convertedSent = ConvertedSent(document_id=document_id,
                                          line_id=line_id,
                                          words=words,
                                          clusters=clusters,
                                          pos_tags=pos_tags,
                                          gram_roles=gram_roles)
            convertedDoc.sentences.append(convertedSent)
            if index + 1 == ontonotes_df.shape[0] or row["document_id"] != ontonotes_df.loc[index + 1, "document_id"]:
                # The line is the last sentence in the current documents, and the next line is the start of a new document
                if self.allennlp_csv is None:
                    convertedDoc.get_entity_ids_from_ontonotes()
                else:
                    selected_cell = allennlp_df.loc[(allennlp_df['document_id'] == str(document_id)) &
                           (allennlp_df['experiment_id'] == self.experiment_id) & (allennlp_df['epoch'] == self.epoch),
                                                    "cluster"].iloc[0]
                    allennlp_clusters = string2clusters(selected_cell)
                    convertedDoc.update_clusters(allennlp_clusters, map)
                converted_documents.append(convertedDoc)
                document_id += 1
                convertedDoc = ConvertedDoc(document_id=document_id)
        return converted_documents


    def get_ct_scores_for_ontonotes(self, result_csv, filter_len=0):
        # Load ground truth annotations
        converted_documents = self.load_converted_documents_from_csv()
        # Load anllenlp clusters
        # converted_documents = load_converted_documents_from_csv(ontonotes_csv, experiment_id=experiment_id, epoch=epoch, allennlp_csv=allennlp_csv, map_json=map_json)
        # CALCULATE CENTERING SCORES FOR EACH DOCUMENTS
        df = pd.DataFrame(columns=["document_id", "experiment_id", "epoch"] + self.metrics)
        for converted_document in converted_documents:
            centeringDiscourse = CenteringDiscourse(converted_document, ranking="grl")
            if centeringDiscourse.valid_len <= filter_len:
                continue
            if centeringDiscourse.document_id in EXCLUDED_DOCS:
                continue
            final_CT_scores, unnormalized_CT_scores = calculate_permutation_scores(centeringDiscourse)
            final_CT_scores["document_id"] = converted_document.document_id
            final_CT_scores["experiment_id"] = self.experiment_id
            final_CT_scores["epoch"] = self.epoch
            final_CT_scores["valid_len"] = centeringDiscourse.valid_len
            df = df.append(final_CT_scores, ignore_index=True)
        mean_line = {"experiment_id": self.experiment_id, "epoch": self.epoch}
        for metric in self.metrics:
            mean_line[metric] = float(df[metric].mean())
        self.mean_df = self.mean_df.append(mean_line, ignore_index=True)
        df = df.append(mean_line, ignore_index=True)
        df.to_csv(result_csv, index=False, sep='\t', float_format='%.2f')



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-path", help="prefix", type=str, default="/Users/eleanorjiang/iCloud/RESEARCH/CT/RESULTS"
    )
    parser.add_argument(
        "--result-file", help="the name of the result file", type=str, default=""
    )
    parser.add_argument("-o",
        "--output", help="the name of the result file of mean CT scores", type=str, default="mean_ct_scores.csv"
    )
    parser.add_argument('-j', '--load-json', action='store_true', help='load experiment configs from json')
    parser.add_argument("-json",
        "--experiment-json", help="the json file where the experiment-ids and the epoches are stored", type=str,
                        default="/Users/eleanorjiang/iCloud/RESEARCH/CT/RESULTS/experiment_id.epoch.json"
    )
    parser.add_argument("-e",
        "--experiment-ids", nargs="+", default=["naive-2021.6.6.10"],
                        type=str,
                        help="the experiment ids, being EMBEDDING-DATA, "
                             "e.g.: gold, naive-2021.6.6.100 no-embedding-2021.5.6.100 coref-spanbert-base-2021.1.10.100"
    )
    parser.add_argument(
        "--epoch", nargs="+", default=["best", "1"], type=str,
                        help="the epoch, being either a number in [0, 40] or best"
    )
    parser.add_argument("-p",
        "--paras", nargs="+", default=None, type=float,
                        help="gamma, big+gamma, gate_escape_factor"
    )
    args = parser.parse_args()
    ctontonotes = CTOntonotes(save_path=args.save_path)
    ctontonotes.ontonotes_csv = os.path.join(ctontonotes.save_path, "ontonotes_test.csv")
    ctontonotes.map_json = os.path.join(ctontonotes.save_path, "map_allennlp2ontonotes.json")

    if args.load_json:
        with open(args.experiment_json) as f:
            experiment_dict = json.load(f)
    else:
        experiment_dict = {}
        for experiment_id in args.experiment_ids:
            experiment_dict[experiment_id] = args.epoch

    for experiment_id, epoches in experiment_dict.items():
        for epoch in epoches:
            if args.result_file == "":
                result_csv = os.path.join(args.save_path, "document_level_ct_scores", f"ct_scores_{experiment_id}.{epoch}.csv")
            else:
                result_csv = os.path.join(args.save_path, f"ct_scores_{args.result_file}.csv")

            if experiment_id == "gold":
                ctontonotes.allennlp_csv = None
            else:
                ctontonotes.allennlp_csv = os.path.join(ctontonotes.save_path, "allennlp_predictions",
                                                        f"{experiment_id}.{epoch}.csv")
            ctontonotes.experiment_id = experiment_id
            ctontonotes.epoch = epoch
            ctontonotes.get_ct_scores_for_ontonotes(result_csv=result_csv)
    mean_csv = os.path.join(ctontonotes.save_path, args.output)
    ctontonotes.mean_df.to_csv(mean_csv, index=False, sep='\t', float_format='%.2f')

