from pathlib import Path
from typing import Dict, List, Optional, Tuple, DefaultDict, NamedTuple, Union, Dict, Any, Optional
from allennlp.common import plugins
from allennlp.models.archival import Archive, load_archive
from allennlp.predictors.predictor import Predictor
from allennlp.data.dataset_readers.dataset_utils.span_utils import TypedSpan

"""
1.2 Get Prediction of c2f-coref Model

predictor is a CorefPredictor
"""


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
        dataset_reader_to_load: str = "train",
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