# ct-coref
Code for Investigating Coreference Resolution Systems by Centering Theory

## Requirements
- Download Allennlp from https://github.com/allenai/allennlp.
- Download Allennlp models from https://github.com/allenai/allennlp-models.
- Replace the ``coref`` folder in ``allenlp-models`` with the ``coref`` folder provided here.
- Install Allennlp from source following the instructions.

## Corpus-based Analysis of CT
- The analysis logistics is provided in ``Corpuse_Analysis_on_OntoNotes.ipynb``. This document contains some useful statistics about the dataset we conducted experiments on, check out Section 2.
- To reproduce the results, run ``corpusAnalysis.py``.

## Analysis on Coreference Systems
-  ``Coreference_System_Analysis.ipynb``

## The Correlation between Coref F1 and CT-based Scores
-  ``F1_CT.ipynb``
