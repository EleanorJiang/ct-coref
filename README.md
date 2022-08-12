# Investigating the Role of Centering Theory in Neural Coreference Resolution
<p float="middle">
  <img src="/image/CT_vs_Coref.png" width="400" />
  <img src="/image/ct_example.png" width="400" />
</p>


This is an implementation of the paper 
[Investigating the Role of Centering Theory in Neural Coreference Resolution](https://www.overleaf.com/project/5fd265b15622577f9fc605b8).

Centering theory (CT; [Grosz et al, 1995](https://aclanthology.org/J95-2003.pdf)) 
provides a linguistic analysis of the structure of discourse. 
According to the theory, local coherence of discourse arises from 
the manner and extent to which successive utterances make reference to the same entities. 
In this paper, we investigate the connection between centering theory and modern coreference 
resolution systems. 
We provide an operationalization of centering and systematically investigate 
if the neural coreference resolvers adhere to the rules of centering theory by defining various 
discourse metrics and developing a search-based methodology. 
Our information-theoretic analysis 
reveals a positive dependence between coreference and centering; but also shows that high-quality 
neural coreference resolvers may not benefit much from explicitly modeling centering ideas. 
Our analysis further shows that contextualized embeddings contain much of the coherence information, 
which helps explain why CT can only provide little gains to modern neural coreference resolvers 
which make use of pretrained representations.
Finally, we discuss factors that contribute to coreference which are not modeled by CT such as 
commonsense and recency bias. 



## Install Dependencies
```
conda create --name ct python=3.6 numpy pandas
conda activate ct
pip install -U pip setuptools wheel
pip install -U spacy
python -m spacy download en_core_web_sm
pip install pytorch-transformers
conda install -c conda-forge python=3.6 allennlp
conda install -c conda-forge allennlp-models
```
or `pip install allennlp-models`

## Operationalization of Centering Theory
The operationalization of centering theory is implemented in  `ct/centering.py`:

### CenteringUtterance
`class ConvertedSent`
```python
class ConvertedSent:
    def __init__(self, sentence_id, document_id, words, coref_spans=None, pos_tags=None, gram_role=None, srl=None,
                 top_spans=[], clusters=[], offset=None) -> None:
        self.document_id = document_id
        self.sentence_id = sentence_id
        self.words = words
        self.pos_tags = pos_tags
        self.gram_role = gram_role
        self.srl = srl
        self.top_spans = top_spans
        self.clusters = clusters
        self.offset = offset
        self.coref_spans = coref_spans
```

`class CenteringUtterance`:
```python
CenteringUtterance(sentence: ConvertedSent, 
                    candidate="clusters", 
                    ranking="grl",
                    CB=None, 
                    transition=Transition.NA, 
                    cheapness=None, 
                    coherence=None, 
                    salience=None)
```

## Running the experiments
### Step 1: Train coreference models
```ON
python get_coref_F1.py
```
### Step 2: Get CT scores
```ON
python c2f_analysis.py  \
--data_dir path/to/coreference/models
-e coref-spanbert-base-2021.1.5 \
-dp 100 \
--epoch best \
-r path/to/save/results
```

## Contact

[comment]: <> (### Citation)

[comment]: <> (**If this code or the paper were usefull to you, consider citing it:**)

[comment]: <> (```bibtex)

[comment]: <> (@article{jiang-etal-2022-investigating,)

[comment]: <> (      title="Investigating the Role of Centering Theory in Neural Coreference Resolution", )

[comment]: <> (      author="Yuchen Eleanor Jiang and Tianyu Liu and Ryan Cotterell and Mrinmaya Sachan",)

[comment]: <> (      booktitle = "Arxiv",)

[comment]: <> (      month = jul,)

[comment]: <> (      year = "2022",)

[comment]: <> (      address = "Seattle, United States",)

[comment]: <> (      publisher = "Association for Computational Linguistics",)

[comment]: <> (      url = "https://aclanthology.org/2022.naacl-main.111",)

[comment]: <> (      pages = "1550--1565",)

[comment]: <> (})

[comment]: <> (```)

To ask questions or report problems, please contact yucjiang@ethz.ch.
