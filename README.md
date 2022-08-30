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

### ConvertedSent
A class representing the annotations available for a single `CONLL` formatted sentence
 or a sentenece with coref predictions.

#### Parameters 
- document_id: `int`.
- line_id: `int`. The true sentence id within the document.
- words: `List[str]`. A list of tokens corresponding to this sentence.
            The Onotonotes tokenization, need to be mapped.
- clusters: `Dict[int, List[Tuple[int, int]]]`.
- pos_tags: `List[str]`. The pos annotation of each word.
- srl_frames: `List[Tuple[str, List[str]]]`.
        A dictionary keyed by the verb in the sentence for the given
        Propbank frame labels, in a BIO format.
- named_entities: `List[str]`. The BIO tags for named entities in the sentence.
- gram_roles: `Dict[str, List[Tuple[int, int]]]`. The keys are 'subj', 'obj'.
                The values are lists of spans.

### ConvertedDoc
A class representing the annotations for a `CONLL` formatted document.
#### Parameters 
- document_id: `int`.
- sentences: `List[ConvertedSent]`.
- entity_ids: `List[int]`. A list of entity ids that appear in this documents 
        according to the `clusters` in all the `convertedSent`s. 


###  CenteringUtterance
A class representing the centering properties for `ConvertedSent`.

#### Parameters
Ontonotes Annotations:
- document_id: `int`.
- line_id: `int`. The true sentence id within the document.
- words: `List[str]`. A list of tokens corresponding to this sentence.
            The Onotonotes tokenization, need to be mapped.
- clusters: `Dict[int, List[Tuple[int, int]]]`.
- pos_tags: `List[str]`. The pos annotation of each word.
- named_entities: `List[str]`. The BIO tags for named entities in the sentence.
- gram_roles: `Dict[str, List[Tuple[int, int]]]`. The keys are 'subj', 'obj'.
                The values are lists of spans.
- semantic_roles:  the spans of different semantic roles in this uttererance,
            a dict  where the keys are 'ARG0', 'ARG1'.
            The values are lists of spans.

Utterance-level properties:
- ranking: `str`. either `grl` or `srl`.
- CF_list: `List[int]`.
- CF_weights: `Dict[int, float]`.
        The keys are entity id's and the values are their corresponding weights.
- CP: `int`. The highest ranked element in the CF_list.

Discourse-level properties:
- CB_list: `List[int]`. A list of `entity_id`s which are the CB candidates in this utterance.
- CB_weights: `Dict[int, float]`. The keys are `entity_id`s and the values are their weights.
- CB: `int`. The highest ranked entity in the `CB_list`.
- first_CP: `int`. The first mentioned entity in the utterance.
- transition: `Transition`
- cheapness: `bool`. Cb(Un) = Cp(Un-1)
- coherence: `bool`. Cb(Un) = Cb(Un-1)
- salience: `bool`. Cb(Un) = Cp(Un)
- nocb: `bool`. The `CB_list` is empty.

*Note that the init function automatically setup all the utterance-level properties,
e.g. create the `CF_list` with the correct ranking. However, the discourse-level properties need to be set manually.*


###  CenteringDiscourse
A class representing a discourse with centering properties.

#### Parameters
- document_id: `int`.
- utterances: `List[CenteringUtterance]`.
- ranking: `str`. either `grl` or `srl`.
- first_CP: `int`. The first mentioned entity in the entire discourse.
- len: `int`. The number of utterances in this discourse.
- salience: the ratio of salient transitions to all transitions (`len-1`).
- coherence: the ratio of coherent transitions to all transitions (`len-1`).
- cheapness: the ratio of cheap transitions to all transitions (`len-1`).
- nocb: the ratio of transitions with nocb to all transitions (`len-1`).

### Usage
#### Step 1: 
Create a list of `convertedSent` by 
```python
converted_sentence = ConvertedSent(document_id=document_id,  # int
                              line_id=line_id,  # int
                              words=words,  # List[str]
                              clusters=clusters,
                              pos_tags=pos_tags,
                              gram_roles=gram_roles)
```
#### Step 2: 
Add CT properties to the `converted_document` (the list of `convertedSent`) by constructing a `CenteringDiscourse` object: 
```python
centeringDiscourse = CenteringDiscourse(converted_document, ranking="grl")
```

#### Step 3 (optional): 
Calculte the CT scores:
```python
final_CT_scores, unnormalized_CT_scores = centering.calculate_permutation_scores(centeringDiscourse)
```
- `unnormalized_CT_scores`: `Dict[str, float]`. A dict of unnormalized CT scores,
  where the scores are the ratio of the number of uttterances where a certain CT predicate being true to the total numbers of uttterances.
- `final_CT_scores`: `Dict[str, float]`. A dict of final CT scores. For example, `{"nocb": 0, "salience": 0, "coherence": 0, "cheapness": 0, "transition": 0, "kp": 0}`


## Running the experiments
### Step 1: Train coreference models
```ON
python get_coref_F1.py
```
### Step 2: Get CT scores
```ON
 python -m ct.ct_ontonotes \
--experiment-ids gold, coref-spanbert-base-2021.1.5 \
--epoch best \
--save-path path/to/coreference/models
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
