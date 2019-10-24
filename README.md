# awesome-document-similarity (WIP)
A curated list of resources on document similarity measures (papers, tutorials, code, ...)

## Document Representations

In order to compute the similarity of two documents, a numeric representation of the documents must be derived.

These representations are usually vectors of real numbers, whereby the vectors can be either sparse or dense.
The terms "sparse" and "dense" refer to the number of zero vs. non-zero elements in the vectors. A sparse vector is one that contains mostly zeros and few non-zero entries. A dense vector contains mostly non-zeros. 

[![](https://research.swtch.com/sparse0b.png)](https://research.swtch.com/sparse)

In addition, we distinguish document representations based on the type of data that they rely on, e.g., text, topics, citations.

### Text

- VSM
- TF-IDF (IDF variants, BM25,)


#### BERT and other Transformer models

- Long-form document classification with BERT. [Blogpost](https://andriymulyar.com/blog/bert-document-classification), [Code](https://github.com/AndriyMulyar/bert_document_classification)
- See ICLR 2020 reviews: 
  - [BERT-AL: BERT for Arbitrarily Long Document Understanding](https://openreview.net/forum?id=SklnVAEFDB)
  - [Blockwise Self-Attention for Long Document Understanding](https://openreview.net/forum?id=H1gpET4YDB)
- [Easy-to-use interface to fine-tuned BERT models for computing semantic similarity](https://github.com/AndriyMulyar/semantic-text-similarity)

### Topics

- LDA
- LSI

### Citations

- Bibliographic coupling

- Co-Citation

- Co-Citation Proximity Analysis (+IDF)


## Similarity Measures

- Jaccard Similarity

- Jensen-Shannon distance

- Word Mover Distance

- Cosine similarity

- Manhatten distance

### Siamese Networks


