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

Siamese networks [(Bromley, Jane, et al. "Signature verification using a siamese time delay neural network". Advances in neural information processing systems. 1994.)](http://papers.nips.cc/paper/769-signature-verification-using-a-siamese-time-delay-neural-network.pdf) are neural networks containing two or more identical subnetwork components.

> It is important that not only the architecture of the subnetworks is identical, but the weights have to be shared among them as well for the network to be called "siamese". The main idea behind siamese networks is that they can learn useful data descriptors that can be further used to compare between the inputs of the respective subnetworks. Hereby, inputs can be anything from numerical data (in this case the subnetworks are usually formed by fully-connected layers), image data (with CNNs as subnetworks) or even sequential data such as sentences or time signals (with RNNs as subnetworks).

#### Tasks

Binary classifcation

Multi-label classification

#### Loss functions

#### Concatenations

- Reimers, N. and Gurevych, I. 2019. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. The 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP 2019) (2019).

#### Applications

- Mueller, J. and Thyagarajan, A. 2014. Siamese Recurrent Architectures for Learning Sentence Similarity. Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence (AAAI-16). 2012 (2014), 1386–1393. DOI:https://doi.org/10.1109/CVPR.2014.180.

