# Awesome Document Similarity Measures

A curated list of resources, such as papers, tutorials, code, etc., on the topic of document similarity measures.

## Motivation

The goal of this repository is to provide a comprehensive overview for students and reseachers.  
Document similarity measures are basis the several downstream applications in the area of natural language processing (NLP) and information retrieval (IR). 
Among the most common applications are clustering, duplicate or plagirism detection and content-based recommender systems.
We selected the following content while having primarily the recommender systems application in mind.
In particular, we focus on literature recommender systems that need to assess the similarity of long-form and rich content documents. "Long-form" refers to the amount of document content from +100 sentences, whereas rich content means that documents contain aside from text also images, mathematical equations and citations/links.  

### Dimensions of Similarity

Documents might be declared as similar, when they, e.g., cover the same topic, use a common set of words or are written in the same font. 
In IR the dimension of similarity defines the understanding of similarity.
We distinguish between the following dimensions: lexical, structural and semantic document similarity.

Moreover, similarity is not a binary decision. 
In many cases declaring two things as similar or not, is not suitable. Instead, the degree of similarity is measured. Similarity measures express therefore document similarity as normalised scalar score, which is within an interval of zero to one.
The highest degree of similarity is measured as one. 
When two objects are dissimilar, the degree of similarity is zero.

- D. Lin, “An Information-Theoretic Definition of Similarity,” Proc. ICML, pp. 296–304, 1998.


### Lexical Similarity

The lexical document similarity of two documents depends on the words, which occur in the document text. 
A total overlap between vocabularies would result in a lexical similarity of 1, whereas 0 means both documents share no words. 
This dimension of similarity can be calculated by a simple word-to-word comparison. 
Methods like stemming or stop word removal increase the effectiveness of lexical similarity.

- J. B. Lovins, “Development of a stemming algorithm,” Mech. Transl. Comput. Linguist., vol. 11, no. June, pp. 22–31, 1968.
- C. Fox, “A stop list for general text,” ACM SIGIR Forum, vol. 24, no. 1–2. pp. 19–21, 1989.

### Strutural Similarity

Whereas lexical similarity focuses on the vocabulary, structural similarity describes the conceptual composition of two documents. 
Structural document similarity stretches from graphical components, like text layout, over similarities in the composition of text segments, e.g., paragraphs or sentences that are lexical similar, to the arrangement of citations and hyperlinks.

Structural similarity is mainly used for semi-structured document formats, such as XML or HTML. 
A common, yet expensive in computing time, approach is to calculate the minimum cost edit distance between any two document structures. 
The cost edit distance is the number of actions that are required to change a document so it is identical to another document

- C. Shin and D. Doermann, “Document image retrieval based on layout structural similarity,” Int. Conf. Image Process. Comput. Vision, Pattern Recognit., pp. 606–612, 2006.
- D. Buttler, “A short survey of document structure similarity algorithms,” Int. Conf. Internet Comput., pp. 3–9, 2004.

### Semantic Similarity

Two documents are considered as semantically similar when they cover related topics or have the same semantic meaning. 
A proper determination of the semantic similarity is essential for many IR systems, since in many use cases the users information need is rather about the semantic meaning than the vocabulary or structure of a document. 
However, measuring topical relatedness is a complex task. 
Therefore, lexical or structural similarity is often used to approximate semantic similarity.
The example below shows that approximating semantic by lexical similarity is not always suitable.

1. Our earth is round.
2. The world is a globe.

Even if both sentences are lexically different, because they have only one out of five words in common, the semantic meaning of the sentences is synonymous.

## Document Representations

In order to compute the similarity of two documents, a numeric representation of the documents must be derived.

These representations are usually vectors of real numbers, whereby the vectors can be either sparse or dense.
The terms "sparse" and "dense" refer to the number of zero vs. non-zero elements in the vectors. A sparse vector is one that contains mostly zeros and few non-zero entries. A dense vector contains mostly non-zeros. 

[![](https://research.swtch.com/sparse0b.png)](https://research.swtch.com/sparse)

In addition, we distinguish document representations based on the type of data that they rely on, e.g., text, topics, citations.

In the context of machine learning approaches that produce dense vector representations, the process is often refered to as learning of document features.

### Traditional Text-based

- Bag-of-Words
- VSM
- TF-IDF (IDF variants, BM25,)

### Word-level

- Word2Vec. [Paper](https://arxiv.org/pdf/1301.3781.pdf%5D)
- Glove. [Paper](https://www.aclweb.org/anthology/D14-1162). [Code](https://nlpython.com/implementing-glove-model-with-pytorch/)
- FastText [Paper](https://arxiv.org/pdf/1607.01759.pdf). [Code](https://fasttext.cc/)

### Word Context

- Contextualized Word Vectors (CoVe). [Paper](http://papers.nips.cc/paper/7209-learned-in-translation-contextualized-word-vectors.pdf), [Code](https://github.com/salesforce/cove)
- Embeddings from Language Models (ELMo). [Paper](https://arxiv.org/pdf/1802.05365.pdf)
- Contextual String Embeddings (Zalando Flair). [Paper](http://aclweb.org/anthology/C18-1139)

### From word to sentence level

- Average
- Weighted Average
- Smooth Inverse Frequency. [A Simple but Tough-to-Beat Baseline for Sentence Embeddings](https://openreview.net/pdf?id=SyK00v5xx)

### Sentence-level

- Skip-thoughts. [Paper](https://arxiv.org/pdf/1506.06726.pdf). [Code](https://github.com/ryankiros/skip-thoughts)
- Quick-Thoughts. [Paper](https://arxiv.org/pdf/1803.02893.pdf)
- Universal Sentence Encoder. [Paper](https://arxiv.org/abs/1803.11175)

> We find that using a similarity based on angular distance performs better on average than raw cosine similarity.

- InferSent. [Paper](https://arxiv.org/abs/1705.02364) [Code](https://github.com/facebookresearch/InferSent)

InferSent is a sentence embeddings method that provides semantic representations for English sentences. It is trained on natural language inference data and generalizes well to many different tasks.


### BERT and other Transformer Language Models

- BERT [Paper](https://arxiv.org/abs/1810.04805)
- GPT [Paper](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
- Generative Pre-Training-2 (GPT-2) [Paper](https://www.techbooky.com/wp-content/uploads/2019/02/Better-Language-Models-and-Their-Implications.pdf)
- Universal Language Model Fine-tuning (ULMFiT) [Paper](https://arxiv.org/pdf/1801.06146.pdf)
- XLNet
- Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context. [Paper](https://arxiv.org/abs/1901.02860) [Code](https://github.com/kimiyoung/transformer-xl)


BERT pooling strategies:

> Bert is amazing at encoding texts, just pool the contextualized embeddings. If not pre-training, use the 2nd to last layer as it usually gets better results (see the research in the bert-as-a-service github repo). [Reddit](https://www.reddit.com/r/MachineLearning/comments/dkjlq2/d_im_looking_for_successfailure_stories_applying/)

- https://github.com/hanxiao/bert-as-service#q-what-are-the-available-pooling-strategies

Overcoming BERT's 512 token limit:

> The Transformer self-attention mechanism imposes a quadratic cost with respect to sequence length, limiting its wider application, especially for long text.

- Long-form document classification with BERT. [Blogpost](https://andriymulyar.com/blog/bert-document-classification), [Code](https://github.com/AndriyMulyar/bert_document_classification)
- See ICLR 2020 reviews: 
  - [BERT-AL: BERT for Arbitrarily Long Document Understanding](https://openreview.net/forum?id=SklnVAEFDB)
  - [Blockwise Self-Attention for Long Document Understanding](https://openreview.net/forum?id=H1gpET4YDB)
- [Easy-to-use interface to fine-tuned BERT models for computing semantic similarity](https://github.com/AndriyMulyar/semantic-text-similarity)
- Ye, Z. et al. 2019. BP-Transformer: Modelling Long-Range Context via Binary Partitioning. (2019). [Paper](https://arxiv.org/pdf/1911.04070.pdf) [Code](https://github.com/yzh119/BPT)

### Document-level

- Doc2Vec [Paper](http://arxiv.org/abs/1507.07998), [Paper 2](http://www.jmlr.org/proceedings/papers/v32/le14.pdf)
- Fuzzy Bag-of-Words Model for Document Representation. [Paper](http://www.msrprojectshyd.com/upload/academicprojects/122b84310415012104464a741f5b6f9116.pdf)
### Topic-oriented

- LSA/LSI
- LDA
- LDA2Vec

### Citations

- Bibliographic coupling

- Co-Citation

- Co-Citation Proximity Analysis (+IDF)

#### Neural / Dense representations

- Cite2vec: Citation-Driven Document Exploration via Word Embeddings. [Paper](http://hdc.cs.arizona.edu/papers/tvcg_cite2vec_2017.pdf)

- hyperdoc2vec: Distributed Representations of Hypertext Documents. [Paper](https://arxiv.org/abs/1805.03793)

- Graph Embedding for Citation Recommendation. [Paper](https://arxiv.org/abs/1812.03835)

General graph embedding methods:

- DeepWalk: Online Learning of Social Representations. [Paper](https://arxiv.org/abs/1403.6652)
- LINE: Large-scale Information Network Embedding. [Paper](https://arxiv.org/abs/1503.03578). [Code](https://github.com/tangjianpku/LINE)
- node2vec. [Paper](https://www.kdd.org/kdd2016/papers/files/rfp0218-groverA.pdf)

Various implementations:
- [GraphVite - graph embedding at high speed and large scale](https://github.com/DeepGraphLearning/graphvite)


### Mathematical a

### Hybird

## Similarity / Distance Measures

Nearest neighbours in embedding space are considered to be similar.

-  Euclidean Distance

- Jaccard Similarity

- Jensen-Shannon distance

- Cosine similarity: Cosine-similarity treats all dimensions equally.

- Soft cosine

- Manhatten distance = L1 norm (see also [Manhattan LSTM](http://www.mit.edu/~jonasm/info/MuellerThyagarajan_AAAI16.pdf))

- Edit distance

- Levenshtein Distance

- Word Mover Distance [Paper](http://proceedings.mlr.press/v37/kusnerb15.pdf), [Paper 2](http://arxiv.org/abs/1811.01713)

- Supervised Word Moving Distance (S-WMD)

### Siamese Networks

Siamese networks [(Bromley, Jane, et al. "Signature verification using a siamese time delay neural network". Advances in neural information processing systems. 1994.)](http://papers.nips.cc/paper/769-signature-verification-using-a-siamese-time-delay-neural-network.pdf) are neural networks containing two or more identical subnetwork components.

> It is important that not only the architecture of the subnetworks is identical, but the weights have to be shared among them as well for the network to be called "siamese". The main idea behind siamese networks is that they can learn useful data descriptors that can be further used to compare between the inputs of the respective subnetworks. Hereby, inputs can be anything from numerical data (in this case the subnetworks are usually formed by fully-connected layers), image data (with CNNs as subnetworks) or even sequential data such as sentences or time signals (with RNNs as subnetworks).

- Siamese LSTM
- SMASH-RNN: [Jiang, J. et al. 2019. Semantic Text Matching for Long-Form Documents. The World Wide Web Conference on - WWW ’19 (New York, New York, USA, 2019), 795–806.](http://dl.acm.org/citation.cfm?doid=3308558.3313707)
- [Liu, B. et al. 2018. Matching Article Pairs with Graphical Decomposition and Convolutions. (Feb. 2018).](http://arxiv.org/abs/1802.07459) [(Code)](https://github.com/BangLiu/ArticlePairMatching)


#### Tasks

Binary classifcation

Multi-label classification

#### Loss functions

#### Concatenations

The two Siamese subnetworks encode input data into *u* and *v*. Various approaches exist to then concatenate the encoded input.

- Reimers, N. and Gurevych, I. 2019. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. The 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP 2019) (2019).

Variations

| Paper | Concatenation | Comment |
|-------|---------------|---------|
| InferSent | `(u;v;\|u-v\|;u*v)` | no evaluation |
| Sentence-BERT | `(u;v;\|u-v\|)` | The most important component is the element-wise difference `\|u−v\|` ... The element-wise difference measures the distance between the dimensions of the two sentence embeddings, ensuring that similar pairs are closer and dissimilar pairs are. |
| Universal Sentence Encoder | | |

#### MLP on top of Siamese sub networks

>  A matching network with multi-layer perceptron (MLP) is a standard way to mix multi-dimensions of information [(Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks)](https://doi.org/10.1145/2766462.2767738)


#### Applications

- Mueller, J. and Thyagarajan, A. 2014. Siamese Recurrent Architectures for Learning Sentence Similarity. Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence (AAAI-16). 2012 (2014), 1386–1393. DOI:https://doi.org/10.1109/CVPR.2014.180.



## See also

- [Text Similarities: Estimate the degree of similarity between two texts](https://medium.com/@adriensieg/text-similarities-da019229c894) [Repo](https://github.com/adsieg/text_similarity)
- [Text Classification Algorithms: A Survey](https://github.com/kk7nc/Text_Classification) [(Paper)](https://www.mdpi.com/2078-2489/10/4/150)
- [Michael J. Pazzani, Daniel Billsus. Content-Based Recommendation Systems](https://link.springer.com/chapter/10.1007/978-3-540-72079-9_10) [(PDF)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.130.8327&rep=rep1&type=pdf)
- [Charu C. Aggarwal. Content-Based Recommender Systems](https://link.springer.com/chapter/10.1007/978-3-319-29659-3_4)

Awesome:
- [Awesome Sentence Embeddings](https://github.com/Separius/awesome-sentence-embedding)
- [Awesome Neural Models for Semantic Match](https://github.com/NTMC-Community/awesome-neural-models-for-semantic-match)
- [Awesome Network Embedding](https://github.com/chihming/awesome-network-embedding)
