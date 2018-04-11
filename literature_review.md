===========================================================================================
# Papers directly related to our work

## Topic Compositional Neural Language Model (2018)
- Datasets: APNEWS, IMDB and BNC
- LSTM model designed to simultaneously capture both the global semantic meaning and the local word- ordering structure in a document
- Compare their models against several baseline models (including the LCLM model!)

## Topic-RNN: an RNN with long-range semantic dependency (2017)
- Datasets: Penn Treebank, IMDB
- Extend RNN with topic model

## Topically Driven Neural Language Model (2017)
- Datasets: APNews, IMDB, BNC
- Combine LSTM with topic model

## Contextual LSTM (CLSTM) models for Large scale NLP tasks (2016)
- Dataset: English Wikipedia and subset of articles from English Google News
- Extend LSTM model with different contextual features (e.g. topics)

## Personalizing universal RNNLM with user characteristic features by social network crowdsourcing 
- Dataset: crawled facebook corpus
- Augmented RNNLM with user-specific input vector which aufments the 1-of-N encoding feature of each word

## Efficient Transfer Learning Schemes for Personalized Language Modeling Using Recurrent Neural Network (2017)
- Dataset: WMT14 ENG corpus, English bible corpus, drama (Friends) corpus
- Test three transfer learning schemes to create personalized language models

## Deep Multi-Task Learning with Shared Memory (2016)
- Dataset: movie reviews and amazon reviews
- Equip task-specific LSTM network with an external shared memory 

## Combining LSTM and Latent Topic Modeling for Mortality Prediction (2017)
- Dataset: MIMIC-III dataset which contains data about patients admitted to critical care  
- Combine LSTM and latent topic model in different ways

## Document context language models (2016)
- Datset: Penn Treebank, subset of North American News Text corpus
- Present different LSTM architectures that incorporate contextual information

## Larger-Context language modelling with RNNs (2016)
- Datasets: IMDB, BBC, Penn Treebank
- Extend LSTM with contextual information e.g. using BoW vector of preceding sentence

===========================================================================================
# Papers indirectly related to our work

## Generative Knowledge Transfer for Neural Language Models (2017)
- Dataset: WSJ corpus
- Propose generative knowledge transfer technique that trains RNNLM (student network) using text and output probabilities generated from a previously trained RNN (teacher network)


## A Unified Architecture for Natural Language Processing: Deep Neural Networks with Multitask Learning (2008)
- Dataset:
- Single CNN that, given a sentence, outputs a host of language processing predictions

## Domain Adaptation for Large-Scale Sentiment Classification: A Deep Learning Approach (2011)
- Dataset: Amazon reviews, available at: http://www.cs.jhu.edu/~mdredze/datasets/sentiment/

