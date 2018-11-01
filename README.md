# Cnn-Articles-Summarization-Pipeline-using-Dataset-API

Dataset API from tensorflow is used for building the complex data pipelines for building machine learning or deep learning models

The pipeline build using Dataset API by default is part of tensorflow graph

What problems tensorflow Dataset API can solve.

1)CPU - GPU problem
GPU doesnt have any memory all the data preparation happens in CPU.while the GPU finishes trainiing first batch of data it 
requests the CPU for next batch until CPU prepares the data GPU is idle . to solve this problem we can inturn in background 
process the data in CPU before GPU finishes the training of a batch.

<img src="https://github.com/santhoshkolloju/Cnn-Articles-Summarization-Pipeline-using-Dataset-API/blob/master/Screen%20Shot%202018-11-01%20at%204.57.58%20PM.png"/>

dataset = dataset.prefetch(4)
This command prefetches 4 batches in the queue always.

2) Parallelize the data transformations
<img src="https://github.com/santhoshkolloju/Cnn-Articles-Summarization-Pipeline-using-Dataset-API/blob/master/Screen%20Shot%202018-11-01%20at%205.10.59%20PM.png"/>

When preparing a batch of data we may need to preprocess the data, as this operation is independent for each example we can 
run it on multiple cores of the cpu

dataset = dataset.map(map_fn = parse_fn,num_parallel_calls = 16)

3)Caching Data
All the above operations are lazy operations they happen only when the data is requested.
Instead if your data can fit into the memory use the cache transformation to cache it in the memory during the first epoch so 
that subsequent epochs avoid the memory overhead associated with reading parsing and transforming it
dataset.cache()


<h1>CNN Daily Mail Data set Example </h1>

cnn daily mail data contain raw file(.story) lets see how to parse them using the data set API and build a pipeline.

<pre>
import tensorflow as tf
from glob import glob
import numpy as np
impot string
PAD_TOKEN = '[PAD]'
UNKNOWN_TOKEN = '[UNK]'
START_DECODING = '[START]'
STOP_DECODING = '[STOP]'

#vocab file is \n spearted text file with a single word in each line it should also . contain the pad token and unknown token ,
start and stop token at the beginning

vocab_table  =tf.contrib.lookup.index_table_from_file(vocabulary_file ='vocablefile.txt')

</pre>
