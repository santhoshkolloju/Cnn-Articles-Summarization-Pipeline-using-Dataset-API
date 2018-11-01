# Cnn-Articles-Summarization-Pipeline-using-Tensorflow Dataset-API

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
import multiprocessing
PAD_TOKEN = '[PAD]'
UNKNOWN_TOKEN = '[UNK]'
START_DECODING = '[START]'
STOP_DECODING = '[STOP]'


#helper functions
def load_doc(filename):
	# open the file as read only
	file = open(filename, encoding='utf-8')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text
 
 # clean a list of lines
def clean_lines(lines):
	cleaned = list()
	# prepare a translation table to remove punctuation
	table = str.maketrans('', '', string.punctuation)
	for line in lines:
		# strip source cnn office if it exists
		index = line.find('(CNN) -- ')
		if index > -1:
			line = line[index+len('(CNN)'):]
		# tokenize on white space
		line = line.split()
		# convert to lower case
		line = [word.lower() for word in line]
		# remove punctuation from each token
		line = [w.translate(table) for w in line]
		# remove tokens with numbers in them
		line = [word for word in line if word.isalpha()]
		# store as string
		cleaned.append(' '.join(line))
	# remove empty strings
	cleaned = [c for c in cleaned if len(c) > 0]
	return cleaned 
  
  def split_story(file):
    # find first highlight
    filename = file.decode(sys.getdefaultencoding())
    doc  = load_doc(filename)
    index = doc.find('@highlight')
    # split into story and highlights
    story, highlights = doc[:index], doc[index:].split('@highlight')
    # strip extra white space around each highlight
    highlights = [h.strip() for h in highlights if len(h) > 0]
    story = clean_lines(story)
    highlights = clean_lines(highlights)
    return " ".join(story[:max_enc_steps]), " ".join(highlights[:max_dec_steps])
 
  
  

#vocab file is \n spearted text file with a single word in each line it should also . contain the pad token and unknown token ,
start and stop token at the beginning

vocab_table  =tf.contrib.lookup.index_table_from_file(vocabulary_file ='vocablefile.txt', default_value = 0, 
                                                      delimiter ="\n",num_oov_buckets =0
                                                      )
print("Number of cores on your system",multiprocessing.cpu_count())
#dont use all the cores 

#path to cnn stories
files =  glob(../cnn/stories/*)

dataset = tf.data.Dataset.from_tensor_slices((files))

dataset =  dataset.map(lambda file : tuple(
tf.py_func(split_story,[file],[tf.string,tf.string])
),num_parallel_calls = 16)

dataset = dataset.map(lambda story,summaty : (tf.string_split([story]).values,tf.string_split([summary]).values),num_parallel_calls = 16)

dataset = datset.map(lambda story,summary :
{"stroy_tokens":vocab_table.lookup(story),
"summary_tokens":vocab_table.lookup(summary)
"story_len":tf.size(story),"summary_len":tf.size(summary)
}
,num_parallel_calls = 16)

dataset.padded_batch(4,
                      padded_shapes = {
                      "story_tokens" : tf.TensorShape([None]),#None means padd it to longest length
                      "summary_tokens" : tf.TensorShape([None],
                      "story_len":[],#No padding 
                      "summary_len":[]
                      
                      ),
                      "padding_values"={
                      "story_tokens" :tf.cast(1,tf.int64),
                      "summary_tokens" : tf.cast(1,tf.int64),
                      "story_len": tf.cast(1,tf.int64),
                      "summary_len":tf.cast(1,tf.int64) #thought we are not padding this elements still some dummy value
                                                         #needs to be given some bug in tnesoflow 
                      
                      }

)

dataset = dataset.prefetch(4) #prefetch 4 batches

iterator  = datset.make_intializable_iterator()

sess =tf.Session()
sess.run(tf.tables_intializer())
sess.run(iterator.intializer)
next_batch = iterator.get_next()
print(sees.run(next_element)) # returns numpy arrays of a single batch

</pre>
