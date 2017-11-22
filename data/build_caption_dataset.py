## Utility Library
import threading
import os.path
import random
import sys
from datetime import datetime
from collections import Counter
from collections import namedtuple
from tqdm import tqdm
import timeit

## Scientific Library
import numpy as np
import tensorflow as tf

## Custom Library
from coco_utils import load_coco_data

output_dir_train = '../dataset/train' ## Set this to the convinient location for output of train shards
output_dir_test = '../dataset/test' ## Set this to the convinient location for output of test shards
output_dir_eval = '../dataset/eval' ## Set this to the convinient location for output of eval shards

num_of_threads = 16


def _float32_feature(value):
     return tf.train.Feature(float_list=tf.train.FloatList(value=[float(x) for x in value]))


def _int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value.astype(int)]))

def _int64_feature_list(values):
    """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])



def _to_sequence_example(vgg16_feature,caption_id):
    """ Build a SequenceExample proto for an image-caption pair.
    Args:
        vgg16_features: Features from fc7 of VGG16 Architecture
        caption_ids: List of integer ids corresponding to the caption words

    Returns:
        A SequenceExample proto.
    """

    context = tf.train.Features(feature={
        "image/vgg16_features":_float32_feature(vgg16_feature)
    })

    feature_lists = tf.train.FeatureLists(feature_list={
    "image/caption_ids":_int64_feature_list(caption_id)
    })
    sequence_example = tf.train.SequenceExample(
    context = context, feature_lists = feature_lists
    )

    return sequence_example

def _process_image_file(thread_idx, ranges, name, vgg16_features, caption_ids,num_shards,output_dir):
    """Processes and saves subset of images as TFRecord files in one thread

    Args:
      thread_idx: Integer thread identifier within [0, len(ranges)].
      ranges: A list of pairs of integers specifying the ranges of the dataset to
        process in parallel.
      name: Unique identifies specifying the dataset.
      vgg16_features: features form fc7 layer of VGG16 architecture.
      caption_ids: List of integer ids corresponding to the caption words.

    """

    #Each thread produces N shards where N = num_shards/num_threads.

    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards/num_threads)

    shard_ranges = np.linspace(ranges[thread_idx][0],ranges[thread_idx][1],
                               num_shards_per_batch+1).astype(int)

    num_images_in_thread = ranges[thread_idx][1] - ranges[thread_idx][0]

    counter = 0
    for s in xrange(num_shards_per_batch):

        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_idx*num_shards_per_batch + s
        output_filename = "%s-%.5d-of-%.5d" % (name, shard, num_shards)
        output_file = os.path.join(output_dir, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        vgg16_features_in_shard = np.arange(shard_ranges[s],shard_ranges[s+1],dtype=int)
        for i in vgg16_features_in_shard:
            vgg16_feature = vgg16_features[i]
            caption_id = caption_ids[i]

            sequence_example = _to_sequence_example(vgg16_feature,caption_id)
            if sequence_example is not None:
                writer.write(sequence_example.SerializeToString())
                shard_counter += 1
                counter += 1

            if not counter % 1000:
                print("%s [thread %d]: Processed %d of %d items in thread batch." %
                     (datetime.now(), thread_idx, counter, num_images_in_thread))
                sys.stdout.flush()

        writer.close()
        print("%s [thread %d]: Wrote %d image-caption pairs to %s" %
             (datetime.now(), thread_idx, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print("%s [thread %d]: Wrote %d image-caption pairs to %d shards." %
         (datetime.now(), thread_idx, counter, num_shards_per_batch))
    sys.stdout.flush()

def _process_dataset(name,vgg16_features,caption_ids,num_shards,output_dir):
    """ Process a complete data set and saves it as a TFRecord.
    Args:
      name: Unique identifier specifying the dataset
      vgg16_features: features from fc7 layer of VGG16 arcitecture
      caption_ids: List of integer ids corresponding to the caption words.
      num_shards: Integer number of shards for the output files.
    """
    num_threads = min(num_shards, num_of_threads)
    spacing =  np.linspace(0, len(vgg16_features),num_threads+1).astype(int)
    ranges = []
    threads = []
    for i in xrange(len(spacing) - 1):
        ranges.append([spacing[i],spacing[i+1]])

    #Create a mechanism for monitoring when all threads are finished
    coord = tf.train.Coordinator()

    #Launch thread for each batch
    print("Launching %d threads for spacing: %s"%(num_threads, ranges))
    for thread_idx in xrange(len(ranges)):
        args = (thread_idx, ranges, name, vgg16_features, caption_ids, num_shards,output_dir)
        print('Starting thread %d'%(thread_idx))
        t = threading.Thread(target=_process_image_file, args=args)
        t.start()
        threads.append(t)

    #Wait for all thread to terminate
    coord.join(threads)
    print("%s: Finished processing all %d image-caption pairs in data set '%s'." %
         (datetime.now(), len(vgg16_features), name))

def main(argv):
    print('Loading Data')
    start = timeit.default_timer()
    data = load_coco_data(pca_features=False)

    print('Data Loaded in %ds'%(timeit.default_timer()-start))

    ## Collect training data
    vgg16_features_train = []
    caption_ids_train = []
    print('Preparing training data')

    for i,idx in tqdm(enumerate(data['train_image_idxs'])):
        vgg16_features_train.append(data['train_features'][idx])
        caption_ids_train.append(data['train_captions'][i])


    ## Collect Validation data
    vgg16_features_val = []
    caption_ids_val = []
    print('Preparing validation data')

    for i,idx in tqdm(enumerate(data['val_image_idxs'])):
        vgg16_features_val.append(data['val_features'][idx])
        caption_ids_val.append(data['val_captions'][i])



    ## Redistribute the dataset
    train_cutoff = int(0.85 * len(vgg16_features_val))
    val_cutoff = int(0.90 * len(vgg16_features_val))

    vgg16_features_train = vgg16_features_train + vgg16_features_val[:train_cutoff]
    caption_ids_train = caption_ids_train + caption_ids_val[:train_cutoff]
    vgg16_features_test = vgg16_features_val[val_cutoff:]
    caption_ids_test = caption_ids_val[val_cutoff:]
    vgg16_features_val = vgg16_features_val[train_cutoff:val_cutoff]
    caption_ids_val = caption_ids_val[train_cutoff:val_cutoff]

    print("Length of training data: %d"%(len(vgg16_features_train)))
    print("Length of validation data: %d"%(len(vgg16_features_val)))
    print("Length of test data: %d"%(len(vgg16_features_test)))
    print("Vocabulary Size: %d"%(len(data['idx_to_word'])))

    start = timeit.default_timer()
    print('Preparing tf records for training data: ')
    _process_dataset("train", vgg16_features_train,caption_ids_train,256,output_dir_train)
    print('Completed in %ds'%(timeit.default_timer()-start))
    start = timeit.default_timer()
    print('Preparing tf records for validation data: ')
    _process_dataset("val", vgg16_features_val,caption_ids_val,4,output_dir_eval)
    print('Completed in %ds'%(timeit.default_timer()-start))
    start = timeit.default_timer()
    print('Preparing tf records for testingdata: ')
    _process_dataset("test", vgg16_features_test,caption_ids_test,8,output_dir_test)
    print('Completed in %ds'%(timeit.default_timer()-start))

if __name__ == "__main__":
    tf.app.run()
