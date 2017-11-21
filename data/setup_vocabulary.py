import numpy as np
from coco_utils import load_coco_data

BASE_VOCAB_PATH = '/media/srivatsa/982ED6FB2ED6D17C/glove.6B/glove.6B.'


def load_vocabulary_embedding(dimension=50):
    """
    Params:
    dimension: the dimensionality of the word vector
    Returns:
    A map of word to index
    Inverse map of index to map
    numpy array of word vectors with index mapping as above
    """
    coco_data = load_coco_data()
    embeddings = np.random.rand(len(coco_data['word_to_idx']), dimension)
    assert(dimension == 50 or dimension ==
           100 or dimension == 200 or dimension == 300)
    vocab_path = BASE_VOCAB_PATH + str(dimension) + 'd.txt'
    ctr,ctr1 = 0,1
    with open(vocab_path, 'r') as vocabulary_file:
        for idx, line in enumerate(vocabulary_file):
            tokens = line.split(' ')
            try:
                idx = coco_data['word_to_idx'][tokens[0]]
                embeddings[idx] = map(float, tokens[1:])
                ctr1 += 1
                #x = input()
            except KeyError:
                ctr += 1
                continue
            if(idx % 10000 == 0):
                print('%d tokens processed'%(idx))

    print('%d words not present while %d words processed'%(ctr,ctr1))

    return np.array(embeddings)
