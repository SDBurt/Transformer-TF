
import tensorflow as tf
import tensorflow_datasets as tfds

class Preprocess():
    def __init__(self, cfg):
        super(Preprocess, self).__init__()

        
        self.buffer_size = cfg.buffer_size
        self.batch_size = cfg.batch_size
        self.max_length = cfg.max_length

        self.train_dataset = []
        self.val_dataset = []

        # Use TFDS to load the Portugese-English translation dataset from the TED Talks Open Translation Project.
        examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                        with_info=True, as_supervised=True)
        self.train_examples, self.val_examples = examples['train'], examples['validation']

        # Create a custom subwords tokenizer from the training dataset. 
        self.tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in self.train_examples), target_vocab_size=2**13)
        self.tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in self.train_examples), target_vocab_size=2**13)

    def encode(self, lang1, lang2):
        lang1 = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(
            lang1.numpy()) + [self.tokenizer_pt.vocab_size+1]

        lang2 = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(
            lang2.numpy()) + [self.tokenizer_en.vocab_size+1]

            
        return lang1, lang2

    def tf_encode(self, pt, en):
        return tf.py_function(self.encode, [pt, en], [tf.int64, tf.int64])

    def filter_max_length(self, x, y):
        return tf.logical_and(tf.size(x) <= self.max_length,
                              tf.size(y) <= self.max_length)

    def get_data(self):
        
        self.train_dataset = self.train_examples.map(self.tf_encode)
        self.train_dataset = self.train_dataset.filter(self.filter_max_length)
        self.train_dataset = self.train_dataset.cache()
        self.train_dataset = self.train_dataset.shuffle(self.buffer_size).padded_batch(
            self.batch_size, padded_shapes=([-1], [-1]))
        self.train_dataset = self.train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        self.val_dataset = self.val_examples.map(self.tf_encode)
        self.val_dataset = self.val_dataset.filter(self.filter_max_length).padded_batch(
            self.batch_size, padded_shapes=([-1], [-1]))

        return self.train_dataset, self.val_dataset
