
import os, time, argparse, logging
from datetime import datetime
from pathlib import Path

from tqdm import trange
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from models import Encoder, Decoder, Transformer
from masks import create_masks
from processing import Preprocess

import os
CUDA_VISIBLE_DEVICES=0

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info('Reading Arguments')

parser = argparse.ArgumentParser()

# Preprocessing ---------------------------------------------------------------
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--max_length', type=int, default=20)
parser.add_argument('--buffer_size', type=int, default=2000)

# Model -----------------------------------------------------------------------
parser.add_argument('--num_layers',  type=int, default=2)
parser.add_argument('--d_model', type=int, default=64)
parser.add_argument('--dff', type=int, default=256)
parser.add_argument('--num_heads',  type=int, default=4)
parser.add_argument('--dropout_rate', type=float, default=0.05)

# Training --------------------------------------------------------------------
parser.add_argument('--epochs', type=int, default=5)

# Saving / Logging ------------------------------------------------------------
parser.add_argument('--log_dir', type=str, default='./logs/')
parser.add_argument("--log_freq", type=int, default=2)
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/')
parser.add_argument('--checkpoint_freq',type=int, default=2)
parser.add_argument('--extension', type=str, default=None)

cfg = parser.parse_args()

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        
        logger.info('Initializing CustomSchedule')

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps
        
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


class TMLU():
    def __init__(self):
        super(TMLU, self).__init__()

        logger.info('Initializing TMLU')

        self.global_step = 0

        # Data Pipeline
        data_pipeline = Preprocess(cfg)
        self.train_dataset, self.val_dataset = data_pipeline.get_data()

        self.tokenizer_pt = data_pipeline.tokenizer_pt
        self.tokenizer_en = data_pipeline.tokenizer_en

        cfg.input_vocab_size = self.tokenizer_pt.vocab_size + 2
        cfg.target_vocab_size = self.tokenizer_en.vocab_size + 2

        # Model
        self.transformer = Transformer(cfg)

        # Optimizer
        learning_rate = CustomSchedule(cfg.d_model)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        # Loss and Metrics
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        # Build writers for logging
        self.build_writers()

        checkpoint_path = "./checkpoints/train"
        self.ckpt = tf.train.Checkpoint(transformer=self.transformer, optimizer=self.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, checkpoint_path, max_to_keep=5)

    def build_writers(self):

        logger.info('Initializing Build Writers')

        if not Path(cfg.checkpoint_dir).is_dir():
            os.mkdir(cfg.checkpoint_dir)

        if not Path(cfg.log_dir).is_dir():
            os.mkdir(cfg.log_dir)

        if cfg.extension is None:
            cfg.extension = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        self.log_path = cfg.log_dir + cfg.extension
        self.writer = tf.summary.create_file_writer(self.log_path)
        self.writer.set_as_default()

        #self.saved_models_dir = cfg.saved_models_dir
        self.checkpoint_dir = cfg.checkpoint_dir
        self.checkpoint_prefix = self.checkpoint_dir + "ckpt_{epoch}"

    def log_scalar(self, name, scalar):
        if (self.global_step % cfg.log_freq) == 0:
            tf.summary.scalar(name, scalar, step=self.global_step)

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        
        return tf.reduce_mean(loss_)

    def train_step(self, inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
        
        with tf.GradientTape() as tape:
            predictions, _ = self.transformer(inp,tar_inp, True, 
                enc_padding_mask, combined_mask, dec_padding_mask)

            loss = self.loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, self.transformer.trainable_variables)    
        self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(tar_real, predictions)

        self.log_scalar('Training/Loss', self.train_loss.result())
        self.log_scalar('Training/Accuracy', self.train_accuracy.result())

        self.global_step += 1

    def evaluate(self, inp_sentence):
        start_token = [self.tokenizer_pt.vocab_size]
        end_token = [self.tokenizer_pt.vocab_size + 1]
        
        # inp sentence is portuguese, hence adding the start and end token
        inp_sentence = start_token + self.tokenizer_pt.encode(inp_sentence) + end_token
        encoder_input = tf.expand_dims(inp_sentence, 0)
        
        # as the target is english, the first word to the transformer should be the
        # english start token.
        decoder_input = [self.tokenizer_en.vocab_size]
        output = tf.expand_dims(decoder_input, 0)

        for i in range(cfg.max_length):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                encoder_input, output)
        
            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = self.transformer(encoder_input, 
                                                        output,
                                                        False,
                                                        enc_padding_mask,
                                                        combined_mask,
                                                        dec_padding_mask)
            
            # select the last word from the seq_len dimension
            predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
            
            # return the result if the predicted_id is equal to the end token
            if tf.equal(predicted_id, self.tokenizer_en.vocab_size+1):
                return tf.squeeze(output, axis=0), attention_weights
            
            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)

        return tf.squeeze(output, axis=0), attention_weights

    def translate(self, sentence, plot=''):
        result, attention_weights = self.evaluate(sentence)
        
        predicted_sentence = self.tokenizer_en.decode([i for i in result if i < self.tokenizer_en.vocab_size])  

        print('Input: {}'.format(sentence))
        print('Predicted translation: {}'.format(predicted_sentence))
        

    def checkpoint(self):

        self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint:
            print("Restored from {}".format(self.ckpt_manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

    def train(self):
        
        logger.info('Training')

        self.checkpoint()


        for epoch in trange(cfg.epochs):
            
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            
            # inp -> portuguese, tar -> english
            for (batch, (inp, tar)) in enumerate(self.train_dataset):
                self.train_step(inp, tar)
                
            if (epoch + 1) % 5 == 0:
                ckpt_save_path = self.ckpt_manager.save()
                print ('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))
                
            self.translate("este é um problema que temos que resolver.")
            print ("Real translation: this is a problem we have to solve .")

    def test(self):
        
        logger.info('Testing')

        self.checkpoint()

        self.translate("este é um problema que temos que resolver.")
        print("Real translation: this is a problem we have to solve .")

        self.translate("os meus vizinhos ouviram sobre esta ideia.")
        print("Real translation: and my neighboring homes heard about this idea .")

        self.translate("vou então muito rapidamente partilhar convosco algumas histórias de algumas coisas mágicas que aconteceram.")
        print("Real translation: so i 'll just share with you some stories very quickly of some magical things that have happened .")

        self.translate("este é o primeiro livro que eu fiz.")
        print("Real translation: this is the first book i've ever done.")



def main():
    
    # tf.config.gpu.set_per_process_memory_growth()
    tmlu = TMLU()
    tmlu.train()
    tmlu.test()


if __name__ == '__main__':
    main()