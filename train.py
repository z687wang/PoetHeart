import tensorflow as tf
import numpy as np
import time
from rnn import RNN, get_batches
from load_data import get_data

batch_size = 10
num_steps = 50          
lstm_size = 128         
num_layers = 2          
learning_rate = 0.01    
keep_prob = 0.5

epochs = 20
print_every_n = 50
save_every_n = 200

text, encoded, vocab, int_to_vocab, vocab_to_int = get_data()

model = RNN(len(vocab), batch_size=batch_size, num_steps=num_steps,
                lstm_size=lstm_size, num_layers=num_layers, 
                learning_rate=learning_rate)


def train():
    saver = tf.train.Saver(max_to_keep=100)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        # Use the line below to load a checkpoint and resume training
        #saver.restore(sess, 'checkpoints/______.ckpt')
        counter = 0
        for e in range(epochs):
            # Train network
            new_state = sess.run(model.initial_state)
            loss = 0
            for x, y in get_batches(encoded, batch_size, num_steps):
                counter += 1
                start = time.time()
                feed = {model.inputs: x,
                        model.targets: y,
                        model.keep_prob: keep_prob,
                        model.initial_state: new_state}
                batch_loss, new_state, _ = sess.run([model.loss, 
                                                    model.final_state, 
                                                    model.optimizer], 
                                                    feed_dict=feed)
                if (counter % print_every_n == 0):
                    end = time.time()
                    print('Epoch: {}/{}... '.format(e+1, epochs),
                        'Training Step: {}... '.format(counter),
                        'Training loss: {:.4f}... '.format(batch_loss),
                        '{:.4f} sec/batch'.format((end-start)))
            
                if (counter % save_every_n == 0):
                    saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))
        
        saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))
train()
