import tensorflow as tf
from primus import CTC_PriMuS
import ctc_utils
import ctc_model
import argparse
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

import os

parser = argparse.ArgumentParser(description='Train model.')
parser.add_argument('-corpus', dest='corpus', type=str, required=True, help='Path to the corpus.')
parser.add_argument('-set',  dest='set', type=str, required=True, help='Path to the set file.')
parser.add_argument('-save_model', dest='save_model', type=str, required=True, help='Path to save the model.')
parser.add_argument('-vocabulary', dest='voc', type=str, required=True, help='Path to the vocabulary file.')
parser.add_argument('-semantic', dest='semantic', action="store_true", default=False)
args = parser.parse_args()

# Load primus
primus = CTC_PriMuS(args.corpus,args.set,args.voc, args.semantic, val_split = 0.1)

# Parameterization
img_height = 128
params = ctc_model.default_model_params(img_height,primus.vocabulary_size)
max_epochs = 64000
dropout = 0.5

# Create the model
model = ctc_model.ctc_crnn(params)

# Define the optimizer
optimizer = tf.keras.optimizers.Adam()

# Training loop
for epoch in range(max_epochs):
    batch = primus.nextBatch(params)

    with tf.GradientTape() as tape:
        # Forward pass
        logits = model(batch['inputs'])
            # Compute the loss value
        sparse_targets = ctc_utils.sparse_tuple_from(batch['targets'])
        loss_value = tf.nn.ctc_loss(labels=sparse_targets,
                                     logits=logits, 
                                     label_length=None, 
                                     logit_length=batch['seq_lengths'], 
                                     logits_time_major=False, 
                                     blank_index=-1)

    # Compute gradients
    grads = tape.gradient(loss_value, model.trainable_variables)
    # Update weights
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if epoch % 1000 == 0:
        print ('Loss value at epoch ' + str(epoch) + ':' + str(loss_value.numpy()))
        # Save the model
        model.save(args.save_model + '_' + str(epoch))
        # Convert the model to a frozen graph and save as .pb file
        loaded = tf.saved_model.load(args.save_model + '_' + str(epoch))
        infer = loaded.signatures['serving_default']
        frozen_func = convert_variables_to_constants_v2(infer)
        frozen_func.graph.as_graph_def()
        with tf.io.gfile.GFile(args.save_model + '_' + str(epoch) + '.pb', 'wb') as f:
            f.write(frozen_func.graph.as_graph_def().SerializeToString())



# import tensorflow as tf
# from primus import CTC_PriMuS
# import ctc_utils
# import ctc_model
# import argparse

# import os

# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# tf.reset_default_graph()
# sess = tf.InteractiveSession(config=config)

# parser = argparse.ArgumentParser(description='Train model.')
# parser.add_argument('-corpus', dest='corpus', type=str, required=True, help='Path to the corpus.')
# parser.add_argument('-set',  dest='set', type=str, required=True, help='Path to the set file.')
# parser.add_argument('-save_model', dest='save_model', type=str, required=True, help='Path to save the model.')
# parser.add_argument('-vocabulary', dest='voc', type=str, required=True, help='Path to the vocabulary file.')
# parser.add_argument('-semantic', dest='semantic', action="store_true", default=False)
# args = parser.parse_args()

# # Load primus

# primus = CTC_PriMuS(args.corpus,args.set,args.voc, args.semantic, val_split = 0.1)

# # Parameterization
# img_height = 128
# params = ctc_model.default_model_params(img_height,primus.vocabulary_size)
# max_epochs = 64000
# dropout = 0.5

# # Model
# inputs, seq_len, targets, decoded, loss, rnn_keep_prob = ctc_model.ctc_crnn(params)
# train_opt = tf.train.AdamOptimizer().minimize(loss)


# saver = tf.train.Saver(max_to_keep=None)
# sess.run(tf.compat.v1.global_variables_initializer())

# # Training loop
# for epoch in range(max_epochs):
#     batch = primus.nextBatch(params)

#     _, loss_value = sess.run([train_opt, loss],
#                              feed_dict={
#                                 inputs: batch['inputs'],
#                                 seq_len: batch['seq_lengths'],
#                                 targets: ctc_utils.sparse_tuple_from(batch['targets']),
#                                 rnn_keep_prob: dropout,
#                             })

#     if epoch % 1000 == 0:
#         # VALIDATION
#         print ('Loss value at epoch ' + str(epoch) + ':' + str(loss_value))
#         print ('Validating...')

#         validation_batch, validation_size = primus.getValidation(params)
        
#         val_idx = 0
        
#         val_ed = 0
#         val_len = 0
#         val_count = 0
            
#         while val_idx < validation_size:
#             mini_batch_feed_dict = {
#                 inputs: validation_batch['inputs'][val_idx:val_idx+params['batch_size']],
#                 seq_len: validation_batch['seq_lengths'][val_idx:val_idx+params['batch_size']],
#                 rnn_keep_prob: 1.0            
#             }            
                        
            
#             prediction = sess.run(decoded,
#                                   mini_batch_feed_dict)
    
#             str_predictions = ctc_utils.sparse_tensor_to_strs(prediction)
    

#             for i in range(len(str_predictions)):
#                 ed = ctc_utils.edit_distance(str_predictions[i], validation_batch['targets'][val_idx+i])
#                 val_ed = val_ed + ed
#                 val_len = val_len + len(validation_batch['targets'][val_idx+i])
#                 val_count = val_count + 1
                
#             val_idx = val_idx + params['batch_size']
    
#         print ('[Epoch ' + str(epoch) + '] ' + str(1. * val_ed / val_count) + ' (' + str(100. * val_ed / val_len) + ' SER) from ' + str(val_count) + ' samples.')        
#         print ('Saving the model...')
#         saver.save(sess,args.save_model,global_step=epoch)
#         print ('------------------------------')
