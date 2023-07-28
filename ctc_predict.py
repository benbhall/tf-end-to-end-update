import argparse
import tensorflow as tf
import ctc_utils
import cv2
import numpy as np

parser = argparse.ArgumentParser(description='Decode a music score image with a trained model (CTC).')
parser.add_argument('-image',  dest='image', type=str, required=True, help='Path to the input image.')
parser.add_argument('-model', dest='model', type=str, required=True, help='Path to the trained model.')
parser.add_argument('-vocabulary', dest='voc_file', type=str, required=True, help='Path to the vocabulary file.')
args = parser.parse_args()

# Read the dictionary
dict_file = open(args.voc_file,'r')
dict_list = dict_file.read().splitlines()
int2word = dict()
for word in dict_list:
    word_idx = len(int2word)
    int2word[word_idx] = word
dict_file.close()

# Load the saved model
model = tf.keras.models.load_model(args.model)

# Constants that are saved inside the model itself
WIDTH_REDUCTION, HEIGHT = model.get_layer('width_reduction').get_weights()[0], model.get_layer('input_height').get_weights()[0]

image = cv2.imread(args.image,False)
image = ctc_utils.resize(image, HEIGHT)
image = ctc_utils.normalize(image)
image = np.asarray(image).reshape(1,image.shape[0],image.shape[1],1)

seq_lengths = [ image.shape[2] / WIDTH_REDUCTION ]

# Predict
prediction = model.predict(image)

# Decode prediction
decoded = tf.nn.ctc_decode(prediction, seq_lengths, greedy=True)

str_predictions = ctc_utils.sparse_tensor_to_strs(decoded)
for w in str_predictions[0]:
    print (int2word[w]),
    print ('\t'),
