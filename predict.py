import pickle
import tensorflow as tf
from preprocessing import apply_tokenizer, text_for_pred

# loading model
model = tf.keras.models.load_model('best_model.h5')

# loading tokenizer 
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# text to predict for
txt = ["A Fencer Strives to Crack a Saber Ceiling"]
prediction = text_for_pred(tokenizer, txt, model)

if prediction[0][0] > prediction[0][1]:
	print("Phew! You are safe! Go ahead...")
else:
	print("It's a clickbait!!!!!")

