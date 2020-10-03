import tensorflow as tf
from preprocessing import apply_tokenizer, text_for_pred


model = tf.keras.models.load_model('best_model.h5')

txt = ["A Fencer Strives to Crack a Saber Ceiling"]
prediction = text_for_pred(tokenizer, txt, model)

if prediction[0][0] > prediction[0][1]:
	print("Phew! You are safe! GO ahead...")
else:
	print("It's a clickbait!!!!!")

