from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

clf = pickle.load(open('model/MultinomialNB', 'rb'))
vectorizer = pickle.load(open('model/bw_vectorizer1000.pkl', 'rb'))

@app.route('/')

def my_form():
	return render_template('form.html')

# @app.route('/', methods=['POST'])
# def my_form_post():
# 	text = request.form['u']
# 	processed_text = text.upper()
# 	return processed_text

@app.route('/', methods = ['POST'])
def check_toxicity():
	text = request.form['u']
	processed_text = text.lower()
	processed_text = vectorizer.transform([processed_text])
	prediction = clf.predict(processed_text)


	labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
	predc = []
	for i,j in zip(prediction[0], labels):
		if i == 1:
			predc.append(j)
	if len(predc)== 0:
		i ='comment in not toxic'
	else:
		i = str(predc)


	return i

	

if __name__ == '__main__':
	app.run(use_reloader = True)

