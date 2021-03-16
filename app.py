from flask import Flask, request, render_template
import pickle
import os

dir_path = os.getcwd()

app = Flask(__name__)

clf_file = os.path.join(dir_path, 'model', 'final_model.pkl')
clf = pickle.load(open(clf_file, 'rb'))

# 'model/final_vectorizer.pkl'
# 'model/final_model.pkl'

### load vectorizer
vec_file = os.path.join(dir_path, 'model', 'final_vectorizer.pkl')
vectorizer = pickle.load(open(vec_file, 'rb'))

@app.route('/')

def my_form():
	return render_template('form1.html', p = "")

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
		i ='Comment is not toxic'
		return render_template('form1.html', p = i)
	else:
		i = str(predc)
		return render_template('form1.html', p = i)

	return i

	

if __name__ == '__main__':
	app.run(port = 13000, use_reloader = True, debug=True)

