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
	i = [labels[i] for i in range (0,len(prediction[0])) if prediction[0][i] == 1]
	if len(i)== 0:
		out ='Comment is not toxic'
		return render_template('form1.html', p = out)
	else:
		out = " | ".join(i)
		return render_template('form1.html', p = out)
	return render_template('form1.html', p = "")

	

if __name__ == '__main__':
	app.run(port = 13000, use_reloader = True, debug=True)

