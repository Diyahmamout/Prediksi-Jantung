from flask import Flask, render_template
from sklearn.datasets import load_wine
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from flask import request
import pandas as pd
import csv
from sklearn.naive_bayes import GaussianNB
import numpy as np
from joblib import dump, load
import array
from decimal import Decimal

app = Flask(__name__)


clf = load('heart.joblib')
clf_250 = load('heart_250.joblib')
clf_200 = load('heart_200.joblib')
clf_100 = load('heart_100.joblib')

medic = pd.read_csv('data.csv')



np.set_printoptions(suppress=True, infstr='inf', formatter={'complex_kind':'{:.10f}'.format})

@app.route('/', methods=['GET','POST'])
def index():
	y_250 = medic.iloc[:250,10].values.astype('int32')
	x_250 = (medic.iloc[:250,0:10].values).astype('int32')
	(xTrain250, xTest250, yTrain250, yTest250) = train_test_split(x_250, y_250, test_size = 0.25, random_state = 42)
	accuracy_250 = '{:.2f}'.format(float(accuracy_score(yTest250, clf_250.predict(xTest250))))
	accuracy_250 = float(accuracy_250)*100

	y_200 = medic.iloc[:200,10].values.astype('int32')
	x_200 = (medic.iloc[:200,0:10].values).astype('int32')
	(xTrain200, xTest200, yTrain200, yTest200) = train_test_split(x_200, y_200, test_size = 0.25, random_state = 42)
	accuracy_200 = '{:.2f}'.format(float(accuracy_score(yTest200, clf_200.predict(xTest200))))
	accuracy_200 = float(accuracy_200)*100

	y_100 = medic.iloc[:100,10].values.astype('int32')
	x_100 = (medic.iloc[:100,0:10].values).astype('int32')
	(xTrain100, xTest100, yTrain100, yTest100) = train_test_split(x_100, y_100, test_size = 0.25, random_state = 42)
	accuracy_100 = '{:.2f}'.format(float(accuracy_score(yTest100, clf_100.predict(xTest100))))
	accuracy_100 = float(accuracy_100)*100

	y = medic.iloc[:294,10].values.astype('int32')
	x = (medic.iloc[:294,0:10].values).astype('int32')
	(xTrain, xTest, yTrain, yTest) = train_test_split(x, y, test_size = 0.25, random_state = 42)
	accuracy = '{:.2f}'.format(float(accuracy_score(yTest, clf.predict(xTest))))
	accuracy = float(accuracy)*100

	data = pd.read_csv("heart data.csv")
	heart_data = data.iloc[:, 0:11]
	heart_label = data.iloc[:, 10]

	print(heart_data)
	data_prediksi = []
	data_label = []
	data_names = [0,1]
	data_sample = []
	data_sample2 = []

	for i in range(293):
	    data_prediksi.append(heart_data.iloc[i].tolist())
	    data_label.append(heart_label.iloc[i].tolist())
	    data_prediksi[i][5] = int(data_prediksi[i][5])
		        
	counter = 0
	for n in range(293):
		if counter < 5 :
		    if data_prediksi[n][10] == 0 : 
		        data_sample.append(data_prediksi[n])
		        counter = counter + 1 
	counter = 0
	for n in range(293):
		if counter < 5 :
		    if data_prediksi[n][10] == 1 : 
		        data_sample.append(data_prediksi[n])
		        counter = counter + 1 
	   
	(trainX, testX, trainY, testY) = train_test_split(data_prediksi, data_label, test_size = 0.25, random_state = 42)

	if (request.method == 'POST'):
		data = request.form.to_dict([])
		nama = data['nama']
		umur = pd.to_numeric(data['umur'])
		sex = pd.to_numeric(data['sex'])
		nyeri = pd.to_numeric(data['nyeri'])
		tekanan = pd.to_numeric(data['tekanan_darah'])
		kolesterol = pd.to_numeric(data['kolesterol'])
		guldar = pd.to_numeric(data['guldar'])
		elektrokardiografi = pd.to_numeric(data['elektrokardiografi'])
		denyut = pd.to_numeric(data['denyut'])
		sesak = pd.to_numeric(data['sesak'])
		depresi = pd.to_numeric(data['depresi'])
		
		xPredict = np.array([[umur, sex, nyeri, tekanan, kolesterol, guldar, elektrokardiografi, denyut, sesak, depresi]])
		xPredict = xPredict.ravel()
		predict = [xPredict]

		#xPredict = [[54,0,3,130,294,0,1,100,1,0]]
		print("Hasil:")

		print(clf.predict(predict))

		res = clf.predict(predict)
		res_250 = clf_250.predict(predict)
		res_200 = clf_200.predict(predict)
		res_100 = clf_100.predict(predict)

		res_prob = clf.predict_proba(predict)
		res_prob_1 = '{:.5f}'.format(float(np.squeeze(clf.predict_proba(predict)[:,0], axis=0)))
		res_prob_2 = '{:.5f}'.format(float(np.squeeze(clf.predict_proba(predict)[:,1], axis=0)))

		res_prob_250 = clf_250.predict_proba(predict)
		res_prob_250_1 = '{:.5f}'.format(float(np.squeeze(clf_250.predict_proba(predict)[:,0], axis=0)))
		res_prob_250_2 = '{:.5f}'.format(float(np.squeeze(clf_250.predict_proba(predict)[:,1], axis=0)))

		res_prob_200 = clf_200.predict_proba(predict)
		res_prob_200_1 = '{:.5f}'.format(float(np.squeeze(clf_200.predict_proba(predict)[:,0], axis=0)))
		res_prob_200_2 = '{:.5f}'.format(float(np.squeeze(clf_200.predict_proba(predict)[:,1], axis=0)))

		res_prob_100 = clf_100.predict_proba(predict)
		res_prob_100_1 = '{:.5f}'.format(float(np.squeeze(clf_100.predict_proba(predict)[:,0], axis=0)))
		res_prob_100_2 = '{:.5f}'.format(float(np.squeeze(clf_100.predict_proba(predict)[:,1], axis=0)))

		if (res == 0):
			diagnose = "tidak berpotensi terkena serangan jantung."
		else:
			diagnose = "berpotensi terkena serangan jantung."

		if (res_250 == 0):
			diagnose2 = "tidak berpotensi terkena serangan jantung."
		else:
			diagnose2 = "berpotensi terkena serangan jantung."

		if (res_200 == 0):
			diagnose3 = "tidak berpotensi terkena serangan jantung."
		else:
			diagnose3 = "berpotensi terkena serangan jantung."

		if (res_100 == 0):
			diagnose4 = "tidak berpotensi terkena serangan jantung."
		else:
			diagnose4 = "berpotensi terkena serangan jantung."


		print(accuracy)
		display="block"

		return render_template('base.html', result = diagnose, prob=round(res_prob[0][0]*100, 2), prob1 = round(res_prob_250[0][0]*100, 2), 
			prob2 = round(res_prob_200[0][0]*100, 2), prob_0 = round(float(res_prob_1)*100, 2), prob_1 = round(float(res_prob_2)*100, 2),
			result2 = diagnose2, prob_0_2 = round(float(res_prob_250_1)*100, 2), prob_1_2 = round(float(res_prob_250_2)*100, 2), 
			result3 = diagnose3, prob_0_3 = round(float(res_prob_200_1)*100, 2), prob_1_3 = round(float(res_prob_200_2)*100,2), result4 = diagnose4, prob_0_4 = res_prob_100_1, prob_1_4 = res_prob_100_2, 
			acc1 = accuracy, acc2 = accuracy_250, acc3 = accuracy_200, data_sample=enumerate(data_sample), display=display )

	else:	
		
		display = "none"
		

		return render_template("base.html", acc1 = accuracy, acc2 = accuracy_250, acc3 = accuracy_200, data_sample=enumerate(data_sample), display=display)

if __name__ == "__main__":
	app.run(debug=True)