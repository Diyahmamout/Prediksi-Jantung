from flask import Flask, render_template
from sklearn.datasets import load_wine
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import pandas as pd
import csv

app = Flask(__name__)

@app.route('/')
def index():
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

	knn1 = KNeighborsClassifier(n_neighbors = 3)
	knn2 = KNeighborsClassifier(n_neighbors = 5)
	knn3 = KNeighborsClassifier(n_neighbors = 7)
	knn4 = KNeighborsClassifier(n_neighbors = 9)
	knn5 = KNeighborsClassifier(n_neighbors = 11)

	knn1.fit(trainX, trainY)
	knn2.fit(trainX, trainY)
	knn3.fit(trainX, trainY)
	knn4.fit(trainX, trainY)
	knn5.fit(trainX, trainY)

	score1 =  accuracy_score(testY, knn1.predict(testX))
	score2 =  accuracy_score(testY, knn2.predict(testX))
	score3 =  accuracy_score(testY, knn3.predict(testX))
	score4 =  accuracy_score(testY, knn4.predict(testX))
	score5 =  accuracy_score(testY, knn5.predict(testX))

	print("KNN with K = 3  : " + str(round((score1*100), 2)) + '%')
	print("KNN with K = 5  : " + str(round((score2*100), 2)) + '%')
	print("KNN with K = 7  : " + str(round((score3*100), 2)) + '%')
	print("KNN with K = 9  : " + str(round((score4*100), 2)) + '%')
	print("KNN with K = 11 : " + str(round((score5*100), 2)) + '%')

	return render_template("base.html", data_sample=enumerate(data_sample))

if __name__ == "__main__":
	app.run(debug=True)