import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
import csv
from sklearn.model_selection import cross_val_score
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from naive_bayes import naive_bayes
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from simple_ann import artificial_neural_net
import matplotlib.pyplot as plt

#Introduction:
#In this research we are trying to predict the type of a forest given some
#characteristics of a forest. Because sometimes is important to know which
#attributes are the most important to classify an instance we are focusing our
#experiment on improving model that can output the attribute importance
#by feature selection procedures. We compare the results against an artificial
#neural network problem.

#This is a supervised learning project. We have to create an statistical model
#that maps a set of features into a label that we want to predict.

#Load the datasets. One dataset to training our classifiers and other dataset
#where we are going to use our ML model to predict
#We remove the first column as it does not provide extra information
with open("train.csv",'r') as forest_data:
    csv_reader=csv.DictReader(forest_data)
    attribute_names=csv_reader.fieldnames[1:]

forest_set=np.genfromtxt("train.csv", dtype=None, delimiter=',', skip_header=1)[:,1:]

#create a dictionary to store the result
results_dictionary={}

#Get information about our dataset
instances_dataset=len(forest_set)
attributes_dataset=forest_set.shape[1]-1

#Define the attributes and the labels we are predicting
attributes=forest_set[:,:attributes_dataset]
labels=forest_set[:,attributes_dataset]

#Get the number of unique labels
number_unique_labels=len(np.unique(labels))

#Split our dataset into a training and testing set
attributes_training, attributes_testing, labels_training, labels_testing=\
train_test_split(attributes,labels, test_size=0.20)

#Get the number of attributes
number_attributes=attributes_training.shape[1]

#Initial classification task
#Decision Tree Classifier
#Train the Decision tree
decision_tree=tree.DecisionTreeClassifier()
decision_tree.fit(attributes_training, labels_training)

#Evaluate the tree
tree_performance=\
cross_val_score(decision_tree, attributes_testing, labels_testing, cv=10).mean()

#store results
results_dictionary['decision_tree_accuracy']=tree_performance

#Naive bayes can take a fair amount of time to be implemented. Hence I decide
#to create a variable to decide whether or not. The implementation of the
#naive bayes was cred by me.

#In my implementation of naive we have to indicate the attributes we want to
#be consider as numerical or categorical. We create this matrix to do so.
numerical_indication_matrix=[1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\
,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

naive_bayes_activation= False
#naive bayes implementation
if naive_bayes_activation == True:
    naive_bayes=naive_bayes()

    #classify
    classification_matrix_naive_bayes=\
    naive_bayes.naive_bayes_classifier(\
    attributes_training,labels_training,attributes_testing,numerical_indication_matrix)
    np.savetxt('naive_bays.csv',classification_matrix_naive_bayes, delimiter=',')
    nb_accuracy=accuracy_score(labels_testing,classification_matrix_naive_bayes)
    results_dictionary['naive_bayes_accuracy']=nb_accuracy

#Random Forest Classifier
random_forest=RandomForestClassifier()
random_forest.fit(attributes_training, labels_training)
random_forest_performance=\
cross_val_score(random_forest, attributes_testing, labels_testing, cv=10).mean()
#store results
results_dictionary['random_forest_accuracy']=random_forest_performance

#Artificial Neural Networks
ann_performance=artificial_neural_net(number_attributes,number_unique_labels\
,attributes_training,labels_training, attributes_testing, labels_testing,\
attributes_testing, accuracy=True)

results_dictionary['ann_accuracy']=ann_performance

#Feature Selection Process

#First we do a correlation analysis and delete those variables with high correlation
#These variables transform the same kind of information and might have a negative
#impact on the creation of the mapping function.
#We remove the attributes with a correlation higher than 90%
correlation_matrix=pd.DataFrame(attributes).corr()

#Correlation matrix to csv
correlation_matrix.to_csv('correlation_matrix.csv')

#Visual analysis of the correlation between attributes
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
cax = ax.matshow(correlation_matrix, vmin=-1, vmax=1)
fig.colorbar(cax)
plt.title('Correlation between attributes')
plt.ylabel('attribute')
plt.xlabel('attribute')
#save figure we created
plt.savefig('correlation_matrix.png')
plt.clf()
#There are not highly correlated features (correlation >90%)


#Feature selection process
#We are using the recursive_feature_elimination_crossvalidation. We need a
#model to use as an estimator to calculate the feature importance. We use rf
classifier_estimator=RandomForestClassifier()

#recursive_feature_elimination_crossvalidation implementation on skcikit
recursive_feature_elimination_crossvalidation=\
RFECV(classifier_estimator, step=1, cv=10, scoring='accuracy').fit\
(attributes_training,labels_training)

#Number of features after the feature selection process
number_features_after_feature_selection=\
recursive_feature_elimination_crossvalidation.n_features_

#Boolean mask that indicates which attributes we keep after the feature selection
boolean_filter=recursive_feature_elimination_crossvalidation.support_

#Try the classifiers in the new dataset
#Create the datasets
feature_selection_attributes_training=attributes_training[:,boolean_filter]
feature_selection_attributes_testing=attributes_testing[:,boolean_filter]

#Decision tree implementation implementation in new dataset
dt_feature_selection_rfecv=tree.DecisionTreeClassifier()
dt_feature_selection_rfecv.fit(feature_selection_attributes_training,labels_training)
dt_rfecv_performance=cross_val_score(dt_feature_selection_rfecv,\
feature_selection_attributes_testing,labels_testing, cv=10).mean()
results_dictionary['Decision_tree_rfecv_accuracy']=dt_rfecv_performance

#Random Forest implementation implementation in new dataset
rf_feature_selection_rfecv=RandomForestClassifier()
rf_feature_selection_rfecv.fit(feature_selection_attributes_training,labels_training)
rf_rfecv_performance=cross_val_score(rf_feature_selection_rfecv,\
feature_selection_attributes_testing,labels_testing, cv=10).mean()
results_dictionary['Random_forest_rfecv_accuracy']=rf_rfecv_performance

#Artificial Neural Networks implementation in new dataset
ann_performance_selection=artificial_neural_net(number_features_after_feature_selection,\
number_unique_labels, feature_selection_attributes_training, labels_training,\
feature_selection_attributes_testing, labels_testing,\
feature_selection_attributes_testing, accuracy=True)

results_dictionary['ann_rfecv_accuracy']=ann_performance_selection

#naive_bayes implementation in new dataset
naive_bayes_activation_rfecv= False
#naive bayes implementation
if naive_bayes_activation_rfecv == True:
    naive_bayesr_rfecv=naive_bayes()
    numerical_indication_rfecv=np.array(numerical_indication_matrix)[boolean_filter]

    #classify
    classification_matrix_naive_bayes=\
    naive_bayes_rfecv.naive_bayes_classifier(\
    feature_selection_attributes_training,labels_training,\
    feature_selection_attributes_testing,numerical_indication_matrix)
    naive_rfecv_perf=accuracy_score(labels_testing,classification_matrix_naive_bayes)
    results_dictionary['naive_bayes_rfecv_accuracy']=ann_performance_selection

#Transform result into csv
with open('results.csv', 'wb') as results:
    results_write= csv.DictWriter(results, results_dictionary.keys())
    results_write.writeheader()
    results_write.writerow(results_dictionary)

#Predict the unlabelled data
forest_test=np.genfromtxt("test.csv", dtype=None ,delimiter=',', skip_header=1)[:,1:]
print forest_test.shape

#Apply the feature selection filter to the data
forest_test_feature_selection=forest_test[:,boolean_filter]
print forest_test_feature_selection.shape

#predict the labels
ann_prediction=artificial_neural_net(number_features_after_feature_selection,\
number_unique_labels, feature_selection_attributes_training, labels_training,\
feature_selection_attributes_testing, labels_testing,\
forest_test_feature_selection, accuracy=False, prediction=False)

print ann_prediction
np.savetxt("prediction.csv", ann_prediction, delimiter=",")
