import sys
import math
import sys
import os
import csv
import numpy as np

class naive_bayes():
    ##########   SET OF FUNTIONS FOR THE NAIVE BAYES  CLASSIFIER    ############
    #The naive bayes classifies scans the training set to find the posterior
    #probabilities of an class given an instance. Then, the classifier, assigns
    #to the instance the class with highest posterior probabilities.


    ####### FUNCTION TO CALCULATE THE PRIOR PROBABILITIES OF EACH CLASS ########
    #The following function calculates the prior probabilities of a class/label.
    #We calculate the prior by counting the number of times a label appear and
    #dividing this number by the total number of instances.
    def calculation_prior_probabilities(self,labels):
        #first, find out the name of the unique labels and their frequencies
        unique_labels, frequency_labels_vector=\
         np.unique(labels,return_counts=True)

        #Change the format of the returned vector to perform divisions later.
        frequency_labels_vector.astype(float)

        #Calculate the number of instances 'N' to then find the probabilities
        #using thself.weight_matrix_first_classifiere frequencies of each labels
        number_labels=len(labels)

        #Calculate the number of unique labels
        number_unique_labels=len(unique_labels)

        #Divide the frequencies by the total number of labels to get prior.
        prior_probabilities_vector=frequency_labels_vector/float(number_labels)

        #Create a matrix to display the results properly and keep track
        #of the labels. The matrix dimensions are: (2 x number of unique labels)
        #In each column we store the label identifier/name  and its prior.
        #The firs row contains the name of the labels and the second the priors
        matrix_of_priors=np.concatenate((
        unique_labels.reshape((1,number_unique_labels)),
        prior_probabilities_vector.reshape((1,number_unique_labels))
        ), axis=0)

        #return the generated matrix
        return matrix_of_priors

    ######################        END OF THE FUNCTION        ###################


    ########        FUNCTION TO CALCULATE THE LIKEHOOD PROBABILITIES    ########
    ########    OF A GIVEN VALUE OF AN ATTRIBUTE FOR A SINGLE ATTRIBUTE    #####
    #The function returns the likehood of an attribute value. We will use this
    #Function in combination with other fucntion to calculate the likehood of
    #all the attribute values in an instance.
    def likehood_values_in_single_attribute(self,attributes,labels,\
    attribute_index,value_look_for, m_estimate= True):

        #Make sure the inputs are in an array format.
        labels=np.array(labels)

        #Identify basic characteristics of the inputs.
        number_instances = attributes.shape[0]
        number_attributes = attributes.shape[1]

        #Find the values/names of of the unique labels and their frequencies.
        label_values, frequency_labels = np.unique(labels, return_counts=True)

        #Retrieve attribute column where the value we are querying belongs.
        list_attribute_of_analysis=attributes[:,attribute_index]

        #Find unique attribute values in the dataset and their frequencies.
        #'attribute_domain' stores the unique values in the attribute domain.
        #'attribute_domain_counter'registers the frequency of each unique value.
        attribute_domain , attribute_domain_counter= \
        np.unique(list_attribute_of_analysis, return_counts=True)

        #We create a numpy array to handle the values in a nicer way.
        #Dimensions: (number of unique attribute values x float)
        attribute_domain_matrix=\
        np.array((np.transpose(attribute_domain),\
        np.transpose(attribute_domain_counter)), dtype=float)

        #Sometimes there is not enought data and likehhod return 0 as value.
        #To solve this problem we can smooth probabilities with m-stimate
        #or laplace smoothing.The m-stimate requires the number of different
        #values that the attribute can take. Then using this we get
        #the 'prior estimate' of the attribute value. We will use this later
        k_number_values_attribute_take= float(len(attribute_domain))
        attribute_prior_estimate= float(float(1)/k_number_values_attribute_take)

        #Next step, we create a boolean index of the value we are looking for.
        #With the boolean index we will be able to acquire the labels assigned
        #to the value we are looking for.
        boolean_vector_attribute_domain =\
         [value_look_for == instance_attribute for instance_attribute in\
        list_attribute_of_analysis]

        #Transform the boolean list into arrays to handle it better.
        boolean_vector_attribute_domain=np.array(boolean_vector_attribute_domain)

        #Using the boolean index, generate an array with the labels that
        #corresponds to the attributes values we are analyzing.
        labels_of_observed_attribute=labels[boolean_vector_attribute_domain]

        #Now,we create a vector to store the number of times each label is
        #assigned to the value we want to get the posterior probability.
        #Dimentsion: (number of labels x 1)
        attribute_domain_class_counter_vector=\
        np.zeros((len(label_values),1), dtype=float)

        #We loop all values the labels can acquire and we count how many of
        #these labels are assigned to the value we want.
        for label_index in range(len(label_values)):
            #Get the label we want to count with the index
            label_analysis=label_values[label_index]

            #Obtain an array where the label are the same as the label we are
            #parsing
            vector_labels_equal_parsed_label=\
            np.where(labels_of_observed_attribute==label_analysis)[0]

            #Count the time that the specific label appears when the attribute
            #acquires the value we are looking.
            counter_label_in_attribute=len(vector_labels_equal_parsed_label)

            #Implementation of the m-estimate methodology smooths the likehoods
            #probabilities to avoid problems in the likehood multiplication.
            #Hence, we avoid problems when one of the likehood is 0.
            #Here we add to the counter an estimation probability multipliesd by
            #the number of attributes of the input
            if m_estimate == True:
                counter_label_in_attribute=\
                counter_label_in_attribute+\
                attribute_prior_estimate*number_attributes

            #Replace the value in the count matrix
            attribute_domain_class_counter_vector[label_index]=\
            counter_label_in_attribute
            
        #Get the likehood vector: We divide the number of times that the given
        #attribute value is labelled as a specif class by the class frequency.
        #Hence, we obtain the likehoods of the given value per every class.
        #If we are smothing with the m-stimate we have to add to the denominator
        #the number of attributes that each instance has.
        if m_estimate == True:
            frequency_labels=frequency_labels+number_attributes

        #Calculate the likehoods
        likehoods_attribute_per_class=\
        np.divide(np.transpose(attribute_domain_class_counter_vector),\
        frequency_labels)

        #To keep track of the labels, we create a matrix that indicates which
        #labels belongsto each likehood.
        likehoods_matrix=\
        np.concatenate((label_values.reshape((1,len(label_values)))\
        ,likehoods_attribute_per_class), axis=0)

        #return the matrix we have created
        return likehoods_matrix

    ######################        END OF THE FUNCTION     ######################

    def pdf(self, value_look, mean, std):
        value_look = float(value_look - mean) / std
        return math.exp(-value_look*value_look/2.0) / math.sqrt(2.0*math.pi) /std

    ######################        END OF THE FUNCTION     ######################


    def likehood_values_in_single_attribute_numerical(self,attributes,labels,\
    attribute_index,value_look_for):
        #Make sure the inputs are in an array format.
        labels=np.array(labels)

        #Identify basic characteristics of the inputs.
        number_instances = attributes.shape[0]
        number_attributes = attributes.shape[1]

        #Find the values/names of of the unique labels and their frequencies.
        label_values, frequency_labels = np.unique(labels, return_counts=True)

        #Retrieve attribute column where the value we are querying belongs.
        list_attribute_of_analysis=attributes[:,attribute_index]

        #Now,we create a vector to store the number of times each label is
        #assigned to the value we want to get the posterior probability.
        #Dimentsion: (number of labels x 1)
        attribute_likehood_to_class=\
        np.zeros((len(label_values),1), dtype=float)

        #We loop all values the labels can acquire and we count how many of
        #these labels are assigned to the value we want.
        for label_index in range(len(label_values)):
            #Get the label we want to count with the index
            label_analysis=label_values[label_index]

            boolean_vector_label_analysis=\
            np.array([label_analysis == instance_label for instance_label in\
            labels])

            #Obtain an array where the label are the same as the label we are
            #parsing
            vector_attributes_with_labels_equal_to_parsed_label=\
            list_attribute_of_analysis[boolean_vector_label_analysis]

            #Get the mean and the std of the intances labelled with the label
            #we are analysing
            mean_numeric_attribute_labels=\
            vector_attributes_with_labels_equal_to_parsed_label.mean()
            std_numeric_attribute_labels=\
            vector_attributes_with_labels_equal_to_parsed_label.std()

            #Based of the std and the mean we get the likehood of our numerical
            #value to the label assuming that is a gaussian distribution
            likehood_value=self.pdf(value_look_for, \
            mean_numeric_attribute_labels, std_numeric_attribute_labels)

            attribute_likehood_to_class[label_index]=likehood_value

        likehood_matrix=\
        np.concatenate((label_values.reshape((1,len(label_values)))\
        ,np.transpose(attribute_likehood_to_class)),axis=0)

        return likehood_matrix
    ######################        END OF THE FUNCTION     ######################


    ########        FUNCTION TO CALCULATE THE LIKEHOOD PROBABILITIES    ########
    ##############     OF A GIVEN INSTANCE FOR ALL THE ATTRIBUTES        #######
    #This functions uses the function 'likehood_values_in_single_attribute'
    #to return the likehoods of all the attributes values of a given instance.
    def find_likehood_probabilities_of_instance(self, instance, attributes,\
     labels, numerical_attribute_indication_matrix):
        #Make sure the inputs are in the right format
        instance=np.array(instance)
        attributes=np.array(attributes)

        #First, we get basic information about the inputs.
        number_attributes=instance.shape[0]
        number_instances=attributes.shape[0]
        label_values, frequency_labels = np.unique(labels, return_counts=True)
        number_labels=len(label_values)

        #Create a matrix to store the likehoods for every labels for every
        #attribute in the instance we want to classify.
        #Dimension:(dimension of the feature vector x number different labels)
        instance_likehood_matrix=\
        np.zeros((number_attributes,number_labels),dtype=float)

        #Loop all the attribute values in the instance we want to classify.
        for attribute_values_index in range(number_attributes):
            #Get the value of the attribute we are parsing
            attribute_value_analysis=instance[attribute_values_index]

            numeric_indication=\
            numerical_attribute_indication_matrix[attribute_values_index]

            if numeric_indication ==1:
                likehood_attribute_value_analysis=\
                self.likehood_values_in_single_attribute_numerical(attributes,labels,\
                attribute_values_index,attribute_value_analysis)[1]
            else:
                #Get the likehood of the attribute for each classs
                likehood_attribute_value_analysis=\
                self.likehood_values_in_single_attribute(attributes,labels,\
                attribute_values_index,attribute_value_analysis)[1]

            #Put the resulting vector into the matrix that stores the likehoods.
            instance_likehood_matrix[attribute_values_index]=\
            likehood_attribute_value_analysis

        return instance_likehood_matrix
    ######################        END OF THE FUNCTION     ######################

    ####  FUNCTION TO CALCULATE THE POSTERIOR PROBABILITIES FOR EVERY CLASS ####
    #########     FOR A GIVEN INSTANCE FOR ALL THE ATTRIBUTES        ###########
    #The function calculates the posterior probabilities of a given instance
    #for every class. Following the naive bayes theory we multiply all the
    #likehoods of all the single attributes values of the instance to get the
    #likehood of the instance ssuming that all attributes are independent.
    #The when we will get the instance likehood for that class that we multiply
    #by the  class prior to get the posterior of the class given the instance.
    def posterior_each_class_claculation (self,likehood_matrix, prior_vector):
        #Multiply all the elements of a columns given a matrix with all the
        #likenhoods for every class for every attribute of the instance.
        #Where the likehoods are in the columns
        product_likehood=np.prod(likehood_matrix , axis=0)

        #Dot Multiply the vector with the prior probabilities with the likehood
        #per class vector to get the posterior vector.
        posterior_vector_each_class=np.multiply(product_likehood,prior_vector[1])

        #Create a matrix with the name of the labels and the posteriors.
        posterior_matrix_each_class=np.concatenate\
        ((prior_vector[0].reshape((1,len(prior_vector[0]))),\
        posterior_vector_each_class.reshape((1,len(posterior_vector_each_class)))),axis=0)

        #Return the new matrix
        return posterior_matrix_each_class
    ######################        END OF THE FUNCTION     ######################


    ##########  FUNCTION TO PERFORM THE MAXIMUM A POSTERIORI ESTIMATE ##########
    #The input is a matrix with all the posterior probabilities of an instance.
    #And an indication of the label of every posterior.
    def maximum_aposteriori_estimate(self,posterior_matrix_each_class):
        #Get the index of the maximum posterior probability
        index_maximum_posterior=np.argmax(posterior_matrix_each_class[1])

        #Using the index obtain the class labels
        class_maximum_posterior=\
        int(posterior_matrix_each_class[0][index_maximum_posterior])

        ##return the class
        return class_maximum_posterior
    ######################        END OF THE FUNCTION     ######################

    ##########  FUNCTION TO PERFORM THE NAIVE BAYES CLASSIFIER  ################
    #######        COMPILATION OF NAIVE BAYES FUNCTIONS O CLASSIFY       #######
    #Here we compile all the previously created function to simplify the
    #classification of multiple intances.
    def naive_bayes_classifier_instance(self,attributes,labels, instance_matrix,\
    numerical_indications):
        #Make sure the input is in the required format
        instance_matrix=np.array(instance_matrix).reshape((1,len(instance_matrix)))

        #Basic information of the input we are using
        number_instances_to_classifify=instance_matrix.shape[0]

        #Create a vector to store the classifications.
        classfication_vector=\
        np.zeros((1,number_instances_to_classifify),dtype=int)

        #Loop through all the instances that we want to predict.
        for instance_to_predict_index in range(number_instances_to_classifify):

            #Get the instance we want to predict from the instance meatrix
            instance_to_predict=instance_matrix[instance_to_predict_index,:]

            #Calculate the label priors
            priors=self.calculation_prior_probabilities(labels)

            #Know whether the attribute is numerical or categorical based on
            #The array we created.
            numerical_indicator=numerical_indications[instance_to_predict_index]

            #Get the likeehoods of the instance per each class
            likehood_instance_each_class=\
            self.find_likehood_probabilities_of_instance(instance_to_predict,\
            attributes,labels,numerical_indications)

            #Calculate the posteriors of each class for the given instance.
            posterior_each_class=\
            self.posterior_each_class_claculation(likehood_instance_each_class,priors)

            #Classify by assigning the label with the highest posterior.
            classification_instance=\
            self.maximum_aposteriori_estimate(posterior_each_class)

            #Put the result of the prediciton into the vector that stores the
            #predictions.
            classfication_vector[0,instance_to_predict_index]=\
            classification_instance

        #return the vector with the predictions
        return classfication_vector

    ######################        END OF THE FUNCTION     ######################

    def naive_bayes_classifier(self,attributes,labels, instance_matrix,\
    numerical_indications):
        #Create array with the same number of instances as the matrix we are
        #classifying to store the results.
        classification_matrix=np.zeros((len(instance_matrix),1), dtype=int)

        #Use naibe_bayes classifier for each intance of the dataset
        for instance_index in range(len(instance_matrix)):
            #get the instance we classify
            instance_to_classify=np.array(instance_matrix[instance_index,:])

            #classify the instance
            classification=self.naive_bayes_classifier_instance(attributes,labels,\
            instance_to_classify,numerical_indications)

            #Place the classification into the result matrix
            classification_matrix[instance_index,0]=classification

        return classification_matrix
    ###############        END OF THE NAIVE BAYES FUNCTIONS     #################
