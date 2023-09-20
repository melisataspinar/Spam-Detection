import csv
import warnings
import os
import numpy as np

# TWO HELPER FUNCTIONS TO READ DATASETS AND ANALYZE PREDICTIONS ******************************

# Function that prints out evaluation metrics given a prediction and a ground_truth
def analyze_prediction( prediction, ground_truth ):
    
    TP = np.sum( prediction * ground_truth ).item()
    FP = np.sum( prediction ).item() - TP
    FN = np.sum( ground_truth ).item() - TP
    TN = np.sum( (1-ground_truth) ).item() - FP
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    wrong = FP + FN
    
    print("Confusion Matrix:\n")
    print( "\t\tPrediction")
    print( "        ----------------------------" )
    print("\t\t| Spam\t|   Normal")  
    print( "        ----------------------------" )
    print( "Truth\tSpam    | " + str(TP) +"\t|   " + str(FN) )
    print( "        ----------------------------" )
    print( "\tNormal  | " + str(FP) + "\t|   " + str(TN) )
    print( "        ----------------------------\n" )
    print( "Accuracy: %.2f" % accuracy ) 
    print( "Number of Wrong Predictions: %d" % wrong )

# function that gets the datasets from their directory and returns them as np arrays
def get_datasets():
    # getting directory paths
    root_path = os.path.dirname(os.path.realpath(__file__)).replace('\\','/')[:-5]
    data_path = root_path + "/User/CS464_HW1_datasets"

    # get the data in the form of np arrays
    x_train = np.array( list( csv.reader( open( data_path + "/x_train.csv" ), delimiter = ',') ), dtype="uint32")
    x_test = np.array( list( csv.reader( open( data_path + "/x_test.csv" ), delimiter = ',') ), dtype="uint32")
    y_train = np.array( list( csv.reader( open( data_path + "/y_train.csv" ), delimiter = ',') ), dtype="uint32").flatten()
    y_test = np.array( list( csv.reader( open( data_path + "/y_test.csv" ), delimiter = ',') ), dtype="uint32").flatten()
    
    return x_train, x_test, y_train, y_test


# THREE CLASSES FOR CLASSIFIERS *************************************************************
    
# Class for multinomial naive bayes classifiers
class MNB_Classifier():
    
    def train(self, X, Y):
        X_normal = X[ Y==0 ]
        normal_likelihoods = ( np.sum( X_normal, axis=0 ) ) / np.sum( X_normal )
        self.log_likelihoods_normal = np.log( normal_likelihoods )
        
        X_spam = X[ Y==1 ]
        spam_likelihoods = ( np.sum(X_spam, axis=0) ) / np.sum(X_spam) 
        self.log_likelihoods_spam = np.log( spam_likelihoods )
        
        normal_mails_ratio = (Y.size - np.sum(Y)) / Y.size
        self.log_normal_to_all_ratio = np.log(normal_mails_ratio)
        self.log_spam_to_all_ratio = np.log( 1 - normal_mails_ratio )
    
    def predict( self, X ):
        # to define 0log(0) = 0
        temp = X * self.log_likelihoods_normal
        nans = np.isnan(temp)
        temp[ nans ] = 0
        P_X_given_normal = np.sum( temp, axis = 1 ) + self.log_normal_to_all_ratio
        
        temp = X * self.log_likelihoods_spam
        nans = np.isnan(temp)
        temp[ nans ] = 0
        P_X_given_spam = np.sum( temp, axis=1 ) + self.log_spam_to_all_ratio
        
        prediction = np.zeros( X.shape[0] )
        prediction[ P_X_given_spam > P_X_given_normal ] = 1
        return prediction
    
# Class for multinomial naive bayes classifiers with dirichlet normalization
class MNB_Classifier_Dirichlet():
    
    def __init__( self, alpha = 0 ):
        self.alpha = alpha
    
    def train(self, X, Y):
        X_normal = X[ Y==0 ]
        normal_likelihoods = ( np.sum( X_normal, axis=0 ) + self.alpha ) / ( np.sum( X_normal )+ Y.size * self.alpha)
        self.log_likelihoods_normal = np.log( normal_likelihoods )
        
        X_spam = X[ Y==1 ]
        spam_likelihoods = ( np.sum(X_spam, axis=0) + self.alpha ) / ( np.sum(X_spam) + Y.size * self.alpha)
        self.log_likelihoods_spam = np.log( spam_likelihoods )
        
        normal_mails_ratio = (Y.size - np.sum(Y)) / Y.size
        self.log_normal_to_all_ratio = np.log(normal_mails_ratio)
        self.log_spam_to_all_ratio = np.log( 1 - normal_mails_ratio )
    
    def predict( self, X ):
        # to define 0log(0) = 0
        temp = X * self.log_likelihoods_normal
        nans = np.isnan(temp)
        temp[ nans ] = 0
        P_X_given_normal = np.sum( temp, axis = 1 ) + self.log_normal_to_all_ratio
        
        temp = X * self.log_likelihoods_spam
        nans = np.isnan(temp)
        temp[ nans ] = 0
        P_X_given_spam = np.sum( temp, axis=1 ) + self.log_spam_to_all_ratio
        
        prediction = np.zeros( X.shape[0] )
        prediction[ P_X_given_spam > P_X_given_normal ] = 1
        return prediction
    
# Class for Bernouilli naive bayes classifiers    
class BNB_Classifier():
     
    def train(self, X, Y):
        normal_mails_ratio = ( Y.size - np.sum(Y) ) / Y.size
        spam_mails_ratio = 1 - normal_mails_ratio
                
        self.log_normal_to_all_ratio = np.log( normal_mails_ratio )
        self.log_spam_to_all_ratio = np.log( spam_mails_ratio )
        
        X_normal = X[ Y == 0 ]
        X_spam = X[ Y == 1 ]
        
        self.likelihoods_normal = np.sum( X_normal, axis=0 ) / ( Y.size - np.sum(Y) )
        self.likelihoods_spam = np.sum( X_spam, axis=0 ) / np.sum(Y)
        
    def predict(self, X):
        X[X != 0] = 1
        
        temp = np.log( (1 - X) * (1 - self.likelihoods_normal ) + X * self.likelihoods_normal )
        nans = np.isnan(temp)
        temp[ nans ] = 0
        P_X_given_normal = np.sum( temp, axis=1 ) + self.log_normal_to_all_ratio
        
        temp = np.log( (1 - X) * (1 - self.likelihoods_spam ) + X * self.likelihoods_spam )
        nans = np.isnan(temp)
        temp[ nans ] = 0
        P_X_given_spam = np.sum( temp , axis=1 ) + self.log_spam_to_all_ratio
        
        prediction = np.zeros(X.shape[0])
        prediction[ P_X_given_spam > P_X_given_normal ] = 1
        return prediction


# MAIN *******************************************************************************************

x_train, x_test, y_train, y_test = get_datasets()

# Checking out the dataset
mails = y_train.size
spam_mails = np.sum( y_train )
normal_mails = mails - spam_mails
print( "**************************************************************************************************************************************" )
print( "Checking out the dataset:" )
print("Number of normal mails: " + str( normal_mails ) )
print("Number of spam mails: " + str( spam_mails ) )
print( "Ratio of spam mails: %.2f" % ( spam_mails/mails ) ) 

# Multinomial Naive Bayes
print( "**************************************************************************************************************************************" )
print( "Training a Multinomial Naive Bayes model on the training set, and evaluating it on the test set given:" )
classifier3_2 = MNB_Classifier()
classifier3_2.train( x_train, y_train )
analyze_prediction( classifier3_2.predict( x_test ), y_test )

# Multinomial Naive Bayes with Dirichlet
print( "**************************************************************************************************************************************" )
print( "Extending the classifier so that it can compute an MAP estimate of θ parameters using a fair Dirichlet prior (additive smoothing):" )
classifier3_3 = MNB_Classifier_Dirichlet( alpha = 1 )
classifier3_3.train( x_train, y_train )
analyze_prediction( classifier3_3.predict( x_test ) , y_test )

# Bernoulli Naive Bayes
print( "**************************************************************************************************************************************" )
print( "Training a Bernoulli Naive Bayes model on the training set, and evaluating it on the test set given:" )
classifier3_4 = BNB_Classifier()
classifier3_4.train( x_train, y_train )
analyze_prediction( classifier3_4.predict( x_test ), y_test )
    
    
