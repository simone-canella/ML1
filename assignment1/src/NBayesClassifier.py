import numpy as np


class Nbayes:
    def __init__(self):
        self.trained = False
        self.class_frequency = {}
        self.class_priors = {} 
        self.likelihoods = {}

        self.DEBUG = False

    def fit(self, x_train, y_train):
        
        unique_classes = y_train.unique() # unique classes

        self.class_frequency = y_train.value_counts().to_dict() #count how many times the class appears in the y_train
        
        if self.DEBUG == True:
            print("class frequency: ",self.class_frequency.items())

        for key in unique_classes:
            self.class_priors[key] = self.class_frequency[key] / len(y_train) #compute the probability of occuring a class

        if self.DEBUG == True:
            print("class priors: " ,self.class_priors.items())


        for key in unique_classes:
            subset = x_train[y_train == key] #create a a subset of x_train where appear corresponding value of y_train[key]
            self.likelihoods[key] = {}
            
            for feature in subset.columns:
                count = subset[feature].value_counts() #count how many value occurs in the subset
                conditional_probability = (count / len(subset)).to_dict() #calculate conditional probability (number of i-value / number of total value)

                self.likelihoods[key][feature] = conditional_probability #populate the likelihoods
            
            if self.DEBUG == True:
                print("class likelihoods", self.likelihoods)

        '''    
        for key in unique_classes:  
            subset = x_train[y_train == key] #create a a subset of x_train where appear corresponding value of y_train[key]
            conditional_probability = {}
            
            for feature in range(0, len(subset.columns)):
                count = subset.iloc[:, feature].value_counts() #count how many value occurs in the subset
                conditional_probability[feature] = count / len(subset.iloc[:, feature]) #calculate conditional probability (number of i-value / number of total value)
                
            #print(conditional_probability)
            
        '''             
             
        self.trained = True
    
    def predict(self, x_test):
        if not self.trained:
            raise ValueError
        
        y_predict = [] #list of predictions based on the higest probability between yes or no (max(P(class_priors) * P(class_likelihoods)))

        for yes_no in self.class_priors:
            # print(f"{yes_no}")
        
            for _row, feature in x_test.iterrows():
                # print(f"{feature.Outlook} / {feature.Temperature}")
                # rmc = feature.Temperature

                for feature_dict in feature.to_dict():
                    # print(f"{feature_dict}")
                    zzz = getattr(feature, feature_dict)
                    print(f"{zzz} {self.likelihoods[yes_no][feature_dict][zzz]}")

                    # for value in feature_dict:


                    # print(f"{self.likelihoods[yes_no][feature_dict]}")


        '''
        for _row, value in x_test.iterrows():
            #print(f"row: {row}, Outlook: {value.Outlook}, Temperature: {value.Temperature}, Humidity: {value.Humidity}, Windy: {value.Windy}")
            print(f"Outlook: {value.Outlook}, Temperature: {value.Temperature}, Humidity: {value.Humidity}, Windy: {value.Windy}")
        '''

        #for key in self.class_priors:
         
            #conditional_probability = self.class_priors[key]

            
            #conditional_probability *= self.likelihoods[key][row]
        #print(self.class_priors)
        #print(self.likelihoods['yes'])

        '''
        for key in self.class_priors:
            print("class: ", key)

            for feature in self.likelihoods[key]:
                print("feature for class: ", feature)
                

                for value in self.likelihoods[key][feature]:
                    print("value for each feature: ", value)

                    result = self.class_priors[key] * self.likelihoods[key][feature][value]

                    print(result)
        '''         

        '''
        for each class c
            for each variable x
                for each possible value v for variable x
                    the number of instances of class c that have variable x == value v...
                    ...divided by the number of instances of class c
        '''
        

           

    def test(self, X_test, y_test):
        if not self.trained:
            raise ValueError
        y_predict = self.predict(X_test)
        # return accuracy, fraction of correct predictions:
        return (np.array(y_test) == np.array(y_predict)).sum()/len(y_test)
    