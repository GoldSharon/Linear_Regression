import numpy as np
import matplotlib.pyplot as plt

class Linear_Regression:
    
    def __init__(self,learning_rate=0.01,number_iter=1000):
         
         self.learning_rate = learning_rate
         self.number_iter = number_iter
         self.bias = 0
         self.weight = None
         self.rows = 0
         self.cols = 0
         
    def fit(self,X,y):
        
        self.rows = X.shape[0]
        self.cols = X.shape[1]
        
        self.weight = np.zeros((self.cols,1))
        
        for i in range(self.number_iter):
            
            self.update_parameter(X,y)
            
        print("The model is learnt Successfully 😊")
        mse = np.mean(((y-self.predict(X))**2))
        print("The Mean Squared error is "+str(mse))
        
        if(self.cols==1):
            self.plot(X,y)
            
        
        
    
    def predict(self,X):
        
        return X @ (self.weight) + self.bias
        
    def update_parameter(self,X,y):
        
        y_pred = self.predict(X)
        
        dw = (-2 / self.rows) * (X.T @ (y - y_pred))  # No sum, just the error vector
        db = (-2 / self.rows) * np.sum(y - y_pred)    # Sum of errors for bias update
        
        self.weight -= self.learning_rate*dw
        
        self.bias -=self.learning_rate*db
        
        
   
        
    def plot(self, X, y):
        y_pred = self.predict(X)
        
        plt.scatter(X, y, color='g', label="Actual")
        plt.plot(X, y_pred, 'r', label="Predicted")
        plt.legend()
        plt.show()
        
        
        
        
                