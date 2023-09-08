import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (mean_squared_error, 
                             median_absolute_error, 
                             mean_absolute_percentage_error,
                             r2_score)

TARGET_TO_ANALYZE = 'target'

def MDAPE(actual, predicted, sample_weight=None): # Median Absolute Percentage Error
    return np.median((np.abs(np.subtract(actual, predicted)/ actual))) * 1

def f_relative_error(y, y_pred, mode: int=0):
    error = []
    
    for i, j in zip(y.values, y_pred.values):
        error.append(float((j-i)/i))
    if mode == 0:
        return np.mean(error)+2*np.std(error, ddof=1)
    else: 
        return error

class AutoML:
    def __init__(self, 
                 model,
                 X_test_scaled: pd.DataFrame,
                 target: TARGET_TO_ANALYZE,

                 y_test: pd.DataFrame=None,
                 X_train_scaled: pd.DataFrame=None, 
                 y_train: pd.DataFrame=None,
                 ):
        self.model = model
        self.X_test_scaled = X_test_scaled
        self.X_train_scaled = X_train_scaled
        self.y_test = y_test
        self.y_train = y_train
        self.target = target
        self.y_test_pred = None,
        self.y_train_pred = None
        
    def make_inference(self) -> pd.DataFrame:
        y_test_pred = self.model.predict(self.X_test_scaled)
        self.y_test_pred = pd.DataFrame(data=y_test_pred, 
                                   index=self.X_test_scaled.index, 
                                   columns=[self.target])
        if self.X_train_scaled is not None:
            y_train_pred = self.model.predict(self.X_train_scaled)
            self.y_train_pred = pd.DataFrame(data=y_train_pred, 
                                        index=self.X_train_scaled.index, 
                                        columns=[self.target])
            return (self.y_test_pred, self.y_train_pred)
        return self.y_test_pred
    
    def calculate_metrics(self) -> pd.DataFrame:
        assert self.y_test_pred is not None, "Модель не обучена!"
        MAE_train, MAE_test = median_absolute_error(self.y_train,  self.y_train_pred), \
                              median_absolute_error(self.y_test, self.y_test_pred) # mae
        RMSE_train, RMSE_test = mean_squared_error(self.y_train, self.y_train_pred, squared=False), \
                                mean_squared_error(self.y_test, self.y_test_pred,  squared=False) #rmse
        MAPE_train, MAPE_test = mean_absolute_percentage_error(self.y_train, self.y_train_pred), \
                                mean_absolute_percentage_error(self.y_test, self.y_test_pred) # MAPE
        MDAPE_train, MDAPE_test = MDAPE(self.y_train, self.y_train_pred), \
                                  MDAPE(self.y_test, self.y_test_pred)
        R2_train, R2_test = r2_score(self.y_train, self.y_train_pred), \
                            r2_score(self.y_test, self.y_test_pred)
        
        relative_error_train, relative_error_test = f_relative_error(self.y_train, self.y_train_pred), \
                                                    f_relative_error(self.y_test, self.y_test_pred)
        
        return pd.DataFrame({'MAE':[MAE_train, MAE_test], 'RMSE':[RMSE_train,RMSE_test], 
                             'MAPE':[MAPE_train, MAPE_test], 'MDAPE':[MDAPE_train, MDAPE_test], 
                             'R2': [R2_train, R2_test], 
                             'Relative_error':[relative_error_train, relative_error_test]}, 
                             index=['Train','Test'])      

    def plot_results(self):
        assert self.y_test_pred is not None, "Модель не обучена!"
        plt.rcParams.update({'font.size': 10})
        fig, ax = plt.subplots(1,1, figsize=(14,5))
        self.y_test.plot(marker='o',markersize=0.5, linestyle = 'None', ax=ax, label='y_test', c = 'black')
        self.y_train.plot(marker='o',markersize=0.5, linestyle = 'None', ax=ax, label='y_train', c = 'black')
        self.y_test_pred.plot(marker='o',markersize=0.5,linestyle = 'None',c='r',ax=ax,label='y_test_pred')
        self.y_train_pred.plot(marker='o',markersize=0.5,linestyle = 'None',c='g',ax=ax,label='y_train_pred')
        plt.xticks(rotation=25)  
        plt.ylabel(self.target)
        plt.legend() 
        plt.grid()
        return fig         
        #plt.show()  
