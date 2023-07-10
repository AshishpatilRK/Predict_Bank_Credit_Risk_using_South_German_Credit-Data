# Basic Import
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,accuracy_score, roc_auc_score,classification_report,ConfusionMatrixDisplay
from Predict_Bank_Credit_Risk.exceptions.exception import CustomException
from Predict_Bank_Credit_Risk.logging.logger import logging

from Predict_Bank_Credit_Risk.utils.util import save_object
from Predict_Bank_Credit_Risk.utils.util import evaluate_model

from dataclasses import dataclass
import sys
import os

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
            "XGBClassifier": XGBClassifier(),
            "Random Forest Classifier": RandomForestClassifier(),
            # "Logistic Regression": LogisticRegression(), 
            "SVC Classifier": SVC()
         }
            param={
                "XGBClassifier": {
            'learning_rate': [0.5, 0.1, 0.01, 0.001],
            'max_depth': [3, 5, 10, 20],
            'n_estimators': [10, 50, 100, 200]
        },
                "Random Forest Classifier":{
            "class_weight":["balanced"],
            "n_estimators":[10, 50, 100, 130],
            'max_depth': [2, 4, 1],
            "max_features": ['auto', 'log2'],
        },
        #         "Logistic Regression":{
        #     "penalty":["l1", "l2", "elasticnet", None],
        #     "class_weight":["balanced"],
        #     'C': [0.001, 0.01, 0.1, 1, 10, 100],
        #     "solver":["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]
        # },
                "SVC Classifier": {'C': [0.1, 1, 10, 100, 1000], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']},
    
    
}
            
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models,param)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )
          

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)