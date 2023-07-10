import os
import sys
from Predict_Bank_Credit_Risk.logging.logger import logging
from Predict_Bank_Credit_Risk.exceptions.exception import CustomException
import pandas as pd

from Predict_Bank_Credit_Risk.components.data_ingestion import DataIngestion
from Predict_Bank_Credit_Risk.components.data_transformation import DataTransformation
from Predict_Bank_Credit_Risk.components.model_trainer import ModelTrainer


if __name__=='__main__':
    obj=DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr,test_arr,_=data_transformation.initaite_data_transformation(train_data_path,test_data_path)
    model_trainer=ModelTrainer()
    model_trainer.initate_model_training(train_arr,test_arr)



