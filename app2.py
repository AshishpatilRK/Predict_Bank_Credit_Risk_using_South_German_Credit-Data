from flask import Flask,request,render_template,jsonify
from Predict_Bank_Credit_Risk.pipeline.prediction_pipeline import CustomData,PredictPipeline
from flask_cors import cross_origin

from Predict_Bank_Credit_Risk.logging.logger import logging
from Predict_Bank_Credit_Risk.exceptions.exception import CustomException
from Predict_Bank_Credit_Risk.components.data_ingestion import DataIngestion
from Predict_Bank_Credit_Risk.components.data_transformation import DataTransformation
from Predict_Bank_Credit_Risk.components.model_trainer import ModelTrainer

application=Flask(__name__)
app=application


@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])

def predict_datapoint():
        if request.method == "POST":
                    data=CustomData(
                            status = int(request.form['status']),

                            duration = int(request.form['duration']),

                            credit_history = int(request.form['credit_history']),

                            purpose = int(request.form['purpose']),

                            amount = int(request.form['amount']),

                            savings = int(request.form['savings']),

                            employment_duration = int(request.form['employment_duration']),

                            installment_rate = int(request.form['installment_rate']),

                            personal_status_sex = int(request.form['personal_status_sex']),

                            other_debtors = int(request.form['other_debtors']),

                            present_residence = int(request.form['present_residence']),

                            property = int(request.form['property']),

                            age = int(request.form['age']),

                            other_installment_plans = int(request.form['other_installment_plans']),

                            housing = int(request.form['housing']),

                            number_credits = int(request.form['number_credits']),

                            job = int(request.form['job']),
                            
                            people_liable = int(request.form['people_liable']),

                            telephone = int(request.form['telephone']),

                            foreign_worker = int(request.form['foreign_worker'])
                    )
                    final_new_data=data.get_data_as_dataframe()
                    predict_pipeline=PredictPipeline()
                    pred=predict_pipeline.predict(final_new_data)

                    if pred== 0:
                                label = 'Bad'
                    else:
                                label = 'Good'

                    return render_template('result.html', prediction_text=" The Credit Risk Is {}".format(label))

        # result=round(pred[0],2)

        # return render_template('result.html',final_result=result)
    

@app.route('/train')
def train_pipeline():
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

    model_trainer = ModelTrainer()
    best_model_name, best_model_score, model_report = model_trainer.initiate_model_training(train_arr, test_arr)

    return render_template('train_pipeline.html', best_model_name=best_model_name, best_model_score=best_model_score, model_report=model_report)
    

if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True, port=5000)