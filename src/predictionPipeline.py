
import glob
import os
import shap

import pandas as pd
import numpy as np
import pickle

from datetime import datetime

class predictionPipeline():
    
    def __init__(self, log_path = 'logs', out_path = 'predictions'):
        
        self.log_path = log_path
        self.prediction_output_path = out_path
        
        self.log = self._import_log()
        self.model_path = self.log['best_model_saved']['filepath']
        self.load_best_model()
        
    def load_best_model(self):

        if not os.path.exists(self.model_path):
            print(f"Model not in the {self.model_path}")
            return None

        with open(self.model_path, 'rb') as file:
            self.best_model = pickle.load(file)
            self.selected_features = self.best_model.get_booster().feature_names
            
        print(f"Successfully downlond the model from {self.model_path}")
        
    def __get_latest_log(self):
        # Get a list of all log files in the directory
        log_files = glob.glob(os.path.join(self.log_path, '*.csv'))
        
        # Sort the files based on modification time, newest first
        latest_log = max(log_files, key=os.path.getmtime)
        
        return latest_log

    def _import_log(self):
        
        # Check if the file exists
        if not os.path.exists(self.log_path):
            print(f"File not found: {self.log_path}")
            return None
        latest_log_path = self.__get_latest_log()
        
        # Read the CSV file
        df = pd.read_csv(latest_log_path, index_col='Key')

        log_dict = {}
        for key, value in df['Value'].items():
            key_parts = key.split('.')

            current_dict = log_dict
            for part in key_parts[:-1]:
                if part not in current_dict:
                    current_dict[part] = {}
                current_dict = current_dict[part]

            current_dict[key_parts[-1]] = value

        if 'best_model_saved' in log_dict and 'timestamp' in log_dict['best_model_saved']:
            log_dict['best_model_saved']['timestamp'] = datetime.strptime(
                log_dict['best_model_saved']['timestamp'], "%Y%m%d_%H%M%S"
            )

        return log_dict
        
    def predict(self, df):
        if self.best_model is None:
            print("No model has been loaded. Please load a model first.")
            return None
        
        df = df.reindex(columns=self.selected_features, fill_value=False)
        df_to_predict = df[self.selected_features]
        
        
        # Generate predictions
        predictions = self.best_model.predict_proba(df_to_predict)[:, 1]
        print("Predictions generated successfully.")
        
        # Explain all of the predictions
        explainer = shap.TreeExplainer(self.best_model)
        shap_values = explainer.shap_values(df_to_predict)
        
        # Create a DataFrame with predictions and explanations
        new_columns = {}
        # Prepare all new columns at once
        for i, feature in enumerate(df_to_predict.columns):
            new_columns[f'contr_{feature}'] = shap_values[:, i]
            new_columns[f'val_{feature}'] = df_to_predict.iloc[:, i]
            
        # Use pd.concat to add all new columns at once
        results_df = pd.concat([df_to_predict, pd.DataFrame(new_columns)], axis=1)
        
        results_df['Total_Positive_Contribution'] = results_df[[col for col in results_df.columns if col.startswith('contr_')]].clip(lower=0).sum(axis=1)
        results_df['Total_Negative_Contribution'] = results_df[[col for col in results_df.columns if col.startswith('contr_')]].clip(upper=0).sum(axis=1)        # Store the results for export
        
        self.prediction_results = results_df
        
        # Export to Excel
        self._export_predictions(results_df)
        
        return predictions

    def _export_predictions(self, results_df):
        if not hasattr(self, 'prediction_results') or self.prediction_results is None or results_df is None:
            print("No predictions available to export.")
            return
        
        # Generate a unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"loan_predictions_{timestamp}.xlsx"
        filepath = os.path.join(self.prediction_output_path, filename)
        
        os.makedirs(self.prediction_output_path, exist_ok=True)
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            self.prediction_results.to_excel(writer, sheet_name='Predictions', index=False)
            
            # Write feature importance
            if hasattr(self.best_model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'Feature': self.selected_features,
                    'Importance': self.best_model.feature_importances_
                }).sort_values('Importance', ascending=False)
                feature_importance.to_excel(writer, sheet_name='Feature_Importance', index=False)
            
            print(f"Predictions and explanations successfully exported to: {filepath}")