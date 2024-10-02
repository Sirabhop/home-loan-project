import scipy
import numpy as np
import pandas as pd
import pickle
import os

from datetime import datetime
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

class modelPipeline():
    # Model pipeline for Loan default classification
    
    def __init__(self, X, y):
        self.seed = 42
        self.X = X.drop(columns='SK_ID_CURR')
        self.y = y
        self.selected_features = None
        self.best_model = None
        self.log = {}
        
    def feature_filtering(self, threshold=0.8):
        numeric_columns = self.X.select_dtypes(include=['int64', 'float64']).columns
        categorical_columns = self.X.select_dtypes(include=['object', 'category']).columns

        # Numeric features: correlation (pearson?)
        if len(numeric_columns) > 0:
            corr_matrix = self.X[numeric_columns].corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop_numeric = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        else:
            to_drop_numeric = []

        # Categorical features: chi-square
        to_drop_categorical = []
        if len(categorical_columns) > 0:
            for i in range(len(categorical_columns)):
                for j in range(i + 1, len(categorical_columns)):
                    cramers_v = self.__chi2(self.X[categorical_columns[i]], 
                                          self.X[categorical_columns[j]])
                    if cramers_v > threshold:
                        to_drop_categorical.append(categorical_columns[j])

        # Combine features to drop
        to_drop = list(set(to_drop_numeric + to_drop_categorical))

        # Drop highly correlated features
        self.X = self.X.drop(to_drop, axis=1)
        
        self.log['step1_feature_filtering'] = {
            'threshold': threshold,
            'numeric_cols':  to_drop_numeric,
            'numeric_count': len(to_drop_numeric),
            'categorical_cols': to_drop_categorical,
            'categorical_count': len(to_drop_categorical)
        }
        
        print(f"Dropped {len(to_drop)} features due to high correlation")

    def __chi2(self, x, y):
        confusion_matrix = pd.crosstab(x, y)
        chi2 = scipy.stats.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        min_dim = min(confusion_matrix.shape) - 1
        return np.sqrt(chi2 / (n * min_dim))    
    
    def feature_selection(self, feature_ratio=0.8):
        # Feature selection using Random Forest
        
        rf = RandomForestClassifier(n_estimators=100, random_state=self.seed)
        
        n_features = np.ceil(feature_ratio * self.X.shape[1])
        selector = SelectFromModel(rf, max_features=int(n_features))
        
        selector.fit(self.X, self.y)
        
        # Get selected feature names
        feature_mask = selector.get_support()
        self.selected_features = self.X.columns[feature_mask]
        
        # Update X with selected features
        self.X = self.X[self.selected_features]
        
        print(f"Selected {len(self.selected_features)} best features using Random Forest")

        # Optional: Print top 10 feature importances
        feature_importances = selector.estimator_.feature_importances_
        top_features = sorted(zip(feature_importances, self.selected_features), reverse=True)[:10]
        print("\nTop 10 features by importance:")
        for importance, feature in top_features:
            print(f"{feature}: {importance:.4f}")
                
    def hyper_param_tuner(self, model, param_grid):
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')
        grid_search.fit(self.X, self.y)
        return grid_search.best_estimator_, grid_search.best_params_
    
    def XGBoost(self):
        xgb_params = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [100, 200, 300]
        }
        xgb_model, xgb_best_params = self.hyper_param_tuner(XGBClassifier(random_state=self.seed), xgb_params)

        return xgb_model, xgb_best_params
    
    def LightGBM(self):
        lgbm_params = {
            'num_leaves': [31, 63, 127],
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [100, 200, 300]
        }
        lgbm_model, lgbm_best_params = self.hyper_param_tuner(LGBMClassifier(random_state=self.seed), lgbm_params)
    
        return lgbm_model, lgbm_best_params
    
    def model_selection(self):
        
        xgb_model, xgb_best_params = self.XGBoost()
        lgbm_model, lgbm_best_params = self.LightGBM()
        
        # Compare models
        models = [xgb_model, lgbm_model]
        model_names = ['XGBoost', 'LightGBM']
        best_score = 0
        
        for model, name in zip(models, model_names):
            score = np.mean(cross_val_score(model, self.X, self.y, cv=5, scoring='roc_auc'))
            print(f"{name} ROC-AUC: {score:.4f}")
            if score > best_score:
                best_score = score
                self.best_model = model
        
        print(f"Best model: {model_names[models.index(self.best_model)]} with ROC-AUC: {best_score:.4f}")
    
    def classification_evaluator(self):
        # AUC-ROC
        if self.best_model is None:
            print("Please run modelSelection first")
            return
        
        y_pred_proba = self.best_model.predict_proba(self.X)[:, 1]
        auc_roc = roc_auc_score(self.y, y_pred_proba)
        print(f"AUC-ROC score: {auc_roc:.4f}")

    def save_best_model(self, directory='model'):

        # Create directory if not exist
        if not os.path.exists(directory):
            os.makedirs(directory)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.best_model.__class__.__name__
        filename = f"{model_name}_{timestamp}.pkl"
        filepath = os.path.join(directory, filename)

        # Save the model to pickle
        with open(filepath, 'wb') as file:
            pickle.dump(self.best_model, file)

        print(f"Best model saved as {filepath}")

        self.log['best_model_saved'] = {
            'model_name': model_name,
            'filename': filename,
            'filepath': filepath,
            'timestamp': timestamp
        }
        
        self._export_log_to_csv()
    
    def _export_log(self, directory='logs'):
        
        # Create directory if it not exist
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Ensure the log dictionary is not empty
        if not self.log:
            print("Log is empty. Nothing to export.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"log_{timestamp}.csv"
        filepath = os.path.join(directory, filename)

        flat_log = {}
        for key, value in self.log.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat_log[f"{key}.{sub_key}"] = sub_value
            else:
                flat_log[key] = value

        df = pd.DataFrame.from_dict(flat_log, orient='index', columns=['Value'])
        df.index.name = 'Key'

        df.to_csv(filepath)
        print(f"Log exported to {filepath}")