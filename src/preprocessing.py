from src.metadata import (application_metadata, bureau_bal_metadata, bureau_metada, cc_bal_metadata, inst_pmt_metadata, 
                      pos_cash_bal_metadata, prev_app_metadata, loans_config, reg_area_config)
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tqdm import tqdm

import pandas as pd
import numpy as np
import re

class dataPipeline():
    
    def __init__(self, pipeline_type='train'):
        self.tg = 'TARGET'
        self.pkey = 'SK_ID_CURR'
        self.pipeline_type = pipeline_type
    
    def readCSV(self, file_name):
        return pd.read_csv(f"data/{file_name}.csv")
    
    def importData(self, data):
        
        if data == 'app_train':
            return self.readCSV('application_train')
        elif data == 'app_test':
            return self.readCSV('application_test')
        elif data == 'prev_app':
            return self.readCSV('previous_application')
        elif data == 'bureau':
            return self.readCSV('bureau')
        elif data == 'bureau_bal':
            return self.readCSV('bureau_balance')
        elif data == 'cc':
            return self.readCSV('credit_card_balance')
        elif data == 'inst':
            return self.readCSV('installments_payments')
        elif data =='pos':
            return self.readCSV('POS_CASH_balance')
        
    def _percentile_transform(self, df, col):
        return round(df[col].rank(pct=True), 2)
    
    def _udf_derive_loan_planning(self, row):
        AMT_GOODS_PRICE = row['AMT_GOODS_PRICE']
        AMT_CREDIT = row['AMT_CREDIT']
        if AMT_GOODS_PRICE == AMT_CREDIT:
            return 'exact'
        elif AMT_GOODS_PRICE > AMT_CREDIT:
            return 'lower'
        elif AMT_GOODS_PRICE < AMT_CREDIT:
            return 'higher'
        
    def _clean_feature_name(self, name):
    
        name = re.sub(r'\W+', '_', name)
        if not re.match(r'^[a-zA-Z_]', name):
            name = '_' + name

        return name
    
    def _app_transform(self, df):
        
        # Convert age
        df['APP_AGE'] = round(df['DAYS_BIRTH'] / -365, 2)
        
        # Set up outlier
        df.loc[df['DAYS_EMPLOYED'] == 365243, 'DAYS_EMPLOYED'] = np.nan
        
        # Extract family members
        df['APP_SPOUSE_CNT'] = df.NAME_FAMILY_STATUS.apply(lambda x: 1 if x in ['Civil marriage', 'Married'] else 0)
        df['APP_OTH_FAMILY_MEMBER'] = df['CNT_FAM_MEMBERS'] - df['CNT_CHILDREN'] - df['APP_SPOUSE_CNT'] - 1 #self
        
        # Flag 
        # ister regian
        
        df['APP_REGION_REG'] = df.apply(lambda row: "-".join([str(row[col]) for col in ['REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION']]), axis=1)
        df['APP_CITY_REG'] = df.apply(lambda row: "-".join([str(row[col]) for col in ['REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY']]), axis=1)
        
        df['APP_REGION_REG'] = df['APP_REGION_REG'].apply(lambda x: reg_area_config[x])
        df['APP_CITY_REG'] = df['APP_CITY_REG'].apply(lambda x: reg_area_config[x])
        
        # Combind application document
        df[application_metadata['document_info'][:20]] = df[application_metadata['document_info'][:20]].astype(str)
        df['APP_DOCUMENTS'] = df[application_metadata['document_info'][:20]].agg(''.join, axis=1)
        
        df[application_metadata['document_info'][:20]] = df[application_metadata['document_info'][:20]].astype(int)
        df['APP_MISSING_DOC'] = df[application_metadata['document_info'][:20]].sum(axis=1)
        
        df['APP_APPL_WKND_FLAG'] = df['WEEKDAY_APPR_PROCESS_START'].apply(lambda x: 1 if x in ['SATURDAY', 'SUNDAY'] else 0)
        
        # Calculate credit terms
        df['APP_N_REPAYMENT'] = np.ceil(df['AMT_CREDIT'] / df['AMT_ANNUITY'])
        
        # Check loan planning
        df['APP_LOAN_PLANNING'] = df.apply(lambda row: self._udf_derive_loan_planning(row), axis=1)
        df['APP_BURDEN_PERCENTAGE'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
        df['APP_RISK_REWARD_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
        
        # Handling default
        df['APP_FRIEND_DEFAULT_FLAG'] = df['DEF_30_CNT_SOCIAL_CIRCLE'].apply(lambda x: 1 if x > 0 else 0)
        
        return df.reset_index(drop=True)
            
    def _prev_app_transform(self, df):
        df_pivot = df[df['NAME_CONTRACT_TYPE'] != 'XNA'].pivot_table(index=self.pkey, 
                                                                     columns=['NAME_CONTRACT_TYPE', 'NAME_CONTRACT_STATUS'], 
                                                                     values='SK_ID_PREV', aggfunc='count').fillna(0).reset_index()
        df_pivot.columns = ['SK_ID_CURR', 'CL_AP', 'CL_CN', 'CL_RF', 'CL_OU', 
                             'CSML_AP', 'CSML_CN', 'CSML_RF', 'CSML_OU',
                             'RL_AP', 'RL_CN', 'RL_RF', 'RL_OU',
                             ]
        
        df_pivot['PREV_CNT_APPR'] = df_pivot['CL_AP']+df_pivot['CSML_AP']+df_pivot['RL_AP']
        df_pivot['PREV_CNT_CANC'] = df_pivot['CL_CN']+df_pivot['CSML_CN']+df_pivot['RL_CN']
        df_pivot['PREV_CNT_RFUS'] = df_pivot['CL_RF']+df_pivot['CSML_RF']+df_pivot['RL_RF']
        df_pivot['PREV_CNT_NUSE'] = df_pivot['CL_OU']+df_pivot['CSML_OU']+df_pivot['RL_OU']
        
        df_pivot['PREV_CNT_TOTL'] = df_pivot['PREV_CNT_APPR']+df_pivot['PREV_CNT_CANC']+df_pivot['PREV_CNT_RFUS']+df_pivot['PREV_CNT_NUSE']

        df_pivot['PREV_PCT_APPR'] = df_pivot['PREV_CNT_APPR'] *100/ df_pivot['PREV_CNT_TOTL']
        df_pivot['PREV_PCT_CANC'] = df_pivot['PREV_CNT_CANC'] *100 / df_pivot['PREV_CNT_TOTL']
        df_pivot['PREV_PCT_RFUS'] = df_pivot['PREV_CNT_RFUS'] *100 / df_pivot['PREV_CNT_TOTL']
        df_pivot['PREV_PCT_NUSE'] = df_pivot['PREV_CNT_NUSE'] *100 / df_pivot['PREV_CNT_TOTL']
        
        return df_pivot
    
    def _bureau_bal_transform(self, df_bur, df_bal):
        
        # Get all loan from Bureau data
        df_all_loan = df_bur.groupby(self.pkey)[['SK_ID_BUREAU']].count().reset_index()
        df_all_loan.columns = [self.pkey, 'BR_CNT_PREV_LOAN']
        
        # Pivot bureau balance
        df_bal = df_bal.merge(df_bur[['SK_ID_BUREAU', self.pkey]], on='SK_ID_BUREAU', how='left')
        df_all_dpd = df_bal.pivot_table(index=self.pkey, values='MONTHS_BALANCE', 
                                       columns='STATUS', aggfunc='min').reset_index() # value is negative, min yield max
        
        # Flagginf for DPD
        df_all_dpd['BR_30DPD_FLAG'] =  df_all_dpd['1'].apply(lambda x: 1 if x < 0 else 0)
        df_all_dpd['BR_60DPD_FLAG'] =  df_all_dpd['2'].apply(lambda x: 1 if x < 0 else 0)
        df_all_dpd['BR_90DPD_FLAG'] =  df_all_dpd['3'].apply(lambda x: 1 if x < 0 else 0)
        df_all_dpd['BR_120DPD_FLAG'] =  df_all_dpd['4'].apply(lambda x: 1 if x < 0 else 0)
        df_all_dpd['BR_WRITE_OFF_FLAG'] =  df_all_dpd['5'].apply(lambda x: 1 if x < 0 else 0)

        df_all_dpd['BR_DPD_EVER_FLAG'] =  df_all_dpd.apply(lambda r: 1 if r['1']+r['2']+r['3']+r['4']+r['5'] < 0 else 0, axis=1)

        return df_all_dpd, df_all_loan
    
    def _cc_inst_transform(self, df_cc, df_inst):
        
        df_inst_cc = df_inst[df_inst['NUM_INSTALMENT_VERSION'] == 0].reset_index(drop=True)
        df_inst_cc['CC_GAP_INST_PYMT'] = df_inst_cc['AMT_INSTALMENT'] - df_inst_cc['AMT_PAYMENT']
        
        df_agg_inst_cc = df_inst_cc.pivot_table(index='SK_ID_CURR', 
                                                values = ['NUM_INSTALMENT_NUMBER', 'AMT_INSTALMENT', 'AMT_PAYMENT', 'CC_GAP_INST_PYMT'], 
                                                aggfunc=['min','max','median','mean', "std"]).reset_index()
        df_agg_inst_cc.columns = [
            'SK_ID_CURR', 
            'CC_MIN_GAP', 'CC_MIN_INST', 'CC_MIN_PMNT', 'CC_MIN_INST_NBR',
            'CC_MAX_GAP', 'CC_MAX_INST', 'CC_MAX_PMNT', 'CC_MAX_INST_NBR',
            'CC_MED_GAP', 'CC_MED_INST', 'CC_MED_PMNT', 'CC_MED_INST_NBR',
            'CC_AVG_GAP', 'CC_AVG_INST', 'CC_AVG_PMNT', 'CC_AVG_INST_NBR',
            'CC_SD_GAP', 'CC_SD_INST', 'CC_SD_PMNT', 'CC_SD_INST_NBR',
        ]
        
        df_cc_behavior = df_cc.pivot_table(index=self.pkey, values='AMT_TOTAL_RECEIVABLE', aggfunc=['mean', 'median', 'min', 'max', 'std']).reset_index().round(2)
        df_cc_behavior.columns = [self.pkey, 'CC_AVG_SPD', 'CC_MED_SPD', 'CC_MIN_SPD', 'CC_MAX_SPD', 'CC_SD_SPD']
        
        df_cc_behavior = df_cc_behavior.merge(df_agg_inst_cc.reset_index(drop=True), on=self.pkey, how='inner')
        
        scaler = MinMaxScaler(feature_range=(-1, 1))
        
        df_cc_behavior['CC_SCL_INST_NBR'] = scaler.fit_transform(df_cc_behavior[['CC_MAX_INST_NBR']])
        df_cc_behavior['CC_SCL_MED_SPD'] = scaler.fit_transform(df_cc_behavior[['CC_MED_SPD']])
        
        df_cc_behavior['CC_QDR_INST_NBR'] = df_cc_behavior['CC_SCL_INST_NBR'].apply(lambda x: 0 if x < 0 else 1).astype(int)
        df_cc_behavior['CC_QDR_MED_SPD'] = df_cc_behavior['CC_SCL_MED_SPD'].apply(lambda x: 0 if x < 0 else 1).astype(int)
        
        df_cc_behavior['CC_QDR_SPD_INST'] = df_cc_behavior.apply(lambda row: "_".join([str(row[col]) for col in ['CC_QDR_INST_NBR', 'CC_QDR_MED_SPD']]), axis=1)
        
        df_cc_aging = df_cc.pivot_table(index=self.pkey, values='MONTHS_BALANCE', aggfunc='min').reset_index()
        df_cc_aging.columns = ['SK_ID_CURR', 'CC_MONTHS_BALANCE']
        
        return df_agg_inst_cc, df_cc_aging, df_cc_behavior
    
    def _pos_transform(self, df_pos):
        
        df_pos_agg = df_pos.pivot_table(index='SK_ID_CURR', values=['CNT_INSTALMENT', 'SK_ID_PREV'], aggfunc={'CNT_INSTALMENT':'max', 'SK_ID_PREV':'nunique'}).reset_index()
        df_pos_agg.columns = [self.pkey, 'POS_MAX_INST', 'POS_CNT']
        
        scaler = MinMaxScaler(feature_range=(-1, 1))
        df_pos_agg['POS_SCL_MAX_INST'] = scaler.fit_transform(df_pos_agg[['POS_MAX_INST']])
        df_pos_agg['POS_SCL_CNT'] = scaler.fit_transform(df_pos_agg[['POS_CNT']])
        
        df_pos_agg['POS_QDR_SCL_MAX_INST'] = df_pos_agg['POS_SCL_MAX_INST'].apply(lambda x: 0 if x < 0 else 1).astype(int)
        df_pos_agg['POS_QDR_SCL_CNT'] = df_pos_agg['POS_SCL_CNT'].apply(lambda x: 0 if x < 0 else 1).astype(int)
        
        df_pos_agg['POS_QDR_POS_INST'] = df_pos_agg.apply(lambda row: "_".join([str(row[col]) for col in ['POS_QDR_SCL_MAX_INST', 'POS_QDR_SCL_CNT']]), axis=1)        
        
        return df_pos_agg
    
    def _merge(self, df1, df2):
        
        df_merged = pd.merge(df1, df2, on=self.pkey, how='left')
        
        return df_merged.reset_index(drop=True)
        
    
    def featureEngineering(self, df_app):
        
        # Import all data
        df_prev_app = self.importData('prev_app')
        df_bureau = self.importData('bureau')
        df_bureau_bal = self.importData('bureau_bal')
        df_cc = self.importData('cc')
        df_inst = self.importData('inst')
        df_pos = self.importData('pos')
        
        # Clean and transform data
        df_app = self._app_transform(df_app)
        df_all_dpd, df_all_loan = self._bureau_bal_transform(df_bureau, df_bureau_bal)
        df_agg_inst_cc, df_cc_aging, df_cc_behavior = self._cc_inst_transform(df_cc, df_inst)
        df_pv_prev_app = self._prev_app_transform(df_prev_app)
        df_pos_agg = self._pos_transform(df_pos)
        
        # Left join with main application data with SK_ID_CURR key
        COLS_BUREAU_DPD = [
            'SK_ID_CURR', 'BR_30DPD_FLAG', 'BR_60DPD_FLAG', 'BR_90DPD_FLAG', 'BR_120DPD_FLAG',
            'BR_WRITE_OFF_FLAG', 'BR_DPD_EVER_FLAG'
            ]
        COLS_PREV_APPS = [
            'SK_ID_CURR', 'PREV_CNT_APPR', 'PREV_CNT_CANC', 'PREV_CNT_RFUS', 'PREV_CNT_NUSE',
            'PREV_CNT_TOTL', 'PREV_PCT_APPR', 'PREV_PCT_CANC', 'PREV_PCT_RFUS',
            'PREV_PCT_NUSE'
        ]
        df_main = self._merge(df_app, df_all_dpd[COLS_BUREAU_DPD])
        df_main = self._merge(df_main, df_all_loan)
        df_main = self._merge(df_main, df_agg_inst_cc)
        df_main = self._merge(df_main, df_cc_aging)
        df_main = self._merge(df_main, df_cc_behavior)
        df_main = self._merge(df_main, df_pv_prev_app[COLS_PREV_APPS])
        df_main = self._merge(df_main, df_pos_agg)

        return df_main
        
    def _encode_categorical_variables(self, df):
        # Identify categorical columns
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        
        # Wrap the loop with tqdm
        for column in tqdm(categorical_columns, desc="Encoding categorical variables"):
            unique_values = df[column].nunique()
            
            if unique_values <= 2:
                # Binary categorical variable: Use Label Encoding
                le = LabelEncoder()
                df[column] = le.fit_transform(df[column])
            elif unique_values > 2:
                # Multi-class categorical variable: Use One-Hot Encoding
                df = pd.get_dummies(df, columns=[column], prefix=column)
                        
        return df
    
    def preProcessing(self):
        
        if self.pipeline_type == 'train':
            # Ingest and prepare data from various souces into signle file
            df_main = self.featureEngineering(self.importData('app_train'))
        elif self.pipeline_type == 'prediction':
            df_main = self.featureEngineering(self.importData('app_test'))
        
        # Encode categorical feature
        df_main = self._encode_categorical_variables(df_main)
        
        # Set up feature
        df_main.columns = [self._clean_feature_name(col) for col in df_main.columns]    

        if self.pipeline_type == 'train':
            return df_main.drop(columns=[self.tg]), df_main[[self.tg]]  
        elif self.pipeline_type == 'prediction':
            return df_main