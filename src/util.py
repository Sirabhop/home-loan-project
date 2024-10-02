import pandas as pd
import numpy as np
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

class DataQualityChecker:
    """
    This class performs data quality checking
    1. Null/ Missing data
    2. Duplicates data based on primary key
    3. Distributions of each field
    4. Suspected outlier
    """

    def __init__(self, df: pd.DataFrame, dict_metadata):
        self.df = df
        self.dict_metadata = dict_metadata
        self.primary_key = dict_metadata['primary_key'] if dict_metadata != None else None
        self.df[self.primary_key] = self.df[self.primary_key].astype(str)
        self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        self.categorical_columns = self.df.select_dtypes(include=['object']).columns
        self.numeric_report = pd.DataFrame(index=self.numeric_columns)
        self.categorical_report = pd.DataFrame(index=self.categorical_columns)

    def check_nulls(self) -> None:
        """Checks for null values in each column"""
        null_percentages = (self.df.isnull().sum() / len(self.df) * 100).round(2)
        self.numeric_report["null_percentages"] = null_percentages[self.numeric_columns]
        self.categorical_report["null_percentages"] = null_percentages[self.categorical_columns]

    def check_duplicates(self) -> None:
        """Checks for duplicate rows based on the primary key."""
        duplicate_counts = self.df.groupby(self.primary_key).size() - 1
        self.numeric_report["duplicate_counts"] = duplicate_counts.reindex(self.numeric_columns, fill_value=0)
        self.categorical_report["duplicate_counts"] = duplicate_counts.reindex(self.categorical_columns, fill_value=0)

    def check_distributions(self) -> None:
        """Analyzes the distribution of each field."""
        # Numeric features
        numeric_stats = self.df[self.numeric_columns].agg(['min', 'mean', 'median', 'max', 'std',
                                                           lambda x: x.quantile(0.25),
                                                           lambda x: x.quantile(0.75),
                                                           lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan])
        numeric_stats.index = ['min', 'mean', 'median', 'max', 'std', '25%', '75%', 'mode']
        for stat in numeric_stats.index:
            self.numeric_report[stat] = numeric_stats.loc[stat]

        # Categorical features
        for col in self.categorical_columns:
            value_counts = self.df[col].value_counts()
            self.categorical_report.at[col, 'top_value'] = value_counts.index[0] if not value_counts.empty else np.nan
            self.categorical_report.at[col, 'top_percentage_shares'] = (value_counts.iloc[0] if not value_counts.empty else 0) * 100 / self.df.shape[0]
            self.categorical_report.at[col, 'unique_count'] = self.df[col].nunique()

    def check_outliers(self, threshold: float = 3.0) -> None:
        """
        Detects suspected outliers using the Z-score method.
        Values with a Z-score greater than the threshold (default 3) are considered outliers.
        """
        for column in self.numeric_columns:
            z_scores = np.abs((self.df[column] - self.df[column].mean()) / self.df[column].std())
            outlier_count = (z_scores > threshold).sum()
            self.numeric_report.at[column, 'outliers_percentage'] = outlier_count * 100 / self.df.shape[0]

    def run_all_checks(self) -> Dict[str, pd.DataFrame]:
        """Runs all data quality checks and returns the complete reports."""
        self.check_nulls()
        self.check_duplicates()
        self.check_distributions()
        self.check_outliers()

        # Reorder columns in numeric_report
        numeric_column_order = ['null_percentages', 'duplicate_counts', 'outliers_percentage',
                                'min', '25%', 'median', 'mean', 'mode', '75%', 'max', 'std']
        self.numeric_report = self.numeric_report.reindex(columns=numeric_column_order)

        # Rounding
        self.numeric_report =self.numeric_report.round(3)

        # Grouping columns
        transformed_data = {} 
        
        for key, value_list in self.dict_metadata.items():
            for value in value_list:
                transformed_data[value] = key

        for r in [self.categorical_report, self.numeric_report]:
                
            r.reset_index(inplace=True)
            r['groups'] = r['index'].apply(lambda x: transformed_data[x])
            r.sort_values(by=['groups'], inplace=True)
            r.set_index(keys=['groups', 'index'], inplace=True)
            
        return {"numeric_report": self.numeric_report, "categorical_report": self.categorical_report}
    
    
    

class graphVisualizer():
    
    def __init__(self, df):
        
        self.df = df
        
    def cat(self, index, columns='TARGET', values='SK_ID_CURR', aggfunc='count', p_flag=True):
        
        df_pivot = self.df.pivot_table(index=index, columns=columns, values=values, aggfunc=aggfunc)
        df_pivot_percentage = df_pivot.div(df_pivot.sum(axis=0), axis=1) * 100

        ax = df_pivot_percentage.plot(kind='bar') 

        if p_flag == True:
            for i, col in enumerate(df_pivot_percentage.columns):
                for j, val in enumerate(df_pivot_percentage[col]):
                    plt.text(j, val, f'{val:.1f}%')

        ax.set_ylim(0, 100)

        plt.title(f'Distribution of {index} by {columns} (Percentage)')
        plt.show()
        
    def cdf(self, index, values='TARGET'):
        
        sns.ecdfplot(self.df, x=index, hue=values)
        
        plt.title(f'Distribution of {index}')
        plt.show()
        
    def hist(self, index, values='TARGET'):
        
        sns.histplot(self.df, x=index, hue=values)
        
        plt.title(f'Distribution of {index}')
        plt.show()
        
        
    def pctDfr(self, index, column='TARGET', value='SK_ID_CURR', groupby_flag=True, df=None):
        if df is not None:
            pass
        else:
            df = self.df.copy()
            
        if groupby_flag:
            
            to_pivot = df[~df[index].isnull()].reset_index(drop=True)
            to_pivot[f'P_{index}'] = (round(to_pivot[index].rank(pct=True) * 100, 0)).astype(int)

            df_pivot = to_pivot.pivot_table(index=f'P_{index}', columns=column, values=value, aggfunc='count').reset_index()
            df_pivot['DFR'] = df_pivot[1] * 100/(df_pivot[0]+df_pivot[1])
            
            plt.figure(figsize=(20,4))
            sns.barplot(df_pivot, x=f'P_{index}', y='DFR')
        else:
            df_pivot = df.pivot_table(index=f'{index}', columns=column, values=value, aggfunc='count').reset_index()
            df_pivot['DFR'] = df_pivot[1] * 100/(df_pivot[0]+df_pivot[1])
            plt.figure(figsize=(20,4))
            sns.barplot(df_pivot, x=f'{index}', y='DFR')
            
        average_dfr = df_pivot['DFR'].mean()
        plt.axhline(y=average_dfr, color='r', linestyle='-', label=f'Average DFR: {average_dfr:.2f}%')
        plt.legend()
        plt.show()