"""
Thư viện chứa các hàm chung cho pipeline
"""

import pandas as pd
import numpy as np
import zipfile
import os
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """Class để load và xử lý dữ liệu từ file zip"""
    
    @staticmethod
    def load_from_zip(zip_path: str) -> pd.DataFrame:
        """
        Load dữ liệu từ file zip chứa nhiều file CSV
        
        Parameters:
        -----------
        zip_path : str
            Đường dẫn đến file zip
            
        Returns:
        --------
        pd.DataFrame
            DataFrame hợp nhất của tất cả các trạm
        """
        all_data = []
        
        with zipfile.ZipFile(zip_path, 'r') as z:
            # Lấy danh sách file CSV trong zip
            csv_files = [f for f in z.namelist() if f.endswith('.csv')]
            
            print(f"Found {len(csv_files)} CSV files in zip")
            
            for csv_file in csv_files:
                try:
                    # Đọc từng file
                    with z.open(csv_file) as f:
                        df = pd.read_csv(f)
                        all_data.append(df)
                except Exception as e:
                    print(f"Error reading {csv_file}: {e}")
        
        # Hợp nhất tất cả dataframes
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"Combined shape: {combined_df.shape}")
            return combined_df
        else:
            raise ValueError("No CSV files found in zip")
    
    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Làm sạch dữ liệu cơ bản
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame gốc
            
        Returns:
        --------
        pd.DataFrame
            DataFrame đã làm sạch
        """
        df_clean = df.copy()
        
        # 1. Tạo cột datetime từ year, month, day, hour
        df_clean['datetime'] = pd.to_datetime(
            df_clean[['year', 'month', 'day', 'hour']]
        )
        
        # 2. Sắp xếp theo thời gian
        df_clean = df_clean.sort_values('datetime')
        
        # 3. Xử lý giá trị âm (nếu có)
        pollution_cols = ['pm2.5', 'pm10', 'so2', 'no2', 'co', 'o3']
        for col in pollution_cols:
            if col in df_clean.columns:
                df_clean.loc[df_clean[col] < 0, col] = np.nan
        
        # 4. Xử lý giá trị cực lớn (outliers)
        for col in pollution_cols:
            if col in df_clean.columns:
                q99 = df_clean[col].quantile(0.99)
                df_clean.loc[df_clean[col] > q99 * 10, col] = np.nan
        
        print("Data cleaning completed")
        return df_clean
    
    @staticmethod
    def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Thêm các đặc trưng thời gian
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame đã có cột datetime
            
        Returns:
        --------
        pd.DataFrame
            DataFrame với các đặc trưng thời gian
        """
        df_feat = df.copy()
        
        # Đặc trưng thời gian
        df_feat['hour'] = df_feat['datetime'].dt.hour
        df_feat['dayofweek'] = df_feat['datetime'].dt.dayofweek
        df_feat['dayofmonth'] = df_feat['datetime'].dt.day
        df_feat['month'] = df_feat['datetime'].dt.month
        df_feat['quarter'] = df_feat['datetime'].dt.quarter
        df_feat['year'] = df_feat['datetime'].dt.year
        df_feat['is_weekend'] = df_feat['dayofweek'].isin([5, 6]).astype(int)
        
        # Đặc trưng theo mùa
        df_feat['season'] = df_feat['month'] % 12 // 3 + 1
        
        print("Time features added")
        return df_feat
    
    @staticmethod
    def add_lag_features(df: pd.DataFrame, lag_hours: List[int] = [1, 3, 24]) -> pd.DataFrame:
        """
        Thêm đặc trưng lag cho PM2.5
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame theo thời gian
        lag_hours : List[int]
            Danh sách các độ trễ (giờ)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame với các đặc trưng lag
        """
        df_lag = df.copy()
        
        # Sắp xếp theo datetime và station
        df_lag = df_lag.sort_values(['station', 'datetime'])
        
        for lag in lag_hours:
            col_name = f'pm2.5_lag{lag}'
            df_lag[col_name] = df_lag.groupby('station')['pm2.5'].shift(lag)
        
        print(f"Added lag features: {lag_hours}")
        return df_lag


class EDAVisualizer:
    """Class để tạo các visualization cho EDA"""
    
    @staticmethod
    def plot_pm25_timeseries(df: pd.DataFrame, station: str = None):
        """
        Vẽ chuỗi thời gian PM2.5
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame chứa dữ liệu
        station : str, optional
            Tên trạm cụ thể, nếu None thì lấy trung bình tất cả trạm
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(15, 6))
        
        if station:
            # Lấy dữ liệu một trạm
            station_data = df[df['station'] == station].copy()
            pm25_series = station_data.set_index('datetime')['pm2.5']
            title = f'PM2.5 Time Series - Station: {station}'
        else:
            # Lấy trung bình tất cả trạm
            pm25_series = df.groupby('datetime')['pm2.5'].mean()
            title = 'PM2.5 Time Series - Average All Stations'
        
        # Vẽ chuỗi
        plt.plot(pm25_series.index, pm25_series.values, linewidth=0.5)
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('PM2.5')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    @staticmethod
    def plot_missing_pattern(df: pd.DataFrame):
        """Vẽ biểu đồ missing pattern"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Tính tỷ lệ missing
        missing_percent = (df.isnull().sum() / len(df)) * 100
        missing_percent = missing_percent[missing_percent > 0].sort_values(ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=missing_percent.values, y=missing_percent.index)
        plt.title('Missing Data Percentage by Column')
        plt.xlabel('Percentage Missing (%)')
        plt.ylabel('Column')
        plt.show()
    
    @staticmethod
    def plot_distribution(df: pd.DataFrame, column: str = 'pm2.5'):
        """Vẽ phân phối của một cột"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(12, 5))
        
        # Histogram với KDE
        plt.subplot(1, 2, 1)
        sns.histplot(df[column].dropna(), kde=True, bins=50)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        
        # Boxplot
        plt.subplot(1, 2, 2)
        sns.boxplot(y=df[column].dropna())
        plt.title(f'Boxplot of {column}')
        plt.ylabel(column)
        
        plt.tight_layout()
        plt.show()