"""
Thư viện cho mô hình hồi quy baseline
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

class RegressionModeler:
    """Class để xây dựng và đánh giá mô hình hồi quy"""
    
    @staticmethod
    def prepare_regression_data(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
        """
        Chuẩn bị dữ liệu cho bài toán hồi quy dự báo
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame đã có datetime và các features
        horizon : int
            Số giờ dự báo trước
            
        Returns:
        --------
        pd.DataFrame
            DataFrame với target là PM2.5 tại t+horizon
        """
        df_prep = df.copy()
        
        # Tạo target: PM2.5 tại thời điểm t+horizon
        df_prep = df_prep.sort_values(['station', 'datetime'])
        df_prep['target'] = df_prep.groupby('station')['pm2.5'].shift(-horizon)
        
        # Loại bỏ các dòng có target là NaN
        df_prep = df_prep.dropna(subset=['target'])
        
        print(f"Regression data prepared with horizon={horizon}")
        print(f"Shape: {df_prep.shape}")
        
        return df_prep
    
    @staticmethod
    def split_time_series(df: pd.DataFrame, cutoff_date: str, 
                         target_col: str = 'target') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Chia train/test theo thời gian
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame đã sắp xếp theo thời gian
        cutoff_date : str
            Ngày cắt (format: 'YYYY-MM-DD')
        target_col : str
            Tên cột target
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            Train và test DataFrames
        """
        # Chuyển cutoff_date sang datetime
        cutoff = pd.to_datetime(cutoff_date)
        
        # Chia theo thời gian
        train_df = df[df['datetime'] < cutoff].copy()
        test_df = df[df['datetime'] >= cutoff].copy()
        
        print(f"Train size: {len(train_df)} ({train_df['datetime'].min()} to {train_df['datetime'].max()})")
        print(f"Test size: {len(test_df)} ({test_df['datetime'].min()} to {test_df['datetime'].max()})")
        
        return train_df, test_df
    
    @staticmethod
    def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Chuẩn bị features cho mô hình hồi quy
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame với các cột thô
            
        Returns:
        --------
        Tuple[pd.DataFrame, List[str]]
            DataFrame với features và danh sách tên features
        """
        # Danh sách features mặc định
        base_features = [
            'pm2.5', 'pm10', 'so2', 'no2', 'co', 'o3',  # Ô nhiễm
            'temp', 'pres', 'dewp', 'rain', 'wspm',      # Khí tượng
            'hour', 'dayofweek', 'month', 'is_weekend', 'season'  # Thời gian
        ]
        
        # Thêm lag features nếu có
        lag_features = [col for col in df.columns if 'lag' in col]
        
        # Tất cả features
        all_features = base_features + lag_features
        
        # Chỉ lấy features có trong dataframe
        available_features = [f for f in all_features if f in df.columns]
        
        print(f"Available features: {len(available_features)}")
        
        return df[available_features], available_features
    
    @staticmethod
    def train_baseline_model(X_train: pd.DataFrame, y_train: pd.Series,
                           model_type: str = 'rf') -> object:
        """
        Huấn luyện mô hình baseline
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Features train
        y_train : pd.Series
            Target train
        model_type : str
            Loại mô hình ('rf' hoặc 'lr')
            
        Returns:
        --------
        object
            Mô hình đã huấn luyện
        """
        if model_type == 'rf':
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'lr':
            model = LinearRegression()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.fit(X_train, y_train)
        print(f"{model_type.upper()} model trained")
        
        return model
    
    @staticmethod
    def evaluate_model(model: object, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Đánh giá mô hình
        
        Parameters:
        -----------
        model : object
            Mô hình đã huấn luyện
        X_test : pd.DataFrame
            Features test
        y_test : pd.Series
            Target test
            
        Returns:
        --------
        Dict
            Dictionary chứa các metrics
        """
        # Dự đoán
        y_pred = model.predict(X_test)
        
        # Tính metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # Feature importance (nếu là Random Forest)
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X_test.columns, model.feature_importances_))
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'feature_importance': feature_importance,
            'predictions': y_pred
        }