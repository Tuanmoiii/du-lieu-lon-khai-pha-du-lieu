"""
Thư viện cho phân tích chuỗi thời gian và ARIMA/SARIMA
"""
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import itertools
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesAnalyzer:
    """Class phân tích chuỗi thời gian"""
    
    @staticmethod
    def prepare_time_series(df: pd.DataFrame, station: str = None) -> pd.Series:
        """
        Chuẩn bị chuỗi thời gian từ DataFrame
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame chứa dữ liệu
        station : str, optional
            Tên trạm, nếu None thì lấy trung bình tất cả trạm
            
        Returns:
        --------
        pd.Series
            Chuỗi thời gian PM2.5
        """
        if station:
            # Lấy dữ liệu một trạm
            station_df = df[df['station'] == station].copy()
            ts = station_df.set_index('datetime')['pm2.5']
        else:
            # Lấy trung bình tất cả trạm
            ts = df.groupby('datetime')['pm2.5'].mean()
        
        # Đảm bảo tần suất đều đặn
        ts = ts.asfreq('H')
        
        print(f"Time series prepared")
        print(f"- Length: {len(ts)}")
        print(f"- Frequency: {ts.index.freq}")
        print(f"- Missing values: {ts.isnull().sum()}")
        
        return ts
    
    @staticmethod
    def check_stationarity(series: pd.Series, significance: float = 0.05) -> Dict:
        """
        Kiểm định tính dừng của chuỗi thời gian
        
        Parameters:
        -----------
        series : pd.Series
            Chuỗi thời gian
        significance : float
            Ngưỡng ý nghĩa thống kê
            
        Returns:
        --------
        Dict
            Kết quả kiểm định
        """
        # Loại bỏ NaN
        series_clean = series.dropna()
        
        # ADF Test
        adf_result = adfuller(series_clean)
        adf_pvalue = adf_result[1]
        adf_stationary = adf_pvalue < significance
        
        # KPSS Test
        kpss_result = kpss(series_clean, regression='c')
        kpss_pvalue = kpss_result[1]
        kpss_stationary = kpss_pvalue > significance
        
        print("Stationarity Test Results:")
        print(f"ADF test p-value: {adf_pvalue:.4f} - {'Stationary' if adf_stationary else 'Non-stationary'}")
        print(f"KPSS test p-value: {kpss_pvalue:.4f} - {'Stationary' if kpss_stationary else 'Non-stationary'}")
        
        # Quyết định d
        if adf_stationary and kpss_stationary:
            d = 0  # Đã dừng
            recommendation = "Series is stationary, use d=0"
        elif not adf_stationary and not kpss_stationary:
            d = 1  # Cần differencing
            recommendation = "Series is non-stationary, try d=1"
        else:
            d = 1  # Mâu thuẫn, ưu tiên differencing
            recommendation = "Conflicting results, try d=1"
        
        return {
            'adf_pvalue': adf_pvalue,
            'kpss_pvalue': kpss_pvalue,
            'adf_stationary': adf_stationary,
            'kpss_stationary': kpss_stationary,
            'recommended_d': d,
            'recommendation': recommendation
        }
    
    @staticmethod
    def plot_acf_pacf(series: pd.Series, lags: int = 50):
        """
        Vẽ ACF và PACF plots
        
        Parameters:
        -----------
        series : pd.Series
            Chuỗi thời gian
        lags : int
            Số lag để hiển thị
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # ACF plot
        plot_acf(series.dropna(), lags=lags, ax=axes[0], alpha=0.05)
        axes[0].set_title('Autocorrelation Function (ACF)')
        
        # PACF plot
        plot_pacf(series.dropna(), lags=lags, ax=axes[1], alpha=0.05, method='ywm')
        axes[1].set_title('Partial Autocorrelation Function (PACF)')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def suggest_arima_params(series: pd.Series, d: int = None) -> Dict:
        """
        Gợi ý tham số p và q từ ACF/PACF
        
        Parameters:
        -----------
        series : pd.Series
            Chuỗi thời gian
        d : int, optional
            Bậc differencing đã áp dụng
            
        Returns:
        --------
        Dict
            Gợi ý tham số
        """
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        import matplotlib.pyplot as plt
        
        # Nếu có d, áp dụng differencing
        if d and d > 0:
            series_diff = series.diff(d).dropna()
        else:
            series_diff = series.dropna()
        
        # Tính ACF và PACF
        from statsmodels.tsa.stattools import acf, pacf
        
        nlags = 40
        acf_vals = acf(series_diff, nlags=nlags)
        pacf_vals = pacf(series_diff, nlags=nlags)
        
        # Tìm lag mà ACF cắt (|ACF| < 2/sqrt(n))
        n = len(series_diff)
        cutoff = 2 / np.sqrt(n)
        
        # Gợi ý q (từ ACF)
        q_suggestion = None
        for lag in range(1, min(20, nlags)):
            if abs(acf_vals[lag]) < cutoff:
                q_suggestion = lag - 1
                break
        
        # Gợi ý p (từ PACF)
        p_suggestion = None
        for lag in range(1, min(20, nlags)):
            if abs(pacf_vals[lag]) < cutoff:
                p_suggestion = lag - 1
                break
        
        suggestions = {
            'p_suggestion': p_suggestion if p_suggestion else 1,
            'q_suggestion': q_suggestion if q_suggestion else 1,
            'cutoff_value': cutoff
        }
        
        print("ARIMA Parameter Suggestions:")
        print(f"p (AR order): {suggestions['p_suggestion']}")
        print(f"q (MA order): {suggestions['q_suggestion']}")
        
        # Vẽ để minh họa
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        plot_acf(series_diff, lags=nlags, ax=axes[0], alpha=0.05)
        axes[0].axhline(y=cutoff, color='r', linestyle='--', alpha=0.5)
        axes[0].axhline(y=-cutoff, color='r', linestyle='--', alpha=0.5)
        axes[0].set_title('ACF for Parameter Selection')
        
        plot_pacf(series_diff, lags=nlags, ax=axes[1], alpha=0.05, method='ywm')
        axes[1].axhline(y=cutoff, color='r', linestyle='--', alpha=0.5)
        axes[1].axhline(y=-cutoff, color='r', linestyle='--', alpha=0.5)
        axes[1].set_title('PACF for Parameter Selection')
        
        plt.tight_layout()
        plt.show()
        
        return suggestions


class ARIMAModeler:
    """Class để xây dựng và đánh giá mô hình ARIMA"""
    
    @staticmethod
    def grid_search_arima(train_series: pd.Series, 
                         p_values: List[int], 
                         d_values: List[int], 
                         q_values: List[int]) -> Dict:
        """
        Grid search để tìm tham số ARIMA tối ưu
        
        Parameters:
        -----------
        train_series : pd.Series
            Chuỗi train
        p_values : List[int]
            Danh sách giá trị p
        d_values : List[int]
            Danh sách giá trị d
        q_values : List[int]
            Danh sách giá trị q
            
        Returns:
        --------
        Dict
            Kết quả grid search
        """
        best_aic = np.inf
        best_bic = np.inf
        best_params = None
        best_model = None
        
        results = []
        
        print("Starting ARIMA grid search...")
        
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    try:
                        model = ARIMA(train_series, order=(p, d, q))
                        model_fit = model.fit()
                        
                        aic = model_fit.aic
                        bic = model_fit.bic
                        
                        results.append({
                            'p': p, 'd': d, 'q': q,
                            'AIC': aic, 'BIC': bic
                        })
                        
                        # Chọn theo AIC
                        if aic < best_aic:
                            best_aic = aic
                            best_bic = bic
                            best_params = (p, d, q)
                            best_model = model_fit
                            
                            print(f"New best: ARIMA{best_params} - AIC: {aic:.2f}, BIC: {bic:.2f}")
                    
                    except Exception as e:
                        continue
        
        # Sắp xếp kết quả theo AIC
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('AIC').reset_index(drop=True)
        
        print(f"\nGrid search completed. Best model: ARIMA{best_params}")
        print(f"Best AIC: {best_aic:.2f}, Best BIC: {best_bic:.2f}")
        
        return {
            'best_model': best_model,
            'best_params': best_params,
            'best_aic': best_aic,
            'best_bic': best_bic,
            'all_results': results_df
        }
    
    @staticmethod
    def train_arima(train_series: pd.Series, order: Tuple[int, int, int]) -> object:
        """
        Huấn luyện mô hình ARIMA
        
        Parameters:
        -----------
        train_series : pd.Series
            Chuỗi train
        order : Tuple[int, int, int]
            Tham số (p, d, q)
            
        Returns:
        --------
        object
            Mô hình ARIMA đã huấn luyện
        """
        model = ARIMA(train_series, order=order)
        model_fit = model.fit()
        
        print(f"ARIMA{order} model trained")
        print(model_fit.summary())
        
        return model_fit
    
    @staticmethod
    def forecast_arima(model_fit: object, test_series: pd.Series, 
                      steps_ahead: int = 1) -> pd.Series:
        """
        Dự báo với mô hình ARIMA
        
        Parameters:
        -----------
        model_fit : object
            Mô hình ARIMA đã huấn luyện
        test_series : pd.Series
            Chuỗi test (để biết độ dài)
        steps_ahead : int
            Số bước dự báo trước mỗi lần
            
        Returns:
        --------
        pd.Series
            Dự báo
        """
        # One-step ahead forecast
        forecast = model_fit.forecast(steps=len(test_series))
        
        # Đảm bảo index khớp với test
        forecast.index = test_series.index
        
        return forecast
    
    @staticmethod
    def evaluate_forecast(actual: pd.Series, forecast: pd.Series) -> Dict:
        """
        Đánh giá dự báo
        
        Parameters:
        -----------
        actual : pd.Series
            Giá trị thực tế
        forecast : pd.Series
            Giá trị dự báo
            
        Returns:
        --------
        Dict
            Các metrics đánh giá
        """
        # Loại bỏ NaN
        valid_idx = actual.notna() & forecast.notna()
        actual_valid = actual[valid_idx]
        forecast_valid = forecast[valid_idx]
        
        # Tính metrics
        mae = mean_absolute_error(actual_valid, forecast_valid)
        rmse = np.sqrt(mean_squared_error(actual_valid, forecast_valid))
        mape = np.mean(np.abs((actual_valid - forecast_valid) / actual_valid)) * 100
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'forecast': forecast
        }


class SARIMAModeler(ARIMAModeler):
    """Class kế thừa ARIMAModeler cho SARIMA"""
    
    @staticmethod
    def train_sarima(train_series: pd.Series, 
                    order: Tuple[int, int, int],
                    seasonal_order: Tuple[int, int, int, int]) -> object:
        """
        Huấn luyện mô hình SARIMA
        
        Parameters:
        -----------
        train_series : pd.Series
            Chuỗi train
        order : Tuple[int, int, int]
            (p, d, q)
        seasonal_order : Tuple[int, int, int, int]
            (P, D, Q, s)
            
        Returns:
        --------
        object
            Mô hình SARIMA đã huấn luyện
        """
        model = SARIMAX(
            train_series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        model_fit = model.fit(disp=False)
        
        print(f"SARIMA{order}{seasonal_order} model trained")
        print(model_fit.summary())
        
        return model_fit
    
    @staticmethod
    def grid_search_sarima(train_series: pd.Series,
                          p_values: List[int],
                          d_values: List[int],
                          q_values: List[int],
                          P_values: List[int],
                          D_values: List[int],
                          Q_values: List[int],
                          s_values: List[int]) -> Dict:
        """
        Grid search cho SARIMA
        """
        best_aic = np.inf
        best_params = None
        best_model = None
        
        results = []
        
        print("Starting SARIMA grid search...")
        total_combinations = len(p_values) * len(d_values) * len(q_values) * \
                           len(P_values) * len(D_values) * len(Q_values) * len(s_values)
        print(f"Total combinations: {total_combinations}")
        
        counter = 0
        
        # Giới hạn để không quá nhiều
        for p in p_values[:2]:  # Giới hạn
            for d in d_values:
                for q in q_values[:2]:
                    for P in P_values[:2]:
                        for D in D_values:
                            for Q in Q_values[:2]:
                                for s in s_values:
                                    counter += 1
                                    try:
                                        model = SARIMAX(
                                            train_series,
                                            order=(p, d, q),
                                            seasonal_order=(P, D, Q, s),
                                            enforce_stationarity=False,
                                            enforce_invertibility=False
                                        )
                                        
                                        model_fit = model.fit(disp=False, maxiter=50)
                                        aic = model_fit.aic
                                        
                                        results.append({
                                            'p': p, 'd': d, 'q': q,
                                            'P': P, 'D': D, 'Q': Q, 's': s,
                                            'AIC': aic
                                        })
                                        
                                        if aic < best_aic:
                                            best_aic = aic
                                            best_params = ((p, d, q), (P, D, Q, s))
                                            best_model = model_fit
                                            
                                            print(f"[{counter}] New best: SARIMA{best_params} - AIC: {aic:.2f}")
                                    
                                    except Exception as e:
                                        continue
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('AIC').reset_index(drop=True)
        
        print(f"\nGrid search completed. Best model: SARIMA{best_params}")
        print(f"Best AIC: {best_aic:.2f}")
        
        return {
            'best_model': best_model,
            'best_params': best_params,
            'best_aic': best_aic,
            'all_results': results_df
        }