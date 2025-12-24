import papermill as pm
import datetime

def run_pipeline():
    """Chạy toàn bộ pipeline từng bước một"""
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Bước 1: Tiền xử lý và EDA
    print("="*60)
    print("BƯỚC 1: TIỀN XỬ LÝ VÀ EDA")
    print("="*60)
    
    pm.execute_notebook(
        'notebooks/preprocessing_and_eda.ipynb',
        f'notebooks/output/preprocessing_and_eda_{timestamp}.ipynb',
        parameters={}
    )
    
    # Bước 2: Chuẩn bị basket
    print("\n" + "="*60)
    print("BƯỚC 2: CHUẨN BỊ BASKET")
    print("="*60)
    
    pm.execute_notebook(
        'notebooks/basket_preparation.ipynb',
        f'notebooks/output/basket_preparation_{timestamp}.ipynb',
        parameters={}
    )
    
    # Bước 3: Khai phá luật với Apriori
    print("\n" + "="*60)
    print("BƯỚC 3: KHAI PHÁ LUẬT VỚI APRIORI")
    print("="*60)
    
    pm.execute_notebook(
        'notebooks/apriori_modelling.ipynb',
        f'notebooks/output/apriori_modelling_{timestamp}.ipynb',
        parameters={
            'min_support': 0.01,
            'min_confidence': 0.3,
            'min_lift': 1.2
        }
    )
    
    print("\n" + "="*60)
    print("PIPELINE HOÀN THÀNH!")
    print("="*60)
    print(f"Kết quả đã được lưu với timestamp: {timestamp}")

if __name__ == "__main__":
    run_pipeline()