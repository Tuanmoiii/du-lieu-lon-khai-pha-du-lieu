# SHOPPING CART ANALYSIS 

Dự án sử dụng thuật toán Apriori để phân tích hành vi mua sắm của khách hàng nhằm trả lời các câu hỏi "Khách hàng sẽ mua gì tiếp theo?" hay cụ thể hơn là "Những sản phẩm nào thường xuyên song hành cùng nhau trong một giỏ hàng?"

> Bạn có thể dùng project như một pipeline hoàn chỉnh:  
> *Load dữ liệu → Clean → Basket preparation → Apriori / FP-Growth → Xuất luật + báo cáo*

---

## 📁 Cấu trúc thư mục

shopping_cart_analysis/
├── data/
│ ├── raw/ # Dữ liệu gốc
│ │ └── online_retail.csv
│ └── processed/ # Dữ liệu & output sau xử lý
│ ├── cleaned_uk_data.csv
│ ├── basket_bool.parquet
│ └── rules_apriori_filtered.csv
├── notebooks/ # Notebook phân tích & EDA
│ ├── 01_preprocessing_and_eda.ipynb
│ ├── 02_basket_preparation.ipynb
│ ├── 03_apriori_modeling.ipynb
│ └── runs/ # Notebook đã chạy (output của papermill)
├── src/ # Code Python (library nội bộ)
│ └── shopping_cart_library.py
├── run_papermill.py # Script để chạy toàn bộ pipeline tự động
├── requirements.txt # Các thư viện cần thiết
└── README.md # File hướng dẫn này


---

## Hướng dẫn sử dụng

### 1. Cài đặt môi trường

```bash
git clone <repo_url>
cd shopping_cart_analysis
pip install -r requirements.txt

2. Chuẩn bị dữ liệu

Đặt file gốc online_retail.csv vào data/raw/

Các file xử lý và output sẽ tự động sinh vào data/processed/

Chạy toàn bộ pipeline tự động

python run_papermill.py

Kết quả tự động sinh ra:

data/processed/cleaned_uk_data.csv

data/processed/basket_bool.parquet

data/processed/rules_apriori_filtered.csv

notebook đã chạy:

nằm ở notebooks/runs/...


Thay đổi tham số dễ dàng

Các tham số nằm trong run_papermill.py, ví dụ:

MIN_SUPPORT=0.01
MAX_LEN=3
FILTER_MIN_CONF=0.3
FILTER_MIN_LIFT=1.2

Bạn có thể chạy lại nhiều lần để thử các ngưỡng khác nhau.

Thành phần chính trong project
Component	Mô tả
DataCleaner	Làm sạch dữ liệu, xử lý invoice, số lượng âm, lỗi
BasketPreparer	Tạo ma trận basket (hóa đơn x sản phẩm)
AssociationRulesMiner	Khai phá frequent itemsets & luật kết hợp
Notebook 01	EDA + RFM + phân tích dữ liệu
Notebook 02	Basket matrix
Notebook 03	Apriori + trực quan hóa
run_papermill.py


