import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import networkx as nx
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class DataCleaner:
    def __init__(self, filepath: str):
        """
        Khởi tạo DataCleaner với đường dẫn file dữ liệu
        
        Args:
            filepath (str): Đường dẫn đến file CSV
        """
        self.filepath = filepath
        self.df = None
        self.df_clean = None
        
    def load_data(self):
        """Tải dữ liệu từ file CSV"""
        self.df = pd.read_csv(self.filepath, encoding='unicode_escape')
        print(f"Dữ liệu gốc: {self.df.shape[0]} hàng, {self.df.shape[1]} cột")
        return self.df
    
    def clean_data(self):
        """Thực hiện làm sạch dữ liệu theo các bước đã học"""
        if self.df is None:
            self.load_data()
        
        df = self.df.copy()
        
        # 1. Tạo cột TotalPrice
        df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
        
        # 2. Loại bỏ đơn hàng bị hủy
        df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
        
        # 3. Chỉ giữ dữ liệu từ UK
        df = df[df['Country'] == 'United Kingdom']
        
        # 4. Loại bỏ dòng thiếu CustomerID
        df = df.dropna(subset=['CustomerID'])
        
        # 5. Loại bỏ Quantity hoặc UnitPrice <= 0
        df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
        
        # 6. Chuyển CustomerID về dạng số nguyên
        df['CustomerID'] = df['CustomerID'].astype(int)
        
        self.df_clean = df
        print(f"Dữ liệu đã làm sạch: {self.df_clean.shape[0]} hàng, {self.df_clean.shape[1]} cột")
        return self.df_clean
    
    def create_time_features(self):
        """Tạo các đặc trưng thời gian từ InvoiceDate"""
        if self.df_clean is None:
            self.clean_data()
        
        df = self.df_clean.copy()
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        # Tạo các cột thời gian
        df['Year'] = df['InvoiceDate'].dt.year
        df['Month'] = df['InvoiceDate'].dt.month
        df['Day'] = df['InvoiceDate'].dt.day
        df['Hour'] = df['InvoiceDate'].dt.hour
        df['Weekday'] = df['InvoiceDate'].dt.weekday
        df['DayName'] = df['InvoiceDate'].dt.day_name()
        
        self.df_clean = df
        return self.df_clean


class BasketPreparer:
    def __init__(self, df: pd.DataFrame):
        """
        Khởi tạo BasketPreparer với DataFrame đã làm sạch
        
        Args:
            df (pd.DataFrame): DataFrame đã được làm sạch
        """
        self.df = df
        self.basket = None
        self.basket_bool = None
        
    def create_basket(self):
        """Tạo basket dạng pivot table"""
        # Nhóm theo InvoiceNo và Description
        basket = self.df.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().fillna(0)
        self.basket = basket
        print(f"Kích thước basket: {basket.shape}")
        return self.basket
    
    def encode_basket(self, threshold: int = 1):
        """Mã hóa basket thành định dạng boolean"""
        if self.basket is None:
            self.create_basket()
        
        # Chuyển thành 1/0 (có mặt hay không)
        basket_bool = self.basket.applymap(lambda x: 1 if x >= threshold else 0)
        basket_bool = basket_bool.astype(bool)
        self.basket_bool = basket_bool
        return self.basket_bool
    
    def save_basket(self, filepath: str = 'data/basket_bool.parquet'):
        """Lưu basket_bool ra file"""
        if self.basket_bool is None:
            self.encode_basket()
        
        self.basket_bool.to_parquet(filepath)
        print(f"Đã lưu basket vào: {filepath}")


class AssociationRulesMiner:
    def __init__(self, basket_bool: pd.DataFrame):
        """
        Khởi tạo với basket boolean
        
        Args:
            basket_bool (pd.DataFrame): Ma trận basket đã mã hóa
        """
        self.basket_bool = basket_bool
        self.frequent_itemsets = None
        self.rules = None
    
    def find_frequent_itemsets(self, min_support: float = 0.01):
        """Tìm các tập phổ biến bằng thuật toán Apriori"""
        self.frequent_itemsets = apriori(
            self.basket_bool, 
            min_support=min_support, 
            use_colnames=True,
            max_len=10
        )
        print(f"Tìm thấy {len(self.frequent_itemsets)} tập phổ biến")
        return self.frequent_itemsets
    
    def generate_rules(self, min_confidence: float = 0.3, min_lift: float = 1.2):
        """Sinh luật kết hợp từ các tập phổ biến"""
        if self.frequent_itemsets is None:
            raise ValueError("Cần chạy find_frequent_itemsets trước!")
        
        self.rules = association_rules(
            self.frequent_itemsets, 
            metric="confidence", 
            min_threshold=min_confidence
        )
        
        # Lọc theo lift
        self.rules = self.rules[self.rules['lift'] >= min_lift]
        
        # Sắp xếp theo lift giảm dần
        self.rules = self.rules.sort_values('lift', ascending=False).reset_index(drop=True)
        
        print(f"Sinh được {len(self.rules)} luật kết hợp")
        return self.rules
    
    def save_rules(self, filepath: str = 'data/association_rules.csv'):
        """Lưu luật ra file CSV"""
        if self.rules is None:
            raise ValueError("Chưa có luật để lưu!")
        
        self.rules.to_csv(filepath, index=False)
        print(f"Đã lưu {len(self.rules)} luật vào: {filepath}")


class DataVisualizer:
    def __init__(self):
        """Khởi tạo DataVisualizer"""
        pass
    
    def plot_top_itemsets(self, frequent_itemsets: pd.DataFrame, top_n: int = 20):
        """Vẽ biểu đồ top itemset phổ biến"""
        plt.figure(figsize=(12, 6))
        
        # Lấy top N itemset
        top_items = frequent_itemsets.nlargest(top_n, 'support')
        
        # Chuyển itemset thành string
        top_items['itemset_str'] = top_items['itemsets'].apply(lambda x: ', '.join(list(x)))
        
        # Vẽ bar chart
        plt.barh(range(len(top_items)), top_items['support'])
        plt.yticks(range(len(top_items)), top_items['itemset_str'])
        plt.xlabel('Support')
        plt.title(f'Top {top_n} Itemset Phổ Biến Nhất')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def plot_top_rules(self, rules: pd.DataFrame, top_n: int = 15, metric: str = 'lift'):
        """Vẽ biểu đồ top luật theo metric"""
        plt.figure(figsize=(12, 8))
        
        # Lấy top N luật
        top_rules = rules.nlargest(top_n, metric)
        
        # Tạo label cho từng luật
        labels = []
        for _, row in top_rules.iterrows():
            antecedents = ', '.join(list(row['antecedents']))
            consequents = ', '.join(list(row['consequents']))
            labels.append(f"{antecedents} → {consequents}")
        
        # Vẽ bar chart
        y_pos = np.arange(len(top_rules))
        plt.barh(y_pos, top_rules[metric])
        plt.yticks(y_pos, labels)
        plt.xlabel(metric.capitalize())
        plt.title(f'Top {top_n} Luật Kết Hợp Theo {metric.capitalize()}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def plot_scatter_rules(self, rules: pd.DataFrame):
        """Vẽ scatter plot support vs confidence"""
        plt.figure(figsize=(10, 6))
        
        plt.scatter(rules['support'], rules['confidence'], 
                   c=rules['lift'], cmap='viridis', alpha=0.6, s=100)
        
        plt.colorbar(label='Lift')
        plt.xlabel('Support')
        plt.ylabel('Confidence')
        plt.title('Biểu Đồ Phân Tán: Support vs Confidence (Màu: Lift)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_network_graph(self, rules: pd.DataFrame, top_n: int = 30):
        """Vẽ network graph của các luật"""
        if len(rules) == 0:
            print("Không có luật để vẽ network graph")
            return
        
        # Lấy top N luật
        top_rules = rules.head(top_n)
        
        # Tạo đồ thị
        G = nx.DiGraph()
        
        # Thêm các cạnh
        for _, row in top_rules.iterrows():
            antecedent = list(row['antecedents'])[0] if len(row['antecedents']) == 1 else str(row['antecedents'])
            consequent = list(row['consequents'])[0] if len(row['consequents']) == 1 else str(row['consequents'])
            
            G.add_edge(antecedent, consequent, 
                      weight=row['lift'], 
                      support=row['support'])
        
        # Tính layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Vẽ
        plt.figure(figsize=(14, 10))
        
        # Tính kích thước node dựa trên degree
        node_sizes = [2000 * (G.degree(node) + 1) for node in G.nodes()]
        
        # Vẽ nodes
        nx.draw_networkx_nodes(G, pos, 
                              node_size=node_sizes,
                              node_color='lightblue',
                              alpha=0.8)
        
        # Vẽ edges với độ dày theo lift
        edge_weights = [G[u][v]['weight'] * 2 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, 
                              width=edge_weights,
                              alpha=0.5,
                              edge_color='gray',
                              arrows=True,
                              arrowsize=15)
        
        # Vẽ labels
        nx.draw_networkx_labels(G, pos, 
                               font_size=10,
                               font_weight='bold')
        
        plt.title(f'Network Graph của Top {top_n} Luật Kết Hợp\n(Độ dày cạnh ~ Lift, Kích thước node ~ Degree)', 
                 fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        # In thông tin về các node quan trọng
        print("\n=== THÔNG TIN CÁC SẢN PHẨM QUAN TRỌNG ===")
        
        # Tính degree centrality
        degree_centrality = nx.degree_centrality(G)
        sorted_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
        
        print("\nTop 5 sản phẩm có nhiều kết nối nhất (Hub):")
        for node, centrality in sorted_degree[:5]:
            print(f"  • {node}: Degree Centrality = {centrality:.3f}")