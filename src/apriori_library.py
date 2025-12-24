"""
apriori_library.py - Thư viện khai phá luật kết hợp
"""

import pandas as pd
import numpy as np

# ====================== CLASS DataCleaner ======================
class DataCleaner:
    """Làm sạch dữ liệu giao dịch bán lẻ"""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = None
    
    def load_data(self):
        """Tải dữ liệu từ file CSV"""
        self.df = pd.read_csv(self.filepath, encoding='unicode_escape')
        print(f"Dữ liệu gốc: {self.df.shape[0]} hàng, {self.df.shape[1]} cột")
        return self.df
    
    def clean_data(self):
        """Làm sạch dữ liệu"""
        if self.df is None:
            self.load_data()
        
        # Loại bỏ hàng có InvoiceNo bắt đầu bằng 'C' (hủy đơn)
        self.df = self.df[~self.df['InvoiceNo'].astype(str).str.startswith('C')]
        
        # Loại bỏ hàng có Quantity <= 0
        self.df = self.df[self.df['Quantity'] > 0]
        
        # Loại bỏ hàng có UnitPrice <= 0
        self.df = self.df[self.df['UnitPrice'] > 0]
        
        # Loại bỏ hàng có StockCode không hợp lệ
        invalid_codes = ['POST', 'DOT', 'M', 'D', 'C2', 'BANK CHARGES', 'PADS']
        self.df = self.df[~self.df['StockCode'].isin(invalid_codes)]
        
        print(f"Sau khi làm sạch: {self.df.shape[0]} hàng, {self.df.shape[1]} cột")
        return self.df
    
    def prepare_basket_bool(self, country: str = 'United Kingdom'):
        """Chuẩn bị ma trận basket boolean"""
        # Lọc theo quốc gia
        if country:
            df_country = self.df[self.df['Country'] == country].copy()
        else:
            df_country = self.df.copy()
        
        # Tạo cột Description sạch
        df_country['Description'] = df_country['Description'].str.strip().str.upper()
        
        # Tạo basket
        basket = (df_country.groupby(['InvoiceNo', 'Description'])['Quantity']
                 .sum().unstack().reset_index().fillna(0)
                 .set_index('InvoiceNo'))
        
        # Chuyển đổi sang boolean (1 nếu mua, 0 nếu không)
        basket_bool = basket.applymap(lambda x: 1 if x > 0 else 0)
        
        print(f"Basket boolean: {basket_bool.shape[0]} hóa đơn, {basket_bool.shape[1]} sản phẩm")
        return basket_bool

# ====================== CLASS AssociationRulesMiner ======================
class AssociationRulesMiner:
    """Triển khai thuật toán Apriori"""
    
    def __init__(self, basket_bool):
        self.basket_bool = basket_bool
        self.frequent_itemsets = None
        self.rules = None
    
    def find_frequent_itemsets(self, min_support=0.01):
        """Tìm tập phổ biến bằng Apriori"""
        from mlxtend.frequent_patterns import apriori
        
        self.frequent_itemsets = apriori(
            self.basket_bool, 
            min_support=min_support, 
            use_colnames=True,
            max_len=10
        )
        
        print(f"Apriori: {len(self.frequent_itemsets)} tập phổ biến")
        return self.frequent_itemsets
    
    def generate_rules(self, min_confidence=0.3, min_lift=1.2):
        """Sinh luật kết hợp"""
        if self.frequent_itemsets is None:
            raise ValueError("Chạy find_frequent_itemsets() trước!")
        
        from mlxtend.frequent_patterns import association_rules
        
        self.rules = association_rules(
            self.frequent_itemsets, 
            metric="confidence", 
            min_threshold=min_confidence
        )
        
        # Lọc theo lift
        self.rules = self.rules[self.rules['lift'] >= min_lift]
        
        # Sắp xếp
        self.rules = self.rules.sort_values('lift', ascending=False)
        
        print(f"Apriori: {len(self.rules)} luật")
        return self.rules
    
    def save_rules(self, filepath: str):
        """Lưu luật kết hợp ra file CSV"""
        if self.rules is None:
            raise ValueError("Chưa có luật để lưu! Hãy chạy generate_rules() trước.")
        
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        self.rules.to_csv(filepath, index=False)
        print(f"✅ Đã lưu {len(self.rules)} luật Apriori vào: {filepath}")
# ====================== THÊM CLASS FPGrowthMiner ======================
class FPGrowthMiner:
    """Triển khai thuật toán FP-Growth"""
    
    def __init__(self, basket_bool):
        self.basket_bool = basket_bool
        self.frequent_itemsets = None
        self.rules = None
    
    def find_frequent_itemsets(self, min_support=0.01):
        """Tìm tập phổ biến bằng FP-Growth"""
        from mlxtend.frequent_patterns import fpgrowth
        
        self.frequent_itemsets = fpgrowth(
            self.basket_bool, 
            min_support=min_support, 
            use_colnames=True,
            max_len=10
        )
        
        print(f"FP-Growth: {len(self.frequent_itemsets)} tập phổ biến")
        return self.frequent_itemsets
    
    def generate_rules(self, min_confidence=0.3, min_lift=1.2):
        """Sinh luật kết hợp"""
        if self.frequent_itemsets is None:
            raise ValueError("Chạy find_frequent_itemsets() trước!")
        
        from mlxtend.frequent_patterns import association_rules
        
        self.rules = association_rules(
            self.frequent_itemsets, 
            metric="confidence", 
            min_threshold=min_confidence
        )
        
        # Lọc theo lift
        self.rules = self.rules[self.rules['lift'] >= min_lift]
        
        # Sắp xếp
        self.rules = self.rules.sort_values('lift', ascending=False)
        
        print(f"FP-Growth: {len(self.rules)} luật")
        return self.rules  # <-- THÊM DÒNG NÀY VÀ DẤU NGOẶC ĐÓNG
    
    def save_rules(self, filepath: str):
        """Lưu luật kết hợp ra file CSV"""
        if self.rules is None:
            raise ValueError("Chưa có luật để lưu! Hãy chạy generate_rules() trước.")
        
        # Đảm bảo thư mục tồn tại
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Lưu file
        self.rules.to_csv(filepath, index=False)
        print(f"✅ Đã lưu {len(self.rules)} luật FP-Growth vào: {filepath}")
# ====================== THÊM CLASS DataVisualizer ======================
class DataVisualizer:
    """Trực quan hóa dữ liệu"""
    
    def plot_top_products(self, basket_bool, top_n=20):
        """Vẽ top sản phẩm bán chạy"""
        import matplotlib.pyplot as plt
        
        product_counts = basket_bool.sum().sort_values(ascending=False).head(top_n)
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(product_counts)), product_counts.values)
        plt.xticks(range(len(product_counts)), product_counts.index, rotation=45, ha='right')
        plt.title(f'Top {top_n} sản phẩm bán chạy')
        plt.xlabel('Sản phẩm')
        plt.ylabel('Số hóa đơn')
        plt.tight_layout()
        plt.show()
    
    def plot_rules(self, rules, top_n=10):
        """Vẽ biểu đồ luật kết hợp"""
        import matplotlib.pyplot as plt
        
        if len(rules) == 0:
            print("Không có luật")
            return
        
        top_rules = rules.head(top_n)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Lift
        axes[0].barh(range(len(top_rules)), top_rules['lift'].values)
        axes[0].set_yticks(range(len(top_rules)))
        axes[0].set_title('Top luật theo Lift')
        axes[0].invert_yaxis()
        
        # Confidence
        axes[1].barh(range(len(top_rules)), top_rules['confidence'].values)
        axes[1].set_yticks(range(len(top_rules)))
        axes[1].set_title('Top luật theo Confidence')
        axes[1].invert_yaxis()
        
        # Support vs Confidence
        axes[2].scatter(rules['support'], rules['confidence'], alpha=0.5)
        axes[2].set_xlabel('Support')
        axes[2].set_ylabel('Confidence')
        axes[2].set_title('Support vs Confidence')
        
        plt.tight_layout()
        plt.show()
    def plot_top_rules(self, rules, top_n=15, metric='lift', figsize=(14, 8)):
        """Vẽ biểu đồ top luật theo metric (lift, confidence, support)"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        if len(rules) == 0:
            print("Không có luật để hiển thị")
            return
        
        # Lấy top luật theo metric
        top_rules = rules.sort_values(metric, ascending=False).head(top_n)
        
        # Tạo nhãn cho các luật
        rule_labels = []
        for _, rule in top_rules.iterrows():
            antecedents = ', '.join([str(i)[:15] for i in list(rule['antecedents'])])
            consequents = ', '.join([str(i)[:15] for i in list(rule['consequents'])])
            label = f"{antecedents} → {consequents}"
            if len(label) > 60:
                label = label[:57] + "..."
            rule_labels.append(label)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Biểu đồ chính: Top rules theo metric
        x_pos = np.arange(len(top_rules))
        if metric == 'lift':
            bar_color = 'lightcoral'
            title_suffix = 'LIFT'
        elif metric == 'confidence':
            bar_color = 'skyblue'
            title_suffix = 'CONFIDENCE'
        else:
            bar_color = 'lightgreen'
            title_suffix = metric.upper()
        
        bars = axes[0, 0].barh(x_pos, top_rules[metric].values, color=bar_color)
        axes[0, 0].set_yticks(x_pos)
        axes[0, 0].set_yticklabels(rule_labels, fontsize=9)
        axes[0, 0].set_xlabel(metric.capitalize())
        axes[0, 0].set_title(f'Top {top_n} luật theo {title_suffix}', fontweight='bold')
        axes[0, 0].invert_yaxis()
        
        # Thêm giá trị lên cột
        for bar, value in zip(bars, top_rules[metric].values):
            if metric == 'lift':
                text = f'{value:.2f}'
            elif metric == 'confidence':
                text = f'{value:.3f}'
            else:
                text = f'{value:.4f}'
            
            axes[0, 0].text(
                bar.get_width() * 1.01, 
                bar.get_y() + bar.get_height()/2, 
                text, 
                va='center', 
                fontsize=9,
                fontweight='bold'
            )
        
        # 2. Biểu đồ scatter: Lift vs Confidence
        scatter = axes[0, 1].scatter(
            top_rules['lift'], 
            top_rules['confidence'], 
            s=top_rules['support']*5000, 
            alpha=0.7,
            c=top_rules['lift'], 
            cmap='RdYlGn'
        )
        axes[0, 1].set_xlabel('Lift')
        axes[0, 1].set_ylabel('Confidence')
        axes[0, 1].set_title('Mối quan hệ Lift vs Confidence', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Đánh dấu các điểm đặc biệt
        if len(top_rules) > 0:
            max_lift_idx = top_rules['lift'].idxmax()
            max_conf_idx = top_rules['confidence'].idxmax()
            
            axes[0, 1].scatter(
                top_rules.loc[max_lift_idx, 'lift'], 
                top_rules.loc[max_lift_idx, 'confidence'], 
                s=200, marker='*', c='red', label='Lift cao nhất'
            )
            axes[0, 1].scatter(
                top_rules.loc[max_conf_idx, 'lift'], 
                top_rules.loc[max_conf_idx, 'confidence'], 
                s=200, marker='s', c='blue', label='Confidence cao nhất'
            )
            axes[0, 1].legend()
        
        plt.colorbar(scatter, ax=axes[0, 1], label='Lift')
        
        # 3. Biểu đồ cột: So sánh 3 metrics
        metrics_to_compare = ['lift', 'confidence', 'support'][:3]
        x = np.arange(len(top_rules))
        width = 0.25
        
        for i, metric_name in enumerate(metrics_to_compare):
            # Chuẩn hóa giá trị để cùng tỉ lệ
            values = top_rules[metric_name].values
            if metric_name == 'support':
                values = values * 100  # Nhân 100 để dễ nhìn
            
            offset = (i - 1) * width
            axes[1, 0].bar(x + offset, values, width, label=metric_name.capitalize())
        
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(range(1, len(top_rules) + 1))
        axes[1, 0].set_xlabel('Thứ hạng luật')
        if 'support' in metrics_to_compare:
            axes[1, 0].set_ylabel('Support (%) / Lift / Confidence')
        else:
            axes[1, 0].set_ylabel('Giá trị metric')
        axes[1, 0].set_title('So sánh các metrics của top luật', fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. Biểu đồ phân phối metric
        axes[1, 1].hist(rules[metric], bins=20, edgecolor='black', alpha=0.7)
        axes[1, 1].axvline(
            x=rules[metric].mean(), 
            color='red', 
            linestyle='--', 
            linewidth=2,
            label=f"Trung bình: {rules[metric].mean():.3f}"
        )
        axes[1, 1].axvline(
            x=rules[metric].median(), 
            color='green', 
            linestyle=':', 
            linewidth=2,
            label=f"Trung vị: {rules[metric].median():.3f}"
        )
        axes[1, 1].set_xlabel(metric.capitalize())
        axes[1, 1].set_ylabel('Số luật')
        axes[1, 1].set_title(f'Phân phối {metric.capitalize()}', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(
            f'Phân tích {len(rules)} luật kết hợp (Top {top_n} theo {metric})', 
            fontsize=16, 
            fontweight='bold', 
            y=1.02
        )
        plt.tight_layout()
        plt.show()
        
        # In thống kê
        print(f"Tổng số luật: {len(rules)}")
        print(f"{metric.capitalize()} trung bình: {rules[metric].mean():.3f}")
        print(f"{metric.capitalize()} cao nhất: {rules[metric].max():.3f}")
        print(f"{metric.capitalize()} thấp nhất: {rules[metric].min():.3f}")
        print(f"Confidence trung bình: {rules['confidence'].mean():.3f}")
        print(f"Support trung bình: {rules['support'].mean():.4f}")
    def plot_top_itemsets(self, frequent_itemsets, top_n=20):
        """Vẽ biểu đồ top itemsets phổ biến"""
        import matplotlib.pyplot as plt
        
        if len(frequent_itemsets) == 0:
            print("Không có itemset để hiển thị")
            return
        
        # Lấy top itemsets
        top_itemsets = frequent_itemsets.sort_values('support', ascending=False).head(top_n)
        
        # Tạo nhãn ngắn gọn
        labels = []
        for itemset in top_itemsets['itemsets']:
            items = list(itemset)
            if len(items) > 3:
                label = f"{len(items)} items"
            else:
                label = ', '.join([str(i)[:15] for i in items])
            labels.append(label)
        
        # Vẽ biểu đồ
        plt.figure(figsize=(12, 6))
        bars = plt.barh(range(len(top_itemsets)), top_itemsets['support'].values)
        plt.yticks(range(len(top_itemsets)), labels)
        plt.xlabel('Support')
        plt.title(f'Top {top_n} Itemsets phổ biến nhất')
        plt.gca().invert_yaxis()  # Đảo ngược trục y để cao nhất ở trên
        
        # Thêm giá trị
        for bar, support in zip(bars, top_itemsets['support'].values):
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{support:.4f}', va='center')
        
    def plot_scatter_rules(self, rules, figsize=(12, 10)):
        """Vẽ biểu đồ phân tán Support vs Confidence"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        if len(rules) == 0:
            print("Không có luật để hiển thị")
            return
        
        # Tạo figure với 4 subplot
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Scatter chính: Support vs Confidence (màu sắc theo Lift)
        scatter1 = axes[0, 0].scatter(
            rules['support'], 
            rules['confidence'], 
            c=rules['lift'], 
            cmap='viridis',
            s=rules['lift'] * 30,  # Kích thước điểm theo lift
            alpha=0.7,
            edgecolors='w',
            linewidth=0.5
        )
        axes[0, 0].set_xlabel('Support')
        axes[0, 0].set_ylabel('Confidence')
        axes[0, 0].set_title('Support vs Confidence (màu theo Lift)', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Thêm colorbar
        cbar1 = plt.colorbar(scatter1, ax=axes[0, 0])
        cbar1.set_label('Lift')
        
        # Đánh dấu điểm có lift cao nhất
        if len(rules) > 0:
            max_lift_idx = rules['lift'].idxmax()
            axes[0, 0].scatter(
                rules.loc[max_lift_idx, 'support'], 
                rules.loc[max_lift_idx, 'confidence'], 
                s=200, marker='*', c='red', label=f'Lift cao nhất: {rules.loc[max_lift_idx, "lift"]:.2f}'
            )
            axes[0, 0].legend()
        
        # 2. Scatter: Support vs Lift (màu sắc theo Confidence)
        scatter2 = axes[0, 1].scatter(
            rules['support'], 
            rules['lift'], 
            c=rules['confidence'], 
            cmap='plasma',
            s=50,
            alpha=0.7,
            edgecolors='w',
            linewidth=0.5
        )
        axes[0, 1].set_xlabel('Support')
        axes[0, 1].set_ylabel('Lift')
        axes[0, 1].set_title('Support vs Lift (màu theo Confidence)', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=1.0, color='red', linestyle='--', linewidth=1, label='Lift = 1.0')
        
        cbar2 = plt.colorbar(scatter2, ax=axes[0, 1])
        cbar2.set_label('Confidence')
        axes[0, 1].legend()
        
        # 3. Scatter: Confidence vs Lift (màu sắc theo Support)
        scatter3 = axes[1, 0].scatter(
            rules['confidence'], 
            rules['lift'], 
            c=rules['support'], 
            cmap='coolwarm',
            s=50,
            alpha=0.7,
            edgecolors='w',
            linewidth=0.5
        )
        axes[1, 0].set_xlabel('Confidence')
        axes[1, 0].set_ylabel('Lift')
        axes[1, 0].set_title('Confidence vs Lift (màu theo Support)', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=1.0, color='red', linestyle='--', linewidth=1)
        
        cbar3 = plt.colorbar(scatter3, ax=axes[1, 0])
        cbar3.set_label('Support')
        
        # 4. Histogram 3D: Phân phối đồng thời của 3 metrics
        # Tạo biểu đồ hexbin cho Support vs Confidence
        hb = axes[1, 1].hexbin(
            rules['support'], 
            rules['confidence'], 
            gridsize=20, 
            cmap='YlOrRd',
            mincnt=1
        )
        axes[1, 1].set_xlabel('Support')
        axes[1, 1].set_ylabel('Confidence')
        axes[1, 1].set_title('Mật độ phân bố Support vs Confidence', fontweight='bold')
        
        cbar4 = plt.colorbar(hb, ax=axes[1, 1])
        cbar4.set_label('Số lượng luật')
        
        # Tính toán và hiển thị thống kê
        stats_text = (
            f"Tổng số luật: {len(rules)}\n"
            f"Support: {rules['support'].mean():.4f} ± {rules['support'].std():.4f}\n"
            f"Confidence: {rules['confidence'].mean():.3f} ± {rules['confidence'].std():.3f}\n"
            f"Lift: {rules['lift'].mean():.2f} ± {rules['lift'].std():.2f}"
        )
        
        axes[1, 1].text(
            0.02, 0.98, stats_text,
            transform=axes[1, 1].transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
        
        plt.suptitle(
            f'Phân tích phân tán {len(rules)} luật kết hợp', 
            fontsize=16, 
            fontweight='bold', 
            y=1.02
        )
        plt.tight_layout()
        plt.show()
        
        # In thống kê tương quan
        print("Hệ số tương quan giữa các metrics:")
        print(f"  Support vs Confidence: {rules['support'].corr(rules['confidence']):.3f}")
        print(f"  Support vs Lift: {rules['support'].corr(rules['lift']):.3f}")
        print(f"  Confidence vs Lift: {rules['confidence'].corr(rules['lift']):.3f}")


    def plot_network_graph(self, rules, top_n=30, figsize=(14, 10)):
        """Vẽ network graph của các luật kết hợp"""
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            from matplotlib.colors import Normalize
        except ImportError as e:
            print(f"❌ Lỗi: {e}")
            print("Hãy cài đặt: pip install networkx matplotlib")
            return
        
        if len(rules) == 0:
            print("Không có luật để hiển thị")
            return
        
        # Lấy top rules
        top_rules = rules.sort_values('lift', ascending=False).head(top_n)
        
        # Tạo đồ thị
        G = nx.DiGraph()  # Đồ thị có hướng (directed)
        
        # Thêm các node và edges
        node_sizes = {}
        edge_weights = {}
        
        for _, rule in top_rules.iterrows():
            antecedents = list(rule['antecedents'])
            consequents = list(rule['consequents'])
            
            # Thêm các node (sản phẩm)
            for item in antecedents + consequents:
                if item not in G:
                    G.add_node(item, type='product')
                    node_sizes[item] = 0
                node_sizes[item] += 1
            
            # Thêm edge từ antecedents đến consequents
            # Nếu có nhiều antecedents, tạo node trung gian
            if len(antecedents) > 1:
                ante_node = f"{{{','.join(str(a)[:10] for a in antecedents)}}}"
                G.add_node(ante_node, type='itemset')
                node_sizes[ante_node] = len(antecedents)
                
                for ant in antecedents:
                    G.add_edge(ant, ante_node, weight=rule['confidence'], lift=rule['lift'])
                G.add_edge(ante_node, consequents[0], weight=rule['confidence'], lift=rule['lift'])
            else:
                for ant in antecedents:
                    for cons in consequents:
                        G.add_edge(ant, cons, weight=rule['confidence'], lift=rule['lift'])
        
        if len(G.nodes()) == 0:
            print("Không thể tạo network graph")
            return
        
        # Chuẩn bị visualization
        plt.figure(figsize=figsize)
        
        # Layout
        try:
            pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
        except:
            pos = nx.circular_layout(G)
        
        # Chuẩn bị màu sắc và kích thước
        node_colors = []
        node_sizes_list = []
        
        for node in G.nodes():
            node_type = G.nodes[node].get('type', 'product')
            if node_type == 'itemset':
                node_colors.append('lightcoral')  # Màu cho itemset
                node_sizes_list.append(800 + node_sizes.get(node, 1) * 100)
            else:
                node_colors.append('skyblue')  # Màu cho sản phẩm đơn
                node_sizes_list.append(500 + node_sizes.get(node, 1) * 50)
        
        # Chuẩn bị edge colors và widths
        edge_colors = []
        edge_widths = []
        
        for u, v, data in G.edges(data=True):
            lift = data.get('lift', 1.0)
            # Màu sắc theo lift (xanh = lift cao, đỏ = lift thấp)
            if lift > 2.0:
                edge_colors.append('darkgreen')
            elif lift > 1.5:
                edge_colors.append('green')
            elif lift > 1.2:
                edge_colors.append('limegreen')
            else:
                edge_colors.append('lightgray')
            
            # Độ dày theo confidence
            confidence = data.get('weight', 0.5)
            edge_widths.append(confidence * 5)
        
        # Vẽ network
        # 1. Vẽ nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes_list,
            node_color=node_colors,
            alpha=0.9,
            edgecolors='black',
            linewidths=1
        )
        
        # 2. Vẽ edges
        nx.draw_networkx_edges(
            G, pos,
            edge_color=edge_colors,
            width=edge_widths,
            alpha=0.7,
            arrows=True,
            arrowsize=15,
            arrowstyle='->',
            connectionstyle='arc3,rad=0.1'
        )
        
        # 3. Vẽ labels
        node_labels = {}
        for node in G.nodes():
            if G.nodes[node].get('type') == 'itemset':
                # Rút gọn label cho itemset
                if len(str(node)) > 20:
                    # Sửa thành:
                    if '{' in str(node):
                        # Đếm số items bằng cách đếm dấu phẩy + 1
                        items_str = str(node).strip('{}')
                        item_count = items_str.count(',') + 1 if items_str else 0
                        node_labels[node] = f"{{{item_count} items}}"
                    else:
                        node_labels[node] = node
                else:
                    node_labels[node] = node
            else:
                # Rút gọn label cho sản phẩm
                label = str(node)
                if len(label) > 15:
                    node_labels[node] = label[:12] + "..."
                else:
                    node_labels[node] = label
        
        nx.draw_networkx_labels(
            G, pos,
            labels=node_labels,
            font_size=9,
            font_weight='bold'
        )
        
        # Thêm edge labels (lift values)
        edge_labels = {}
        for u, v, data in G.edges(data=True):
            if data.get('lift', 0) > 1.5:  # Chỉ hiển thị lift cao
                edge_labels[(u, v)] = f"{data.get('lift', 0):.2f}"
        
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_size=8,
            font_color='darkred'
        )
        
        # Thêm legend
        from matplotlib.patches import Patch
        
        legend_elements = [
            Patch(facecolor='skyblue', edgecolor='black', label='Sản phẩm đơn'),
            Patch(facecolor='lightcoral', edgecolor='black', label='Tập sản phẩm'),
            Patch(facecolor='darkgreen', edgecolor='darkgreen', label='Lift > 2.0'),
            Patch(facecolor='green', edgecolor='green', label='Lift > 1.5'),
            Patch(facecolor='limegreen', edgecolor='limegreen', label='Lift > 1.2'),
        ]
        
        plt.legend(
            handles=legend_elements,
            loc='upper left',
            bbox_to_anchor=(1.05, 1),
            borderaxespad=0.
        )
        
        # Tiêu đề và thông tin
        plt.title(
            f'Network Graph của {len(top_rules)} luật hàng đầu\n'
            f'(Tổng: {len(G.nodes())} nodes, {len(G.edges())} edges)',
            fontsize=16,
            fontweight='bold'
        )
        
        # Thông tin thống kê
        stats_text = (
            f"• Top {top_n} luật theo Lift\n"
            f"• Lift trung bình: {top_rules['lift'].mean():.2f}\n"
            f"• Confidence trung bình: {top_rules['confidence'].mean():.3f}\n"
            f"• Số node: {len(G.nodes())}\n"
            f"• Số edge: {len(G.edges())}"
        )
        
        plt.figtext(
            0.02, 0.02, stats_text,
            fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
        
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        # In thông tin thêm
        print(f"Đã tạo network graph với {len(G.nodes())} nodes và {len(G.edges())} edges")
        print(f"Top 5 sản phẩm có nhiều kết nối nhất:")
        node_degrees = dict(G.degree())
        sorted_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
        for node, degree in sorted_nodes:
            if G.nodes[node].get('type') == 'product':
                print(f"  • {node}: {degree} kết nối")
    
    
        plt.tight_layout()
        plt.show()
        