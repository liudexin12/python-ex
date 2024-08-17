import pandas as pd
from typing import List, Tuple, Dict
from datetime import datetime, timedelta
from functools import reduce
import matplotlib.pyplot as plt



# Data Ingestion and Cleaning 数据引入和清理
def load_and_clean_data(file_path: str) -> pd.DataFrame:
    # Reads the CSV file using pandas
    df = pd.read_csv(file_path)
    
    # Converts the 'date' column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Removes any rows with missing values
    df = df.dropna()
    
    # Ensures 'quantity' is integer and 'price' is float
    df['quantity'] = df['quantity'].astype(int)
    df['price'] = df['price'].astype(float)
    
    # Adds a 'total_sale' column (quantity * price)
    df['total_sale'] = df['quantity'] * df['price']
    
    # Returns the cleaned DataFrame
    return df

# Sales Analysis Class 销售分析类
class SalesAnalyzer:
    # Initialize with the cleaned DataFrame
    def __init__(self, data: pd.DataFrame):
        self.data = data
        
    # Return top N products by total revenue
    def top_products(self, n: int) -> List[Tuple[str, float]]:
        # 按产品分组并计算总销售额
        product_sales = self.data.groupby('product_id')['total_sale'].sum()
        # 按总销售额降序排序并获取前 N 个产品
        top_products = product_sales.sort_values(ascending=False).head(n)
        # 转换为元组列表
        return list(top_products.items())

    # Return a dictionary of total daily sales
    def daily_sales(self) -> Dict[str, float]:
        # 按日期分组并计算总销售额
        daily_sales = self.data.groupby(self.data['date'].dt.date)['total_sale'].sum()
        # 转换为字典
        return daily_sales.to_dict()
    # Return top customers by total spending
    def customer_spending(self, top: int) -> List[Tuple[str, float]]:
        # 按客户分组并计算总销售额
        customer_spending = self.data.groupby('customer_id')['total_sale'].sum()
        # 按总支出降序排序并获取前 N 个客户
        top_customers = customer_spending.sort_values(ascending=False).head(top)
        # 转换为元组列表
        return list(top_customers.items())

# Date Range Iterator 日期范围Iterator
class DateRangeIterator:
    # Initializes with start_date and end_date
    def __init__(self, start_date: str, end_date: str):
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        self.current_date = self.start_date

    # Implements __iter__ and __next__ to yield dates in the range
    def __iter__(self):
        return self

    def __next__(self):
        if self.current_date > self.end_date:
            raise StopIteration
        else:
            current = self.current_date
            self.current_date += timedelta(days=1)
            return current.strftime('%Y-%m-%d')

# Performance Monitoring Decorator 
class SalesAnalyzer:
    # Initialize with the cleaned DataFrame
    def __init__(self, data: pd.DataFrame):
        self.data = data
        
    # Return top N products by total revenue
    def top_products(self, n: int) -> List[Tuple[str, float]]:
        # 按产品分组并计算总销售额
        product_sales = self.data.groupby('product_id')['total_sale'].sum()
        # 按总销售额降序排序并获取前 N 个产品
        top_products = product_sales.sort_values(ascending=False).head(n)
        # 转换为元组列表
        return list(top_products.items())

    # Return a dictionary of total daily sales
    def daily_sales(self) -> Dict[str, float]:
        # 按日期分组并计算总销售额
        daily_sales = self.data.groupby(self.data['date'].dt.date)['total_sale'].sum()
        # 转换为字典
        return daily_sales.to_dict()

    # Return top customers by total spending
    def customer_spending(self, top: int) -> List[Tuple[str, float]]:
        # 按客户分组并计算总销售额
        customer_spending = self.data.groupby('customer_id')['total_sale'].sum()
        # 按总支出降序排序并获取前 N 个客户
        top_customers = customer_spending.sort_values(ascending=False).head(top)
        # 转换为元组列表
        return list(top_customers.items())
      
# filter_high_value_sales(data: pd.DataFrame, threshold: float) -> pd.DataFrame
def filter_high_value_sales(data: pd.DataFrame, threshold: float) -> pd.DataFrame:
    # 使用lambda和filter过滤出total_sale>阈值的行
    filtered_data = data[data['total_sale'] > threshold]
    return filtered_data

# map_monthly_sales(data: pd.DataFrame) -> Dict[str, float]
def map_monthly_sales(data: pd.DataFrame) -> Dict[str, float]:
    data['month'] = data['date'].dt.to_period('M')
    # 计算每月的总销售额
    monthly_sales = data.groupby('month')['total_sale'].sum()
    # 转换为字典
    monthly_sales_dict = monthly_sales.to_dict()
    # 转换为字符串
    monthly_sales_dict = {str(k): v for k, v in monthly_sales_dict.items()}
    return monthly_sales_dict

# Functional Programming
# reduce_to_total_revenue(data: pd.DataFrame) -> float
def reduce_to_total_revenue(data: pd.DataFrame) -> float:
    # 计算 total_sale 列的总和
    total_revenue = reduce(lambda x, y: x + y, data['total_sale'])
    return total_revenue

def generate_sales_charts(data: pd.DataFrame) -> None:
    required_columns = {'date', 'product_id', 'quantity', 'total_sale'}
    if not required_columns.issubset(data.columns):
        raise ValueError(f"列缺失: {required_columns}")

    # 1. Plots monthly sales trend (line chart)
    data['date'] = pd.to_datetime(data['date'])
    monthly_sales = data.resample('M', on='date')['quantity'].sum()
    
    plt.figure(figsize=(10, 6))
    plt.plot(monthly_sales.index, monthly_sales.values, marker='o', linestyle='-')
    plt.title('monthly sales trend')
    plt.xlabel('date')
    plt.ylabel('monthly sales')
    plt.grid(True)
    plt.savefig('monthly_sales_trend.png')
    plt.close()

    # 2. Shows top 5 products by sales (bar chart)
    top_products = data.groupby('product_id')['quantity'].sum().nlargest(5)
    
    plt.figure(figsize=(10, 6))
    top_products.plot(kind='bar')
    plt.title('top 5 products by sales')
    plt.xlabel('products')
    plt.ylabel('sales')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.savefig('top_5_products.png')
    plt.close()

    # 3. Displays customer spending distribution (histogram)
    plt.figure(figsize=(10, 6))
    plt.hist(data['total_sale'], bins=20, edgecolor='black')
    plt.title('customer spending distribution')
    plt.xlabel('spending')
    plt.ylabel('customer')
    plt.grid(True)
    plt.savefig('customer_spending_distribution.png')
    plt.close()

file_path = './large_sales_data.csv'

def main():  
    print("Hello word")
   # homework 问题1
    cleaned_data = load_and_clean_data(file_path)
    # 打印处理结果
    print(cleaned_data.head())

    # homework 问题2 
    analyzer = SalesAnalyzer(cleaned_data)
    # 打印处理结果
    #print(analyzer.top_products(5))
    #print(analyzer.daily_sales())
    #print(analyzer.customer_spending(10))

    # homework 问题3
    # 测试效果
    # for date in  DateRangeIterator('2023-01-01', '2023-01-05'):
    #    print(date)
    
    # homework 问题4
    analyzer = SalesAnalyzer(cleaned_data)
    # 打印测试数据
    # print(analyzer.top_products(5))
    # print(analyzer.daily_sales())
    # print(analyzer.customer_spending(10))

    # homework 问题5 测试程序
    # 过滤出销售额大于100的记录
    # filtered_data = filter_high_value_sales(cleaned_data, 100)
    # print("过滤后的数据:")
    # print(filtered_data)
    # # 计算每月的总销售额
    # monthly_sales = map_monthly_sales(cleaned_data)
    # print("每月总销售额:")
    # print(monthly_sales)
    # # 计算总收入
    # total_revenue = reduce_to_total_revenue(cleaned_data)
    # print("总收入:")
    # print(total_revenue)

    # homework 问题7 
    generate_sales_charts(cleaned_data)



if __name__ == "__main__":
    main()