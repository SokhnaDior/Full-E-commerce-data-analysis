import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

def generate_sample_data(n_records=1000):
    """Generate sample e-commerce data for analysis"""
    np.random.seed(42)
    
    # Generate dates
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=x) for x in range(365)]
    
    # Create sample data
    data = {
        'order_id': range(1, n_records + 1),
        'order_date': np.random.choice(dates, n_records),
        'customer_id': np.random.randint(1, 201, n_records),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home & Garden', 'Sports'], n_records),
        'product_id': np.random.randint(1, 51, n_records),
        'quantity': np.random.randint(1, 6, n_records),
        'unit_price': np.random.uniform(10, 500, n_records).round(2),
        'payment_method': np.random.choice(['Credit Card', 'PayPal', 'Debit Card', 'Bank Transfer'], n_records),
        'shipping_method': np.random.choice(['Standard', 'Express', 'Next Day'], n_records)
    }
    
    return pd.DataFrame(data)

def clean_and_prepare_data(df):
    """Clean and prepare the data for analysis"""
    # Create copy to avoid modifying original
    df_clean = df.copy()
    
    # Calculate total sale amount
    df_clean['total_amount'] = df_clean['quantity'] * df_clean['unit_price']
    
    # Convert date to datetime if not already
    df_clean['order_date'] = pd.to_datetime(df_clean['order_date'])
    
    # Extract useful date components
    df_clean['month'] = df_clean['order_date'].dt.month
    df_clean['day_of_week'] = df_clean['order_date'].dt.day_name()
    
    # Remove any duplicates
    df_clean = df_clean.drop_duplicates()
    
    # Sort by date
    df_clean = df_clean.sort_values('order_date')
    
    return df_clean

def perform_analysis(df):
    """Perform various analyses on the data"""
    results = {}
    
    # Sales by category
    category_sales = df.groupby('product_category')['total_amount'].agg(['sum', 'count']).round(2)
    results['category_sales'] = category_sales
    
    # Monthly sales trend
    monthly_sales = df.groupby('month')['total_amount'].sum().round(2)
    results['monthly_sales'] = monthly_sales
    
    # Payment method analysis
    payment_analysis = df.groupby('payment_method').agg({
        'total_amount': 'sum',
        'order_id': 'count'
    }).round(2)
    results['payment_analysis'] = payment_analysis
    
    # Customer purchase frequency
    customer_frequency = df.groupby('customer_id')['order_id'].count().describe().round(2)
    results['customer_frequency'] = customer_frequency
    
    # Average order value by shipping method
    shipping_analysis = df.groupby('shipping_method')['total_amount'].agg(['mean', 'count']).round(2)
    results['shipping_analysis'] = shipping_analysis
    
    return results

def generate_visualizations(df, results):
    """Create visualizations for the analysis"""
    # Set style
    plt.style.use('seaborn')
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('E-Commerce Sales Analysis Dashboard', fontsize=16)
    
    # 1. Category Sales
    results['category_sales']['sum'].plot(kind='bar', ax=axes[0,0], color='skyblue')
    axes[0,0].set_title('Sales by Product Category')
    axes[0,0].set_ylabel('Total Sales ($)')
    
    # 2. Monthly Sales Trend
    results['monthly_sales'].plot(kind='line', marker='o', ax=axes[0,1])
    axes[0,1].set_title('Monthly Sales Trend')
    axes[0,1].set_ylabel('Total Sales ($)')
    
    # 3. Payment Method Distribution
    payment_dist = results['payment_analysis']['order_id']
    payment_dist.plot(kind='pie', autopct='%1.1f%%', ax=axes[1,0])
    axes[1,0].set_title('Payment Method Distribution')
    
    # 4. Average Order Value by Shipping Method
    results['shipping_analysis']['mean'].plot(kind='bar', ax=axes[1,1], color='lightgreen')
    axes[1,1].set_title('Average Order Value by Shipping Method')
    axes[1,1].set_ylabel('Average Order Value ($)')
    
    # Adjust layout
    plt.tight_layout()
    return fig

def main():
    """Main function to run the entire analysis"""
    # Generate and prepare data
    raw_data = generate_sample_data()
    clean_data = clean_and_prepare_data(raw_data)
    
    # Perform analysis
    analysis_results = perform_analysis(clean_data)
    
    # Print key insights
    print("\nKey Insights:")
    print("-------------")
    print(f"Total Sales: ${clean_data['total_amount'].sum():,.2f}")
    print(f"Total Orders: {len(clean_data)}")
    print(f"Average Order Value: ${clean_data['total_amount'].mean():,.2f}")
    print(f"Best Selling Category: {analysis_results['category_sales']['count'].idxmax()}")
    print(f"Most Popular Payment Method: {analysis_results['payment_analysis']['order_id'].idxmax()}")
    
    # Generate visualizations
    plot = generate_visualizations(clean_data, analysis_results)
    
    return clean_data, analysis_results, plot

if __name__ == "__main__":
    data, results, plot = main()
