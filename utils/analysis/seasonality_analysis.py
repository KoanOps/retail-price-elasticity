#!/usr/bin/env python3
"""
Seasonality Analysis Module

This module provides functionality for analyzing hierarchical seasonality effects
across different levels of product aggregation (total, product class, SKU).
It helps determine whether hierarchical seasonality coefficients at the product
class level improve predictions compared to more granular or global models.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os
from datetime import datetime, timedelta
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')

# Create directories for output
os.makedirs('results', exist_ok=True)

# Define simplified versions of the feature creation functions
def add_event_indicators(df):
    """Add basic event indicators to the dataframe"""
    # Create a copy of the dataframe
    data = df.copy()
    
    # Make sure Sold_Date is a datetime
    if not pd.api.types.is_datetime64_any_dtype(data['Sold_Date']):
        data['Sold_Date'] = pd.to_datetime(data['Sold_Date'])
    
    # Create month and day of week
    data['month'] = data['Sold_Date'].dt.month
    data['day_of_week'] = data['Sold_Date'].dt.dayofweek
    
    # Create indicators for months
    for month in range(1, 13):
        data[f'Is_Month_{month}'] = (data['month'] == month).astype(int)
    
    # Create indicators for days of week
    for day in range(7):
        data[f'Is_Day_{day}'] = (data['day_of_week'] == day).astype(int)
    
    # Common retail events (simplified)
    data['Is_Weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
    
    # November and December for holiday season
    data['Is_November'] = data['month'].eq(11).astype(int)
    data['Is_December'] = data['month'].eq(12).astype(int)
    
    # Create a few sample deviation features (using month as proxy)
    data['Sales_Deviation_Thursday_November'] = (data['day_of_week'] == 3) & (data['month'] == 11)
    data['Sales_Deviation_PreChristmas'] = (data['Sold_Date'].dt.day >= 15) & (data['month'] == 12) & (data['Sold_Date'].dt.day < 25)
    data['Sales_Deviation_ChristmasEve'] = (data['Sold_Date'].dt.day == 24) & (data['month'] == 12)
    data['Sales_Deviation_Thanksgiving'] = ((data['month'] == 11) & (data['Sold_Date'].dt.day >= 22) & (data['Sold_Date'].dt.day <= 28) & (data['day_of_week'] == 3))
    data['Sales_Deviation_Week_After_Thanksgiving'] = ((data['month'] == 11) & (data['Sold_Date'].dt.day > 28)) | ((data['month'] == 12) & (data['Sold_Date'].dt.day <= 7))
    data['Sales_Deviation_Week_Before_Thanksgiving'] = (data['month'] == 11) & (data['Sold_Date'].dt.day >= 15) & (data['Sold_Date'].dt.day < 22)
    data['Sales_Deviation_December'] = (data['month'] == 12)
    data['Sales_Deviation_BlackFriday'] = ((data['month'] == 11) & (data['Sold_Date'].dt.day >= 23) & (data['Sold_Date'].dt.day <= 30) & (data['day_of_week'] == 4))
    
    return data

def add_advanced_features(df):
    """Add advanced features (non-promotional)"""
    # Simply return the dataframe as is, without adding promo features
    # This function is kept for compatibility with existing code
    return df.copy()

def test_hierarchical_seasonality(data_path='data/sales.parquet', top_n_classes=10):
    """
    Test if hierarchical seasonality coefficients at product class level improve predictions
    
    Parameters:
    -----------
    data_path : str
        Path to the sales data file
    top_n_classes : int
        Number of top product classes to analyze (by sales volume)
    
    Returns:
    --------
    dict with results of the comparison
    """
    print(f"Testing hierarchical seasonality approach...")
    print(f"Loading data from {data_path}...")
    
    # Load data
    if data_path.endswith('.parquet'):
        data = pd.read_parquet(data_path)
    else:
        data = pd.read_csv(data_path)
    
    # Convert date to datetime if needed
    if data['Sold_Date'].dtype == 'object':
        data['Sold_Date'] = pd.to_datetime(data['Sold_Date'])
    
    # Add features
    print("Adding seasonal features...")
    df = add_event_indicators(data)
    df = add_advanced_features(df)
    
    # Top seasonal features - with promo features removed
    seasonality_features = [
        'Sales_Deviation_Thursday_November', 
        'Sales_Deviation_PreChristmas',
        'Sales_Deviation_ChristmasEve', 
        'Sales_Deviation_Thanksgiving', 
        'Sales_Deviation_Week_After_Thanksgiving',
        'Sales_Deviation_Week_Before_Thanksgiving',
        'Sales_Deviation_December',
        'Sales_Deviation_BlackFriday',
        'Is_Weekend',
        'Is_November',
        'Is_December'
    ]
    
    # Filter to features that exist in the dataset
    seasonality_features = [f for f in seasonality_features if f in df.columns]
    print(f"Using {len(seasonality_features)} seasonal features")
    
    # Determine chronological train/test split (80/20)
    min_date = df['Sold_Date'].min()
    max_date = df['Sold_Date'].max()
    date_range = (max_date - min_date).days
    split_days = int(date_range * 0.8)
    split_date = min_date + pd.Timedelta(days=split_days)
    
    print(f"Date range: {min_date.date()} to {max_date.date()}")
    print(f"Using dates before {split_date.date()} for training, after for testing")
    
    # Identify top product classes by sales volume
    product_class_sales = df.groupby('Product_Class_Code')['Qty_Sold'].sum().sort_values(ascending=False)
    top_classes = product_class_sales.head(top_n_classes).index.tolist()
    
    print(f"Analyzing {len(top_classes)} product classes with highest sales volume")
    
    # Filter data to top classes
    df_filtered = df[df['Product_Class_Code'].isin(top_classes)].copy()
    
    # Group data three ways for comparison:
    # 1. Total sales aggregated across all product classes
    # 2. Sales by product class
    # 3. Sales by individual SKU
    
    # 1. Total sales
    print("Creating total sales dataset...")
    total_sales = df_filtered.groupby('Sold_Date').agg({
        'Qty_Sold': 'sum'
    })
    
    for feature in seasonality_features:
        if feature in df_filtered.columns:
            total_sales[feature] = df_filtered.groupby('Sold_Date')[feature].mean()
    
    total_sales = total_sales.reset_index()
    
    # 2. Sales by product class
    print("Creating product class level dataset...")
    class_sales = df_filtered.groupby(['Product_Class_Code', 'Sold_Date']).agg({
        'Qty_Sold': 'sum'
    })
    
    for feature in seasonality_features:
        if feature in df_filtered.columns:
            class_sales[feature] = df_filtered.groupby(['Product_Class_Code', 'Sold_Date'])[feature].mean()
    
    class_sales = class_sales.reset_index()
    
    # 3. Sales by SKU
    print("Creating SKU level dataset...")
    sku_sales = df_filtered.groupby(['Product_Class_Code', 'SKU_Coded', 'Sold_Date']).agg({
        'Qty_Sold': 'sum'
    })
    
    for feature in seasonality_features:
        if feature in df_filtered.columns:
            sku_sales[feature] = df_filtered.groupby(['Product_Class_Code', 'SKU_Coded', 'Sold_Date'])[feature].mean()
    
    sku_sales = sku_sales.reset_index()
    
    # Filter SKUs with enough observations - fixed to_frame issue
    # Using a different approach to get SKUs with enough observations
    sku_count_series = sku_sales.groupby(['Product_Class_Code', 'SKU_Coded']).size()
    # Convert to DataFrame using pandas methods that work with the type system
    sku_counts_df = pd.DataFrame(sku_count_series)
    sku_counts_df.columns = ['count']  # Name the column
    sku_counts_df = sku_counts_df.reset_index()  # Move index to columns
    
    # Now filter
    valid_skus = sku_counts_df[sku_counts_df['count'] >= 30][['Product_Class_Code', 'SKU_Coded']]
    
    # Ensure we have both Product_Class_Code and SKU_Coded columns
    if 'Product_Class_Code' in valid_skus.columns and 'SKU_Coded' in valid_skus.columns:
        sku_sales = pd.merge(sku_sales, valid_skus[['Product_Class_Code', 'SKU_Coded']], 
                           on=['Product_Class_Code', 'SKU_Coded'], how='inner')
    else:
        # Handle the case where reset_index doesn't create the expected columns
        print("Warning: Unable to properly filter SKUs with enough observations")
        # Just use original SKU data but note the limitation
    
    print(f"Analysis includes {len(valid_skus)} SKUs with at least 30 observations each")
    
    # 1. Train and evaluate total sales model
    print("\nTraining total sales model...")
    total_train = total_sales[total_sales['Sold_Date'] < split_date]
    total_test = total_sales[total_sales['Sold_Date'] >= split_date]
    
    total_model = LinearRegression()
    total_model.fit(total_train[seasonality_features], total_train['Qty_Sold'])
    
    total_train_r2 = total_model.score(total_train[seasonality_features], total_train['Qty_Sold'])
    total_test_r2 = total_model.score(total_test[seasonality_features], total_test['Qty_Sold'])
    
    print(f"Total sales model: Train R² = {total_train_r2:.4f}, Test R² = {total_test_r2:.4f}")
    
    # 2. Train and evaluate product class models
    print("\nTraining product class level models...")
    class_results = {}
    
    for product_class, group in class_sales.groupby('Product_Class_Code'):
        train_group = group[group['Sold_Date'] < split_date]
        test_group = group[group['Sold_Date'] >= split_date]
        
        if len(train_group) < 30 or len(test_group) < 10:
            continue  # Skip classes with too little data
            
        model = LinearRegression()
        model.fit(train_group[seasonality_features], train_group['Qty_Sold'])
        
        train_r2 = model.score(train_group[seasonality_features], train_group['Qty_Sold'])
        test_r2 = model.score(test_group[seasonality_features], test_group['Qty_Sold'])
        
        class_results[product_class] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'coefficients': dict(zip(seasonality_features, model.coef_)),
            'n_train': len(train_group),
            'n_test': len(test_group)
        }
    
    # 3. Train and evaluate SKU models
    print("\nTraining SKU level models...")
    sku_results = {}
    
    for (product_class, sku), group in sku_sales.groupby(['Product_Class_Code', 'SKU_Coded']):
        train_group = group[group['Sold_Date'] < split_date]
        test_group = group[group['Sold_Date'] >= split_date]
        
        if len(train_group) < 30 or len(test_group) < 10:
            continue  # Skip SKUs with too little data
            
        model = LinearRegression()
        model.fit(train_group[seasonality_features], train_group['Qty_Sold'])
        
        train_r2 = model.score(train_group[seasonality_features], train_group['Qty_Sold'])
        test_r2 = model.score(test_group[seasonality_features], test_group['Qty_Sold'])
        
        sku_results[(product_class, sku)] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'product_class': product_class,
            'n_train': len(train_group),
            'n_test': len(test_group)
        }
    
    # Summarize results
    print("\n===== HIERARCHICAL SEASONALITY TEST RESULTS =====")
    
    # Calculate averages
    class_train_r2 = np.mean([r['train_r2'] for r in class_results.values()])
    class_test_r2 = np.mean([r['test_r2'] for r in class_results.values()])
    
    sku_train_r2 = np.mean([r['train_r2'] for r in sku_results.values()])
    sku_test_r2 = np.mean([r['test_r2'] for r in sku_results.values()])
    
    print(f"Total Sales Model: Train R² = {total_train_r2:.4f}, Test R² = {total_test_r2:.4f}")
    print(f"Product Class Models (avg of {len(class_results)}): Train R² = {class_train_r2:.4f}, Test R² = {class_test_r2:.4f}")
    print(f"SKU Models (avg of {len(sku_results)}): Train R² = {sku_train_r2:.4f}, Test R² = {sku_test_r2:.4f}")
    
    # Calculate class-level stats
    class_r2_diff = [r['train_r2'] - r['test_r2'] for r in class_results.values()]
    
    # Calculate overfitting metrics
    total_overfit = total_train_r2 - total_test_r2
    class_overfit = class_train_r2 - class_test_r2
    sku_overfit = sku_train_r2 - sku_test_r2
    
    # Calculate top-performing product classes
    top_pc_by_test_r2 = sorted([(pc, r['test_r2']) for pc, r in class_results.items()], 
                              key=lambda x: x[1], reverse=True)[:3]
    
    # Calculate bottom-performing product classes
    bottom_pc_by_test_r2 = sorted([(pc, r['test_r2']) for pc, r in class_results.items()], 
                                key=lambda x: x[1])[:3]
    
    # Visualize the comparison
    plt.figure(figsize=(12, 8))
    
    # Box plot of test R² values
    class_test_r2_values = [r['test_r2'] for r in class_results.values()]
    sku_test_r2_values = [r['test_r2'] for r in sku_results.values()]
    
    plt.boxplot([class_test_r2_values, sku_test_r2_values], 
               labels=['Product Class Level', 'SKU Level'])
    
    plt.axhline(y=float(total_test_r2), color='r', linestyle='--', 
               label=f'Total Sales Model (R² = {total_test_r2:.4f})')
    
    plt.title('Test R² Comparison Across Modeling Levels')
    plt.ylabel('Test R²')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('results/seasonality_level_comparison.png', dpi=300)
    
    # Visualize coefficient differences across product classes
    plt.figure(figsize=(15, 8))
    feature_to_visualize = seasonality_features[0]  # Pick first feature to visualize
    
    coef_values = [result['coefficients'][feature_to_visualize] for pc, result in class_results.items()]
    pc_labels = list(class_results.keys())
    
    plt.bar(pc_labels, coef_values)
    plt.axhline(y=total_model.coef_[0], color='r', linestyle='--', 
               label='Total Sales Model Coefficient')
    
    plt.title(f'Coefficient for {feature_to_visualize} Across Product Classes')
    plt.xticks(rotation=90)
    plt.ylabel('Coefficient Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results/feature_variation_{feature_to_visualize}.png', dpi=300)
    
    # Visualize per-SKU versus per-class R² for a specific product class
    if len(class_results) > 0 and len(sku_results) > 0:
        sample_class = list(class_results.keys())[0]
        skus_in_class = [(pc, sku) for (pc, sku) in sku_results.keys() if pc == sample_class]
        
        if skus_in_class:
            plt.figure(figsize=(12, 6))
            
            sku_test_r2_in_class = [sku_results[sku_key]['test_r2'] for sku_key in skus_in_class]
            sku_labels = [sku for (_, sku) in skus_in_class]
            
            plt.bar(sku_labels, sku_test_r2_in_class)
            plt.axhline(y=class_results[sample_class]['test_r2'], color='r', linestyle='--',
                       label=f'Product Class Model (R² = {class_results[sample_class]["test_r2"]:.4f})')
            
            plt.title(f'SKU-Level Test R² for Product Class {sample_class}')
            plt.xticks(rotation=90)
            plt.ylabel('Test R²')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'results/sku_vs_class_{sample_class}.png', dpi=300)
    
    # Additional result analysis
    print("\n===== DETAILED ANALYSIS SUMMARY =====")
    print(f"Overfitting (Train R² - Test R²):")
    print(f"  Total Model: {total_overfit:.4f}")
    print(f"  Product Class Models (avg): {class_overfit:.4f}")
    print(f"  SKU Models (avg): {sku_overfit:.4f}")
    
    print("\nTop 3 Product Classes by Test R²:")
    for pc, r2 in top_pc_by_test_r2:
        print(f"  Class {pc}: Test R² = {r2:.4f}")
    
    print("\nBottom 3 Product Classes by Test R²:")
    for pc, r2 in bottom_pc_by_test_r2:
        print(f"  Class {pc}: Test R² = {r2:.4f}")
    
    # Variability in seasonality coefficients
    print("\nSeasonal Feature Coefficient Analysis:")
    
    # Get coefficients for the first feature across all product classes and compare to total
    feature = seasonality_features[0]
    total_coef = total_model.coef_[0]
    class_coefs = [result['coefficients'][feature] for result in class_results.values()]
    coef_variation = np.std(class_coefs) / np.mean(class_coefs) if np.mean(class_coefs) != 0 else 0
    
    print(f"For feature '{feature}':")
    print(f"  Total model coefficient: {total_coef:.4f}")
    print(f"  Mean across product classes: {np.mean(class_coefs):.4f}")
    print(f"  Standard deviation: {np.std(class_coefs):.4f}")
    print(f"  Coefficient of variation: {coef_variation:.4f}")
    
    # Provide practical recommendations
    print("\n===== RECOMMENDATIONS FOR SEASONALITY MODELING =====")
    
    if total_test_r2 > class_test_r2 and class_test_r2 > sku_test_r2:
        print("Recommendation: Use global seasonality model (aggregated across all product classes)")
        print("Justification: The total sales model outperforms both product class and SKU level models.")
    elif class_test_r2 > total_test_r2 and class_test_r2 > sku_test_r2:
        print("Recommendation: Use product class level seasonality models") 
        print("Justification: Product class models outperform both the global model and SKU level models.")
    elif sku_test_r2 > total_test_r2 and sku_test_r2 > class_test_r2:
        print("Recommendation: Use SKU level seasonality models")
        print("Justification: SKU level models outperform both global and product class models.")
    else:
        print("Recommendation: Use a hybrid approach")
        if coef_variation > 0.5:
            print("Justification: There is significant variation in seasonality patterns across product classes.")
            print("Consider using product class coefficients for the top performing classes, and the global model for others.")
        else:
            print("Justification: Seasonality patterns are relatively consistent across product classes.")
            print("Consider using the global seasonality model but with product class-specific intercepts.")
    
    # Integration with HBM recommendations
    print("\n===== INTEGRATION WITH HIERARCHICAL BAYESIAN MODEL =====")
    print("Based on the analysis of seasonality patterns across levels of aggregation:")
    
    if total_test_r2 > 0.3 and total_overfit < 0.3:
        print("1. Consider adding seasonality as fixed effects at the global level in your HBM.")
        print("   This approach recognizes that seasonality impacts all products similarly.")
    
    if class_test_r2 > 0.1 and class_overfit < 0.5:
        print("2. For product class variation, you could add seasonality coefficients in the hierarchical structure.")
        print("   This would allow seasonal impacts to vary by product class while still benefiting from shrinkage.")
    
    if sku_test_r2 < 0 or sku_overfit > 1.0:
        print("3. Avoid modeling seasonality at the SKU level, as it leads to severe overfitting.")
        print("   Instead, let price elasticity vary by SKU while keeping seasonality at higher levels.")
    
    # Identify the most impactful seasonal features
    feature_importances = {}
    for f_idx, feature in enumerate(seasonality_features):
        feature_importances[feature] = abs(total_model.coef_[f_idx])
    
    top_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:5]
    
    print("\nTop 5 most important seasonal features (based on coefficient magnitude):")
    for feature, importance in top_features:
        print(f"  {feature}: {importance:.4f}")
    
    print(f"\nResults saved to results/")
    
    # Return detailed results for further analysis
    return {
        'total_model': {
            'train_r2': total_train_r2,
            'test_r2': total_test_r2,
            'coefficients': dict(zip(seasonality_features, total_model.coef_))
        },
        'class_results': class_results,
        'sku_results': sku_results,
        'seasonality_features': seasonality_features,
        'recommendations': {
            'total_overfit': total_overfit,
            'class_overfit': class_overfit,
            'sku_overfit': sku_overfit,
            'top_features': top_features
        }
    }

if __name__ == "__main__":
    # When run as a script, execute the test with default parameters
    print("Running hierarchical seasonality analysis...")
    results = test_hierarchical_seasonality()
    print("Analysis complete. See generated plots for results.")