"""
Retail Data Simulation Module for Validation and Testing.

This module generates realistic synthetic retail sales data with known true price elasticities,
allowing for rigorous validation of elasticity estimation models against ground truth.

PURPOSE:
- Create controlled test data with known elasticities for model validation
- Generate data that mimics real-world retail patterns and challenges
- Provide reproducible datasets for benchmarking different models
- Support unit testing with smaller, deterministic datasets

ASSUMPTIONS:
- Price elasticities follow a normal distribution (typically negative)
- Log-linear relationship between price and quantity demanded
- SKUs are grouped into product classes with class-level effects
- Prices are log-normally distributed with some variation by SKU
- Random noise follows a normal distribution
- Price and quantity have an inverse correlation (higher price, lower quantity)

DATA GENERATION MODEL:
The core model follows this formula:
    log(quantity) = intercept + elasticity * log(price) + seasonality + noise

EDGE CASES:
- Very low elasticity values could create unrealistic sales patterns
- Using extreme correlation values may create unrealistic data
- Very small datasets may not exhibit expected statistical properties
- Generated data always has perfect column names (unlike real-world data)
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

def generate_synthetic_data(
    n_observations: int = 50000,
    n_skus: int = 200,
    n_product_classes: int = 10,
    n_stores: int = 20,
    start_date: str = "2023-01-01",
    end_date: str = "2023-12-31",
    true_elasticity_mean: float = -1.2,
    true_elasticity_std: float = 0.3,
    correlation: float = -0.5,
    noise_level: float = 0.2,
    output_file: Optional[str] = None,
    promo_frequency: float = 0.15,  # 15% of transactions are on promotion
    promo_discount_mean: float = 0.3,  # 30% average discount during promotions
    promo_elasticity_boost: float = 0.5,  # Elasticity becomes 50% stronger during promotions
    with_seasonality: bool = False,  # Whether to include seasonal patterns
    seasonality_strength: float = 0.3  # Relative impact of seasonality on quantity
) -> pd.DataFrame:
    """
    Generate synthetic retail sales data with known elasticities for model validation.
    
    This function creates a realistic retail dataset that mimics transaction-level sales data
    with prices, quantities, SKUs, product classes, and dates. It assigns known elasticity values
    to each SKU, allowing for true validation of estimation models against ground truth.
    
    Key Features:
    - Creates realistic price-quantity relationships with known elasticities
    - Generates hierarchical data structure (stores, products, classes)
    - Includes seasonal variations in baseline demand
    - Offers configurable noise levels to test model robustness
    - Automatically saves true elasticities for validation
    - Includes promotional events with deeper discounts and higher elasticities
    
    Mathematical Model:
    log(quantity) = base_demand + elasticity * log(price) + seasonality + promo_effect + noise
    Where:
    - base_demand varies by SKU and has correlation with elasticity
    - elasticity is drawn from N(true_elasticity_mean, true_elasticity_std)
    - seasonality follows a sinusoidal pattern with yearly cycles
    - promo_effect increases elasticity during promotional periods
    - noise is drawn from N(0, noise_level)
    
    Args:
        n_observations: Number of sales transactions to generate
        n_skus: Number of unique SKUs
        n_product_classes: Number of product classes
        n_stores: Number of stores
        start_date: Start date for transactions (YYYY-MM-DD)
        end_date: End date for transactions (YYYY-MM-DD)
        true_elasticity_mean: Mean of the true elasticity distribution
        true_elasticity_std: Standard deviation of the true elasticity distribution
        correlation: Correlation between intercept and elasticity
        noise_level: Standard deviation of the noise term
        output_file: Optional file path to save the data
        promo_frequency: Proportion of transactions that occur during promotions
        promo_discount_mean: Average discount percentage during promotions
        promo_elasticity_boost: Factor by which elasticity increases during promotions
        with_seasonality: Whether to include seasonal patterns
        seasonality_strength: Relative impact of seasonality on quantity
        
    Returns:
        DataFrame with synthetic sales data and saves metadata with true elasticities
    """
    logger.info(f"Generating {n_observations} synthetic sales transactions")
    
    # Generate dates
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    days = (end - start).days + 1
    
    # Create product classes
    product_classes = [f"Class_{i:02d}" for i in range(1, n_product_classes + 1)]
    
    # Create SKUs with mapping to product classes
    skus = []
    sku_to_class = {}
    
    skus_per_class = n_skus // n_product_classes
    remainder = n_skus % n_product_classes
    
    for i, cls in enumerate(product_classes):
        # Add extra SKUs to first few classes if n_skus doesn't divide evenly
        extra = 1 if i < remainder else 0
        class_skus = [f"{cls}_SKU_{j:03d}" for j in range(1, skus_per_class + extra + 1)]
        
        for sku in class_skus:
            skus.append(sku)
            sku_to_class[sku] = cls
    
    # Create stores
    stores = [f"Store_{i:02d}" for i in range(1, n_stores + 1)]
    
    # Generate SKU parameters with correlation between intercept and elasticity
    # Using multivariate normal distribution
    means = np.array([5.0, true_elasticity_mean])  # [intercept_mean, elasticity_mean]
    stds = np.array([0.5, true_elasticity_std])    # [intercept_std, elasticity_std]
    
    # Correlation matrix
    rho = correlation
    corr_matrix = np.array([[1.0, rho], [rho, 1.0]])
    
    # Convert to covariance matrix
    cov_matrix = np.diag(stds) @ corr_matrix @ np.diag(stds)
    
    # Generate parameters for each SKU
    sku_params = {}
    for sku in skus:
        # Sample from multivariate normal
        params = np.random.multivariate_normal(means, cov_matrix)
        sku_params[sku] = {
            'intercept': params[0],
            'elasticity': params[1]
        }
    
    # Generate sales data
    data = []
    for _ in range(n_observations):
        # Random date
        days_offset = np.random.randint(0, days)
        transaction_date = start + timedelta(days=days_offset)
        
        # Random SKU and store
        sku = np.random.choice(skus)
        store = np.random.choice(stores)
        product_class = sku_to_class[sku]
        
        # Price level with some randomness for each SKU
        # Log-price follows normal distribution
        base_price = np.exp(sku_params[sku]['intercept'] / 
                         (-3 * sku_params[sku]['elasticity']))
        
        # Determine if this is a promotional transaction
        is_promo = np.random.random() < promo_frequency
        
        # Add variability to price
        price_variability = 0.2  # 20% price variation
        price_factor = np.random.uniform(1 - price_variability, 1 + price_variability)
        
        # Apply promotion discount if applicable
        promo_discount = 0.0
        if is_promo:
            # Random discount between 10% and twice the mean discount
            promo_discount = np.random.uniform(0.1, promo_discount_mean * 2)
            price_factor = price_factor * (1 - promo_discount)
            
        price = base_price * price_factor
        
        # Quantity as a function of price
        log_price = np.log(price)
        
        # Apply elasticity boost during promotions
        effective_elasticity = sku_params[sku]['elasticity']
        if is_promo:
            # Make elasticity more negative (stronger) during promotions
            effective_elasticity = effective_elasticity * (1 + promo_elasticity_boost)
            
        log_qty_mean = (sku_params[sku]['intercept'] + 
                      effective_elasticity * log_price)
        
        # Apply seasonality effects if enabled
        seasonality_multiplier = 1.0
        if with_seasonality:
            # Day of week effect (weekend uplift)
            day_of_week = transaction_date.weekday()
            is_weekend = day_of_week >= 5  # Saturday or Sunday
            day_of_week_effect = 1.2 if is_weekend else 1.0
            
            # Monthly seasonality (yearly cycle with peak in December)
            month = transaction_date.month
            # Monthly factors - peaks in December (holiday season), dips in January (post-holiday)
            month_factors = {
                1: 0.7,   # January - post holiday dip
                2: 0.8,   # February
                3: 0.9,   # March
                4: 1.0,   # April
                5: 1.05,  # May
                6: 1.1,   # June
                7: 1.05,  # July
                8: 1.1,   # August - back to school
                9: 1.0,   # September
                10: 1.1,  # October
                11: 1.3,  # November - pre-holiday/Black Friday
                12: 1.5   # December - holiday peak
            }
            month_effect = month_factors[month]
            
            # Holiday effects (major retail events)
            day = transaction_date.day
            
            # Black Friday (November 23-30)
            is_black_friday = (month == 11 and day >= 23 and day <= 30)
            black_friday_effect = 1.8 if is_black_friday else 1.0
            
            # Week before Christmas (December 18-24)
            is_pre_christmas = (month == 12 and day >= 18 and day <= 24)
            pre_christmas_effect = 1.7 if is_pre_christmas else 1.0
            
            # Post-Christmas sales (December 26-31)
            is_post_christmas = (month == 12 and day >= 26)
            post_christmas_effect = 1.4 if is_post_christmas else 1.0
            
            # Combine all seasonal effects
            event_effect = max(black_friday_effect, pre_christmas_effect, post_christmas_effect)
            seasonality_multiplier = day_of_week_effect * month_effect * event_effect
            
            # Product class specific seasonality (some product classes are more seasonal than others)
            class_idx = int(product_class.split('_')[1])
            class_seasonality_factor = 0.5 + (class_idx / n_product_classes)  # 0.5 to 1.5 range
            
            # Apply seasonality with product class variation
            seasonality_multiplier = 1.0 + ((seasonality_multiplier - 1.0) * 
                                          seasonality_strength * class_seasonality_factor)
            
            # Apply seasonality to log_qty_mean
            log_qty_mean = log_qty_mean + np.log(seasonality_multiplier)
        
        # Add noise to log quantity
        log_qty = log_qty_mean + np.random.normal(0, noise_level)
        qty = np.round(np.exp(log_qty))
        
        # Ensure minimum quantity is 1
        qty = max(1, qty)
        
        # Calculate total value
        total_value = price * qty
        
        # Add to data list
        data.append({
            'Transaction_Date': transaction_date,
            'SKU': sku,
            'Product_Class': product_class,
            'Store_ID': store,
            'Qty_Sold': qty,
            'Price_Per_Unit': price,
            'Total_Sale_Value': total_value,
            'Is_Promo': int(is_promo),
            'Promo_Discount': promo_discount if is_promo else 0.0,
            'Seasonality_Factor': seasonality_multiplier if with_seasonality else 1.0
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Create metadata with true elasticities
    metadata = {
        "true_elasticities": {sku: params["elasticity"] for sku, params in sku_params.items()},
        "true_correlation": correlation,
        "data_generation_params": {
            "n_observations": n_observations,
            "n_skus": n_skus,
            "n_product_classes": n_product_classes,
            "n_stores": n_stores,
            "true_elasticity_mean": true_elasticity_mean,
            "true_elasticity_std": true_elasticity_std,
            "correlation": correlation,
            "noise_level": noise_level,
            "promo_frequency": promo_frequency,
            "promo_discount_mean": promo_discount_mean,
            "promo_elasticity_boost": promo_elasticity_boost,
            "with_seasonality": with_seasonality,
            "seasonality_strength": seasonality_strength if with_seasonality else 0.0
        }
    }
    
    # Save to file if specified
    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        # Save data
        df.to_parquet(output_file)
        
        # Save metadata
        metadata_file = os.path.splitext(output_file)[0] + "_metadata.json"
        pd.Series(metadata).to_json(metadata_file)
        
        logger.info(f"Saved synthetic data to {output_file}")
        logger.info(f"Saved metadata to {metadata_file}")
    
    return df

def generate_test_data_small():
    """Generate a small test dataset for unit testing."""
    return generate_synthetic_data(
        n_observations=5000,
        n_skus=50,
        n_product_classes=5,
        n_stores=10,
        true_elasticity_mean=-1.0,
        true_elasticity_std=0.2,
        correlation=-0.3,
        noise_level=0.1
    )

def generate_test_data_medium():
    """Generate a medium-sized test dataset."""
    return generate_synthetic_data(
        n_observations=20000,
        n_skus=100,
        n_product_classes=8,
        n_stores=15,
        true_elasticity_mean=-1.2,
        true_elasticity_std=0.3,
        correlation=-0.5,
        noise_level=0.2
    )

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Output directory
    output_dir = "Retail/data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate small dataset
    small_output = os.path.join(output_dir, "synthetic_small.parquet")
    generate_synthetic_data(
        n_observations=10000,
        n_skus=50,
        n_product_classes=5,
        n_stores=10,
        output_file=small_output
    )
    
    # Generate medium dataset
    medium_output = os.path.join(output_dir, "synthetic_medium.parquet")
    generate_synthetic_data(
        n_observations=50000,
        n_skus=200,
        n_product_classes=10,
        n_stores=20,
        output_file=medium_output
    )
    
    logger.info("Data generation complete.") 