import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# List of microservices
microservices = [
    "User Management Service",
    "Product Catalog Service",
    "Inventory Service",
    "Shopping Cart Service",
    "Order Processing Service",
    "Payment Gateway Service",
    #"Order Fulfillment Service",
    #"Customer Relationship Management (CRM) Service",
    #"Content Management System (CMS) Service",
    "Search Service",
    #"Recommendation Service",
    "Pricing Service",
    "Promotions and Discounts Service",
    #"User Authentication Service",
    "User Authorization Service",
    "Notification Service",
    #"Logging and Monitoring Service",
    "Security Service",
    #"Analytics Service",
    #"Reporting Service",
    #"Image and Media Service",
    "Review and Rating Service",
    "Wishlist Service",
    "Address Management Service",
    #"Shipping and Delivery Service",
    #"Tax Calculation Service",
    "Refund and Returns Service"
    #"A/B Testing Service"
]

# Function to generate random timestamp
def random_timestamp(start_date, end_date):
    delta = end_date - start_date
    random_days = random.randint(0, delta.days)
    random_seconds = random.randint(0, 24*60*60)
    return start_date + timedelta(days=random_days, seconds=random_seconds)

# Generate 10000 records
np.random.seed(0)  # For reproducibility
num_records = 10000
start_date = datetime(2024, 3, 21)
end_date = datetime(2024, 3, 21)
transaction_ids = ['SN{:08d}'.format(i) for i in range(1, num_records+1)]
timestamps = [random_timestamp(start_date, end_date) for _ in range(num_records)]
flags = np.random.choice(['Yes', 'No'], size=num_records, p=[0.9, 0.1])  # 90% Yes, 10% No
microservices_per_row = np.random.randint(1, 6, size=num_records)

data = []
for i in range(num_records):
    selected_microservices = random.sample(microservices, k=microservices_per_row[i])
    data.append({
        'TransactionID': transaction_ids[i],
        'Timestamp': timestamps[i].strftime('%m/%d/%Y %H:%M:%S'),
        'Flag': flags[i],
        'Microservices': ', '.join(selected_microservices)
    })

# Create DataFrame
df = pd.DataFrame(data)

# Write DataFrame to Excel
excel_file = 'transactions.xlsx'
df.to_excel(excel_file, index=False)
print(f"Excel file '{excel_file}' has been generated with {num_records} records.")
