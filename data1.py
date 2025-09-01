import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

df = pd.read_csv("counterfeit_products_renamed.csv")
print("✅ Dataset shape:", df.shape)
print(df.head())
print(df.info(verbose=True))
print(df.isnull().sum())
plt.figure(figsize=(6,4))
sns.countplot(x="fraud_indicator", data=df, palette="Set2")
plt.title("Fraud vs Legit Distribution")
plt.show()

fraud_rate = df["fraud_indicator"].mean()*100
print(f"⚠️ Fraud rate: {fraud_rate:.2f}%")

num_cols = ["cost_usd", "vendor_score", "feedback_count", "page_hits", "sales_vol"]
for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x="fraud_indicator", y=col, data=df, palette="Set3")
    plt.title(f"{col} vs Fraud Indicator")
    plt.show()

df['price_bucket'] = pd.qcut(df['cost_usd'], 5)
price_fraud = df.groupby('price_bucket')['fraud_indicator'].mean()
price_fraud.plot(kind='bar', figsize=(8,4), color='orange')
plt.title("Fraud Rate by Price Bucket")
plt.ylabel("Fraud Rate")
plt.show()

cat_cols = ["product_type", "manufacturer", "vendor_nation", "geo_inconsistency"]
for col in cat_cols:
    plt.figure(figsize=(10,4))
    fraud_rate_by_cat = df.groupby(col)["fraud_indicator"].mean().sort_values(ascending=False).head(10)
    sns.barplot(x=fraud_rate_by_cat.index, y=fraud_rate_by_cat.values, palette="coolwarm")
    plt.title(f"Fraud Rate by {col} (Top 10)")
    plt.xticks(rotation=45)
    plt.ylabel("Fraud Rate")
    plt.show()

bool_cols = ["contact_valid", "return_clarity", "payment_anomaly"]
for col in bool_cols:
    plt.figure(figsize=(6,4))
    sns.barplot(x=col, y="fraud_indicator", data=df, palette="Set2")
    plt.title(f"Fraud Rate by {col}")
    plt.ylabel("Fraud Rate")
    plt.show()


df['trust_score'] = df[['contact_valid','return_clarity','wholesale_avail','payment_anomaly']].sum(axis=1)
trust_fraud = df.groupby('trust_score')['fraud_indicator'].mean()
trust_fraud.plot(kind='bar', figsize=(6,4), color='green')
plt.title("Fraud Rate by Composite Trust Score")
plt.ylabel("Fraud Rate")
plt.show()

eng_cols = ["page_hits", "sales_vol", "saved_items"]
for col in eng_cols:
    plt.figure(figsize=(6,4))
    sns.kdeplot(df[df['fraud_indicator']==0][col], label='Legit', fill=True)
    sns.kdeplot(df[df['fraud_indicator']==1][col], label='Fraud', fill=True)
    plt.title(f"{col} Distribution by Fraud Indicator")
    plt.xlabel(col)
    plt.legend()
    plt.show()

vendor_fraud = df.groupby('vendor_code')['fraud_indicator'].sum().sort_values(ascending=False).head(10)
vendor_fraud.plot(kind='bar', figsize=(10,4), color='red')
plt.title("Top 10 Vendors by Number of Fraudulent Listings")
plt.ylabel("Fraud Count")
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df[num_cols].corr(), cmap="coolwarm", center=0, annot=True)
plt.title("Correlation Heatmap (Numeric Features)")
plt.show()

iso = IsolationForest(contamination=0.05, random_state=42)
df['anomaly'] = iso.fit_predict(df[num_cols])
sns.countplot(x='anomaly', hue='fraud_indicator', data=df)
plt.title("Outlier Detection vs Fraud")
plt.show()

print(" Final Domain-Driven EDA Completed")
