import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# TITLE
st.set_page_config(layout="wide")
st.title("📊 Weekly Product Demand Forecasting - ABC Manufacturing")

# 1. Load Data
st.header("1. Data Overview & Cleaning")

@st.cache_data
def load_data():
    df = pd.read_csv("weekly_product_demand_3years.csv")
    df['week'] = pd.to_datetime(df['week'])
    return df

df = load_data()
st.subheader("📌 Sample Data")
st.dataframe(df.head())

# Mô tả dữ liệu
st.write("**📊 Thống kê mô tả**")
st.dataframe(df.describe())

st.write("**❗ Số lượng giá trị thiếu (null)**")
st.write(df.isnull().sum())

st.write("**📎 Số dòng trùng lặp:**", df.duplicated().sum())

# Phân bố sản phẩm và khu vực
col1, col2 = st.columns(2)
with col1:
    st.write("🔍 Phân bố sản phẩm:")
    st.write(df['product'].value_counts())
with col2:
    st.write("🌍 Phân bố khu vực:")
    st.write(df['region'].value_counts())

# 2. Visualization
st.header("2. Data Visualization")

# Biểu đồ 1: Số lượng bán theo tuần
weekly_sales = df.groupby('week')['units_sold'].sum().reset_index()
fig1, ax1 = plt.subplots(figsize=(12,4))
sns.lineplot(data=weekly_sales, x='week', y='units_sold', ax=ax1)
ax1.set_title("Số lượng bán theo tuần")
st.pyplot(fig1)

# Biểu đồ 2: Giá trung bình theo loại sản phẩm
fig2, ax2 = plt.subplots(figsize=(10,5))
sns.boxplot(data=df, x='product', y='price', ax=ax2)
ax2.set_title("Phân bố giá theo loại sản phẩm")
st.pyplot(fig2)

# Biểu đồ 3: Tổng sản phẩm bán theo khu vực
fig3, ax3 = plt.subplots(figsize=(10,5))
sns.barplot(data=df, x='region', y='units_sold', estimator=sum, ax=ax3)
ax3.set_title("Tổng sản phẩm bán theo khu vực")
st.pyplot(fig3)

# Biểu đồ 4: Ảnh hưởng khuyến mãi đến doanh số
fig4, ax4 = plt.subplots(figsize=(8,5))
sns.boxplot(data=df, x='promotion', y='units_sold', ax=ax4)
ax4.set_title("Tác động khuyến mãi đến doanh số")
ax4.set_xticklabels(['Không KM', 'Có KM'])
st.pyplot(fig4)

# Biểu đồ 5: Ma trận tương quan
df_corr = df[['units_sold', 'price', 'promotion']]
fig5, ax5 = plt.subplots(figsize=(6,4))
sns.heatmap(df_corr.corr(), annot=True, cmap='coolwarm', ax=ax5)
ax5.set_title("Ma trận tương quan")
st.pyplot(fig5)

# 3. Forecasting Model
st.header("3. Forecasting Model")

# Feature Engineering
df['week_num'] = df['week'].dt.isocalendar().week
df['year'] = df['week'].dt.year
df_encoded = pd.get_dummies(df, columns=['product', 'region'], drop_first=True)

features = ['price', 'promotion', 'week_num', 'year'] + \
           [col for col in df_encoded.columns if 'product_' in col or 'region_' in col]
target = 'units_sold'

X = df_encoded[features]
y = df_encoded[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.success("✅ Model Evaluation")
st.write(f"**MAE:** {mae:.2f}")
st.write(f"**RMSE:** {rmse:.2f}")
st.write(f"**R² Score:** {r2:.2f}")

# Biểu đồ thực tế vs dự đoán
st.subheader("🔍 Thực tế vs Dự đoán")
fig6, ax6 = plt.subplots(figsize=(8,5))
ax6.scatter(y_test, y_pred, alpha=0.4)
ax6.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax6.set_xlabel("Giá trị thực tế")
ax6.set_ylabel("Giá trị dự đoán")
ax6.set_title("So sánh thực tế và dự đoán")
st.pyplot(fig6)


