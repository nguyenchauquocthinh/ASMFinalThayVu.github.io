import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

st.set_page_config(page_title="Weekly Product Demand Forecasting", layout="wide")
st.title("📊 Weekly Product Demand Forecasting - ABC Manufacturing")

uploaded_file = st.file_uploader("📁 Upload the file weekly_product_demand_3years.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['week'] = pd.to_datetime(df['week'])

    st.subheader("1️⃣ Sample Data")
    st.dataframe(df.head())

    # EDA Section
    st.subheader("2️⃣ Exploratory Data Analysis")

    weekly_sales = df.groupby('week')['units_sold'].sum().reset_index()
    fig1, ax1 = plt.subplots()
    sns.lineplot(x='week', y='units_sold', data=weekly_sales, ax=ax1)
    ax1.set_title('Weekly Total Units Sold')
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    sns.boxplot(x='product', y='price', data=df, ax=ax2)
    ax2.set_title('Price Distribution by Product')
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    sns.barplot(x='region', y='units_sold', data=df, estimator=sum, ax=ax3)
    ax3.set_title('Total Units Sold by Region')
    st.pyplot(fig3)

    fig4, ax4 = plt.subplots()
    sns.boxplot(x='promotion', y='units_sold', data=df, ax=ax4)
    ax4.set_title('Effect of Promotion on Units Sold')
    ax4.set_xticklabels(['No Promo', 'Promo'])
    st.pyplot(fig4)

    fig5, ax5 = plt.subplots()
    sns.heatmap(df[['units_sold', 'price', 'promotion']].corr(), annot=True, cmap='coolwarm', ax=ax5)
    ax5.set_title('Correlation Heatmap')
    st.pyplot(fig5)

    # Modeling
    st.subheader("3️⃣ Train Forecasting Model")

    df['week_num'] = df['week'].dt.isocalendar().week
    df['year'] = df['week'].dt.year
    df = pd.get_dummies(df, columns=['product', 'region'], drop_first=True)

    features = ['price', 'promotion', 'week_num', 'year'] + [col for col in df.columns if 'product_' in col or 'region_' in col]
    target = 'units_sold'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.success(f"✅ Model Evaluation:")
    st.write(f"**MAE:** {mae:.2f}")
    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**R² Score:** {r2:.2f}")

    fig6, ax6 = plt.subplots()
    ax6.scatter(y_test, y_pred, alpha=0.4)
    ax6.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax6.set_xlabel("Actual")
    ax6.set_ylabel("Predicted")
    ax6.set_title("Actual vs Predicted Units Sold")
    st.pyplot(fig6)

else:
    st.warning("👆 Please upload the CSV file to begin.")
