
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error


df = pd.read_csv("insurance.csv")

print("📊 First 5 Rows of Dataset:")
print(df.head())

print("\n📊 Dataset Info:")
print(df.info())

print("\n📊 Statistical Summary:")
print(df.describe())



# Pairplot
sns.pairplot(df)
plt.suptitle("Pairplot of Dataset", y=1.02)
plt.show()

# Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Smoking vs Charges
sns.boxplot(x="smoker", y="charges", data=df)
plt.title("Smoking vs Charges")
plt.show()


le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])
df['smoker'] = le.fit_transform(df['smoker'])
df['region'] = le.fit_transform(df['region'])


X = df.drop("charges", axis=1)
y = df["charges"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

lr_r2 = r2_score(y_test, y_pred_lr)
lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))

print("\n📈 Linear Regression Results:")
print("R2 Score:", lr_r2)
print("RMSE:", lr_rmse)


rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

rf_r2 = r2_score(y_test, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))

print("\n🌲 Random Forest Results:")
print("R2 Score:", rf_r2)
print("RMSE:", rf_rmse)



print("\n🏆 Model Comparison:")
print("Linear Regression R2:", lr_r2)
print("Random Forest R2:", rf_r2)

if rf_r2 > lr_r2:
    print("👉 Random Forest is the BEST model")
else:
    print("👉 Linear Regression is the BEST model")



plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_rf)
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted (Random Forest)")
plt.show()