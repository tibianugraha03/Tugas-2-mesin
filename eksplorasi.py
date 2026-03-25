#%% 
# ==========================================
# MODUL 1: Regresi Linear Data Sembarang
# ==========================================
import numpy as np # untuk perhitungan saintifik
import matplotlib.pyplot as plt # untuk plotting
from sklearn.linear_model import LinearRegression # perbaikan spasi pada nama library

# Membuat data sembarang penjualan dan harga
penjualan = np.array([6,5,5,4,4,3,2,2,2,1])
harga = np.array([16000, 18000, 27000, 34000, 50000, 68000, 65000, 81000, 85000, 90000])

print("Data Penjualan:", penjualan)
print("Data Harga :", harga)

# Membuat model regresi
# Kita tukar baris dan kolom variabel ini, agar bisa dikalikan dalam operasi matriks
penjualan = penjualan.reshape(-1, 1) 

linreg = LinearRegression()
linreg.fit(penjualan, harga)

# Visualisasi plot hasil regresi
plt.figure(figsize=(8,5))
plt.scatter(penjualan, harga, color='red')
plt.plot(penjualan, linreg.predict(penjualan), color='blue')
plt.title("Visualisasi model regresi data penjualan dan harga")
plt.xlabel("Penjualan")
plt.ylabel("Harga")
plt.show()

#%% 
# ==========================================
# MODUL 2: Regresi Linear dengan Dataset Asli
# ==========================================
import pandas as pd # untuk dataframe

# Membaca data CSV yang sudah diunduh
df = pd.read_csv("FuelConsumptionCo2.csv") 

# Kita ambil kolom mana saja yang akan kita analisis
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'CO2EMISSIONS']]

# Visualisasi hubungan Engine Size dan Emission
plt.figure(figsize=(8,5))
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.title("Data Asli: Engine Size vs Emission")
plt.show()

# Membagi data (Splitting data) menjadi Data Train dan Data Test
# Perbaikan sintaks random array dari modul
msk = np.random.rand(len(df)) < 0.8 
train = cdf[msk]
test = cdf[~msk]

# Membuat model regresi berdasarkan Data Train
regr = LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)

# Koefisien model
print('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)

# Plot hasil regresi final
plt.figure(figsize=(8,5))
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.title("Model Regresi Linear: Engine Size vs Emission")
plt.show()