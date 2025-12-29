Program:
# Dynamic Pricing using Model-Based Reinforcement Learning import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
#
# Step 1: Generate Sales Data #
np.random.seed(0)
prices = np.linspace(10, 100, 50)
demand = 200 - 1.5 * prices + np.random.normal(0, 5, 50)
data = pd.DataFrame({ "Price": prices, "Demand": demand
})
 
#
# Step 2: Train Demand Model #
X = data[["Price"]] y = data["Demand"]
model = LinearRegression() model.fit(X, y)
#
# Step 3: Environment Model #
def predict_demand(price):
price_df = pd.DataFrame({"Price": [price]}) return model.predict(price_df)[0]
#
# Step 4: Policy Optimization #
def optimize_price(price_range): best_price = 0
best_revenue = -1
for price in price_range:
demand = predict_demand(price) revenue = price * max(demand, 0)
if revenue > best_revenue: best_revenue = revenue best_price = price
return best_price, best_revenue

#
# Step 5: Execute Policy #
price_options = np.linspace(10, 100, 50)
optimal_price, max_revenue = optimize_price(price_options) print("Optimal Price:", round(optimal_price, 2))
 
print("Maximum Expected Revenue:", round(max_revenue, 2))
