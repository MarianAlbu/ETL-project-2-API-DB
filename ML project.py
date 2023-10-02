import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression 
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


df_sales_train = pd.read_csv('sales_train.csv')
df_items = pd.read_csv('items.csv')
df_item_categories = pd.read_csv('item_categories.csv')

copy_df_st = df_sales_train.copy()
copy_df_i = df_items.copy()
copy_df_ic = df_item_categories.copy()

joined_fact_df = pd.merge(copy_df_st, copy_df_i, how = 'left', on = ['item_id'])

joined_group = joined_fact_df[['date_block_num', 'shop_id', 'item_category_id', 'item_cnt_day']].groupby(['date_block_num', 'shop_id', 'item_category_id'], as_index = False).sum() 
copy_joined = joined_group.copy()

# Legger til verdier 1 mnd tilbake
copy_joined['date_block_num'] = copy_joined['date_block_num'] + 1
merged_copy_joined_1 = pd.merge(joined_group, copy_joined, how = 'left', on = ['date_block_num', 'shop_id', 'item_category_id'])
merged_copy_1 = merged_copy_joined_1.rename(columns={'item_cnt_day_x': 'item_cnt_day', 'item_cnt_day_y': 'item_cnt_lag_6'})

# Legger til verdier 2 mnd tilbake
merged_copy_1['date_block_num'] = merged_copy_1['date_block_num'] + 1
merged_copy_joined_2 = pd.merge(joined_group, merged_copy_1, how = 'left', on = ['date_block_num', 'shop_id', 'item_category_id'])
merged_copy_2 = merged_copy_joined_2.rename(columns={'item_cnt_day_x': 'item_cnt_day', 'item_cnt_day_y': 'item_cnt_lag_5'})

# Legger til verdier 3 mnd tilbake
merged_copy_2['date_block_num'] = merged_copy_2['date_block_num'] + 1
merged_copy_joined_3 = pd.merge(joined_group, merged_copy_2, how = 'left', on = ['date_block_num', 'shop_id', 'item_category_id'])
merged_copy_3 = merged_copy_joined_3.rename(columns={'item_cnt_day_x': 'item_cnt_day', 'item_cnt_day_y': 'item_cnt_lag_4'})

# Legger til verdier 4 mnd tilbake
merged_copy_3['date_block_num'] = merged_copy_3['date_block_num'] + 1
merged_copy_joined_4 = pd.merge(joined_group, merged_copy_3, how = 'left', on = ['date_block_num', 'shop_id', 'item_category_id'])
merged_copy_4 = merged_copy_joined_4.rename(columns={'item_cnt_day_x': 'item_cnt_day', 'item_cnt_day_y': 'item_cnt_lag_3'})

# Legger til verdier 5 mnd tilbake
merged_copy_4['date_block_num'] = merged_copy_4['date_block_num'] + 1
merged_copy_joined_5 = pd.merge(joined_group, merged_copy_4, how = 'left', on = ['date_block_num', 'shop_id', 'item_category_id'])
merged_copy_5 = merged_copy_joined_5.rename(columns={'item_cnt_day_x': 'item_cnt_day', 'item_cnt_day_y': 'item_cnt_lag_2'})

# Legger til verdier 6 mnd tilbake
merged_copy_5['date_block_num'] = merged_copy_5['date_block_num'] + 1
merged_copy_joined_6 = pd.merge(joined_group, merged_copy_5, how = 'left', on = ['date_block_num', 'shop_id', 'item_category_id'])
merged_copy_6 = merged_copy_joined_6.rename(columns={'item_cnt_day_x': 'item_cnt_day', 'item_cnt_day_y': 'item_cnt_lag_1'})

# Group By SHOP ID
joined_shop = joined_fact_df[['date_block_num', 'shop_id', 'item_cnt_day']].groupby(['date_block_num', 'shop_id'], as_index = False).sum() 
copy_shop = joined_shop.copy()

# Legger til verdier en mnd tilbake
copy_shop['date_block_num'] = copy_shop['date_block_num'] + 1
merged_copy_shop_1 = pd.merge(joined_shop, copy_shop, how = 'left', on = ['date_block_num', 'shop_id'])
merged_shop_1 = merged_copy_shop_1.rename(columns={'item_cnt_day_x': 'item_cnt_day', 'item_cnt_day_y': 'shop_cnt_lag_3'})

# Legger til verdier to mnd tilbake
merged_shop_1['date_block_num'] = merged_shop_1['date_block_num'] + 1
merged_copy_shop_2 = pd.merge(joined_shop, merged_shop_1, how = 'left', on = ['date_block_num', 'shop_id'])
merged_shop_2 = merged_copy_shop_2.rename(columns={'item_cnt_day_x': 'item_cnt_day', 'item_cnt_day_y': 'shop_cnt_lag_2'})

# Legger til verdier tre mnd tilbake
merged_shop_2['date_block_num'] = merged_shop_2['date_block_num'] + 1
merged_copy_shop_3 = pd.merge(joined_shop, merged_shop_2, how = 'left', on = ['date_block_num', 'shop_id'])
merged_shop_3 = merged_copy_shop_3.rename(columns={'item_cnt_day_x': 'item_cnt_day', 'item_cnt_day_y': 'shop_cnt_lag_1'})

merger_item_shop = pd.merge(merged_copy_6, merged_shop_3, how = 'left', on = ['date_block_num', 'shop_id'])
merger_item_shop = merger_item_shop.rename(columns={'item_cnt_day_x': 'item_cnt_day', 'item_cnt_day_y': 'shop_cnt_day'})

# Group By CATEGORY ID
joined_cat = joined_fact_df[['date_block_num', 'item_category_id', 'item_cnt_day']].groupby(['date_block_num', 'item_category_id'], as_index = False).sum() 
copy_cat = joined_cat.copy()

# Legger til verdier en mnd tilbake
copy_cat['date_block_num'] = copy_cat['date_block_num'] + 1
merged_copy_cat_1 = pd.merge(joined_cat, copy_cat, how = 'left', on = ['date_block_num', 'item_category_id'])
merged_cat_1 = merged_copy_cat_1.rename(columns={'item_cnt_day_x': 'item_cnt_day', 'item_cnt_day_y': 'cat_cnt_lag_3'})

# Legger til verdier to mnd tilbake
merged_cat_1['date_block_num'] = merged_cat_1['date_block_num'] + 1
merged_copy_cat_2 = pd.merge(joined_cat, merged_cat_1, how = 'left', on = ['date_block_num', 'item_category_id'])
merged_cat_2 = merged_copy_cat_2.rename(columns={'item_cnt_day_x': 'item_cnt_day', 'item_cnt_day_y': 'cat_cnt_lag_2'})

# Legger til verdier tre mnd tilbake
merged_cat_2['date_block_num'] = merged_cat_2['date_block_num'] + 1
merged_copy_cat_3 = pd.merge(joined_cat, merged_cat_2, how = 'left', on = ['date_block_num', 'item_category_id'])
merged_cat_3 = merged_copy_cat_3.rename(columns={'item_cnt_day_x': 'item_cnt_day', 'item_cnt_day_y': 'cat_cnt_lag_1'})

merger_item_cat = pd.merge(merger_item_shop, merged_cat_3, how = 'left', on = ['date_block_num', 'item_category_id'])
merger_all = merger_item_cat.rename(columns={'item_cnt_day_x': 'item_cnt_day', 'item_cnt_day_y': 'cat_cnt_day'})

merger_all = merger_all.dropna()
merger_all = merger_all.reset_index()
merger_all = merger_all.drop(axis=1, labels=['index'])

# True Naiv
true_naiv_mse = mean_squared_error(merger_all['item_cnt_day'], merger_all['item_cnt_lag_1'])
true_naiv_mae = mean_absolute_error(merger_all['item_cnt_day'], merger_all['item_cnt_lag_1'])
print(f'True naiv MSE : {np.sqrt(true_naiv_mse)}') # 66.3439
print(f'True Naiv MAE : {true_naiv_mae}') # 24.0542

# Data prep
max_min_features = ['item_cnt_lag_1', 'item_cnt_lag_2', 'item_cnt_lag_3', 'item_cnt_lag_4', 'item_cnt_lag_5', 'item_cnt_lag_6', 'shop_cnt_lag_1', 'shop_cnt_lag_2', 'shop_cnt_lag_3', 'cat_cnt_lag_1', 'cat_cnt_lag_2', 'cat_cnt_lag_3']
min_max_merger_nan = MinMaxScaler()
min_max_merger_nan.fit(merger_all[max_min_features])
scaled_max_min = min_max_merger_nan.transform(merger_all[max_min_features])
merger_all[max_min_features] = scaled_max_min

lst = []
for item in merger_all['date_block_num']:
    month = item%12
    print(month)
    lst.append(month)

merger_all['new_column_month'] = lst
merger_all.columns
print(merger_all)

# Split train, test. Test being month 33, train being the rest
df_train = merger_all[merger_all['date_block_num'] <= 32]
df_test = merger_all[merger_all['date_block_num'] == 33]

# features = df_train['item_cnt_lag_1', 'item_cnt_lag_2', 'item_cnt_lag_3', 'item_cnt_lag_4', 'item_cnt_lag_5', 'item_cnt_lag_6', 'shop_cnt_lag_1', 'shop_cnt_lag_2', 'shop_cnt_lag_3', 'cat_cnt_lag_1', 'cat_cnt_lag_2', 'cat_cnt_lag_3']

X_train = np.c_[df_train[['item_cnt_lag_1', 'item_cnt_lag_2', 'item_cnt_lag_3', 'item_cnt_lag_4', 'item_cnt_lag_5', 'item_cnt_lag_6', 'shop_cnt_lag_1', 'shop_cnt_lag_2', 'shop_cnt_lag_3', 'cat_cnt_lag_1', 'cat_cnt_lag_2', 'cat_cnt_lag_3', 'new_column_month']]]
y_train = np.c_[df_train[['item_cnt_day']]]

X_test = np.c_[df_test[['item_cnt_lag_1', 'item_cnt_lag_2', 'item_cnt_lag_3', 'item_cnt_lag_4', 'item_cnt_lag_5', 'item_cnt_lag_6', 'shop_cnt_lag_1', 'shop_cnt_lag_2', 'shop_cnt_lag_3', 'cat_cnt_lag_1', 'cat_cnt_lag_2', 'cat_cnt_lag_3', 'new_column_month']]]
y_test = np.c_[df_test[['item_cnt_day']]]

print(f'X train : {len(X_train)}')
print(f'y train : {len(y_train)}')
print(f'X test : {len(X_test)}')
print(f'y test : {len(y_test)}')

# XGBoost Model

regressor = xgb.XGBRegressor(
    early_stopping_rounds = 5,
    max_depth = 3
)

regressor.fit(X_train, y_train, eval_set = [(X_test, y_test)])

importance_check = pd.DataFrame(regressor.feature_importances_.reshape(1, -1), columns = [['item_cnt_lag_1', 'item_cnt_lag_2', 'item_cnt_lag_3', 'item_cnt_lag_4', 'item_cnt_lag_5', 'item_cnt_lag_6', 'shop_cnt_lag_1', 'shop_cnt_lag_2', 'shop_cnt_lag_3', 'cat_cnt_lag_1', 'cat_cnt_lag_2', 'cat_cnt_lag_3', 'new_column_month']] )

# sorted_idx = regressor.feature_importances_.argsort()
plt.barh(['item_cnt_lag_1', 'item_cnt_lag_2', 'item_cnt_lag_3', 'item_cnt_lag_4', 'item_cnt_lag_5', 'item_cnt_lag_6','shop_cnt_lag_1', 'shop_cnt_lag_2', 'shop_cnt_lag_3', 'cat_cnt_lag_1', 'cat_cnt_lag_2', 'cat_cnt_lag_3', 'new_column_month'], regressor.feature_importances_)
plt.show()
# Scoring xg

# Train xg
r2_train_xg = regressor.score(X_train, y_train)
print(f'Train R2 XG : {r2_train_xg}') # 0.8494 / 3F 0.8204 / 9F 0.8739 / 13F 0.9318

y_pred_train_xg = regressor.predict(X_train)
mse_train_xg = mean_squared_error(y_train, y_pred_train_xg)
print(f'Train MSE XG : {np.sqrt(mse_train_xg)}') # 54.5943 / 3F 59.6264 / 9F 49.9526 / 13F 36.4051

mae_train_xg = mean_absolute_error(y_train, y_pred_train_xg)
print(f'Train MAE XG : {mae_train_xg}') # 18.7412 / 3F 19.9255 / 9F 17.4201 / 13F 16.3526

# Test xg
r2_test_xg = regressor.score(X_test, y_test)
print(f'Test R2 XG : {r2_test_xg}') # 0.5886 / 0.6134 / 3F 0.5799 / 9F 0.6346 / 13F 0.7753

y_pred_test_xg = regressor.predict(X_test)
mse_test_xg = mean_squared_error(y_test, y_pred_test_xg)
print(f'Test MSE XG : {np.sqrt(mse_test_xg)}') # 66.8690 / 64.8278 / 3F 67.5740 / 9F 63.0247 / 13F 51.9722

mae_test_xg = mean_absolute_error(y_test, y_pred_test_xg)
print(f'Train MAE XG : {mae_test_xg}') # 17.2526 / 17.6242 / 3F 17.6046 / 9F 16.5146 / 13F 15.0856

# KNN

model_knn = KNeighborsRegressor(n_neighbors=5)
model_knn.fit(X_train, y_train)

# KNN score
# Train
R2_train_knn = model_knn.score(X_train, y_train)
y_pred_train_knn = model_knn.predict(X_train)
mae_train_knn = mean_absolute_error(y_train, y_pred_train_knn)
mse_train_knn = mean_squared_error(y_train, y_pred_train_knn)
mape_train_knn = mean_absolute_percentage_error(y_train, y_pred_train_knn)
print(f'Knn Train Score:')
print(f'R2_train KNN : {R2_train_knn}') # 0.8976 / 13F 0.8997
print(f'mae_train KNN : {mae_train_knn}') # 15.1599 / 13F 15.0413
print(f'sqrt(mse_train) KNN : {np.sqrt(mse_train_knn)}') # 45.0022 / 13F 44.1516

# Test
R2_test_knn = model_knn.score(X_test, y_test)
y_test_pred_knn = model_knn.predict(X_test)
mae_test_knn = mean_absolute_error(y_test, y_test_pred_knn)
mse_test_knn = mean_squared_error(y_test, y_test_pred_knn)
mape_test_knn = mean_absolute_percentage_error(y_test, y_test_pred_knn)
print(f'Knn Test Score:')
print(f'R2_test: {R2_test_knn}') # 0.6681 / 13F 0.6457
print(f'mae_test: {mae_test_knn}') # 18.4406 / 13F 21.8839
print(f'sqrt(mse_test): {np.sqrt(mse_test_knn)}') # 60.8170 / 13F 65.2645

# LinearRegression Model 
model = LinearRegression()
model.fit(X_train, y_train)

# Scoring

# Train LR
r2_train = model.score(X_train, y_train)
print(f'R2_train : {r2_train}') # LR = 0.6360 / 0.7917 / 9F 0.7936 / 13F 0.8925

y_pred_train = model.predict(X_train)

mse_train = mean_squared_error(y_train, y_pred_train)
print(f'Train MSE : {np.sqrt(mse_train)}') # LR = 84.8895 / 64.2086 / 9F 63.9242 / 13F 45.6988

mae_train = mean_absolute_error(y_train, y_pred_train)
print(f'Train MAE : {mae_train}') # LR = 28.5480 / 20.2689 / 9F 20.4580 / 13F 19.3374

mape_train = mean_absolute_percentage_error(y_train, y_pred_train)
print(f'Train MAPE : {mape_train}')

# Test LR
r2_test = model.score(X_test, y_test)
print(f'R2_test : {r2_test}') # LR = 0.5752 / 0.6076 / 9F 0.6061 / 13F 0.7177

y_pred_test = model.predict(X_test)

mse_test = mean_squared_error(y_test, y_pred_test)
print(f'Test MSE : {np.sqrt(mse_test)}') # LR = 67.9522 / 65.3109 / 9F 65.4313 / 13F 58.2510

mae_test = mean_absolute_error(y_test, y_pred_test)
print(f'Test MAE : {mae_test}') # LR = 22.1257 / 17.0475 / 9F 17.7192 / 13F 19.4204

mape_test = mean_absolute_percentage_error(y_test, y_pred_test)
print(f'Test MAPE : {mape_test}')

# Naiv scoring

pred_naiv = np.mean(y_train)
y_pred_naiv = np.full(y_test.shape, pred_naiv)
print(f'y_pred_naiv : {len(y_pred_naiv)}')

mse_naiv = mean_squared_error(y_test, y_pred_naiv)
print(f'Naiv MSE : {np.sqrt(mse_naiv)}') # 111.1814

mae_naiv = mean_absolute_error(y_test, y_pred_naiv)
print(f'Naiv MAE : {mae_naiv}') # 57.6039

# Concat to df

# LR
y_pred_train_LR_df = pd.DataFrame(y_pred_train)
y_pred_test_LR_df = pd.DataFrame(y_pred_test)

merger_all['y_pred_LR'] = pd.concat([y_pred_train_LR_df, y_pred_test_LR_df], ignore_index = True) # lagre som model 1

# KNN
y_pred_train_KNN_df = pd.DataFrame(y_pred_train_knn)
y_pred_test_KNN_df = pd.DataFrame(y_test_pred_knn)

merger_all['y_pred_KNN'] = pd.concat([y_pred_train_KNN_df, y_pred_test_KNN_df], ignore_index = True) # lagre som model 1

# XG
y_pred_train_XG_df = pd.DataFrame(y_pred_train_xg)
y_pred_test_XG_df = pd.DataFrame(y_pred_test_xg)

merger_all['y_pred_XG'] = pd.concat([y_pred_train_XG_df, y_pred_test_XG_df], ignore_index = True) # lagre som model 2

# Neste model testing kan legges som en egen kolonne ved siden av for å se utvikling

merger_all

# Gjøre om til CSV

csv_merger_nan = merger_all.copy()
csv_merger_nan.isna().sum()

csv_merger_nan.to_csv('finished.csv', sep='\t')

'''
Eks modeller å prøve
- True Naiv
- Naiv lag_1
- OneHotEncoder category
- OneHotEnconder month
- xgboost S
- Group by Date_num and shop_id F
- Group by Date_num and category M
'''