#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import boxcox
from scipy.stats import chi2_contingency
import datetime
import re
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler


# In[2]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.pipeline import Pipeline


# In[3]:


data = pd.read_csv('train.csv')
explore_data = data.copy()
test_data = pd.read_csv('test.csv')
# set to display every columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
explore_data.head(10)


# In[5]:


# Cleaning the DATA

def check_null_cols(data):
    null_val = data.isnull().sum()
    null_cols = []
#     print(f'These are columns including NULL values')
    for i, v in null_val.items():
        if v > 0:
#             print(f"{i} has {v} null vales")
            null_cols.append(i)
            pass
    return null_cols
null_cols = check_null_cols(explore_data)


# In[6]:


# Droping the Deduplicate data
before_dedup = len(explore_data)
after_dedup = len(explore_data.drop_duplicates())
print(f"Before deduplicated {before_dedup}")
print(f"After deduplicated {after_dedup}")


# In[7]:


# Display column names
print("Column Names:")
print(explore_data.columns.tolist())


# In[8]:


# Handle special outlier case
explore_data.loc[(explore_data['Electrical'].isnull()), 'Electrical'] = explore_data['Electrical'].mode().iloc[0]
explore_data.loc[(explore_data['BsmtFinType2'].isnull()) & (explore_data['BsmtFinSF2'] > 0), 'BsmtFinType2'] = explore_data['BsmtFinType2'].mode().iloc[0]


# In[9]:


def num_cat_selection(data):
    tar_col = ['SalePrice']
    data_num = data.select_dtypes(include=['int64', 'float64']).drop(['Id', 'SalePrice'], axis=1)
    data_num = data_num.drop('MSSubClass',axis=1) #it's categorical feature not numerical feature
    num_cols = list(data_num.columns)
    cat_cols = []
    for i in data.columns:
        if i not in data_num.columns:
            cat_cols.append(i)
    cat_cols.remove('Id')
    cat_cols.remove('SalePrice')
    return num_cols, cat_cols, tar_col

num_cols, cat_cols, tar_col = num_cat_selection(explore_data)


# In[10]:


# Define function for fillna value
def fill_missing_value(data, null_cols, num_cols, cat_cols):
    for i in range(len(null_cols)):
        if null_cols[i] in num_cols:
            data[null_cols[i]].fillna(0, inplace=True)
        elif null_cols[i] in cat_cols:
            data[null_cols[i]].fillna('None', inplace=True)
    return data

explore_data = fill_missing_value(explore_data, null_cols, num_cols, cat_cols)


# In[11]:


# convert from float64 to int64 (datetime is also optional, but I would use it for additional calculation later so int64 is best for me)
def convert_col_type(data):
    data['GarageYrBlt'] = data['GarageYrBlt'].astype('int64')
    return data
    
explore_data = convert_col_type(explore_data)


# In[12]:


# Performoing EDA on the data
explore_data.head()


# In[13]:


# defining additional features by feature engineering technique

def feature_engineering(dataset):
    dataset['TotalAge'] = dataset['YrSold'] - dataset['YearBuilt']
    dataset['AgeSinceLastMod'] = dataset['YrSold'] - dataset['YearRemodAdd']

feature_engineering(explore_data)

# Adding numerical features, then variable 'num_cols' need to be updated
num_cols, cat_cols, tar_col = num_cat_selection(explore_data)


# In[14]:


# Analyze numerical distribution data
explore_data_num = explore_data[num_cols]
explore_data_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)


# In[15]:


# further explore at explainability of 'SalePrice' thorugh each feature
def scatter_plot(data, num_cols, bc_tr ,target):
    dataset = data.copy()
    if len(num_cols)//4==0:
        num_rows = len(num_cols)//4
    else:
        num_rows = len(num_cols)//4 + 1
    
    fig = plt.figure(figsize=(30,30))
    if bc_tr==1: 
        for i in range(len(num_cols)):
            plt.subplot(num_rows, 4, i+1)
            # Adding a small value to prevent zero values
            col = num_cols[i]
            dataset[col] = np.log(dataset[col])
            dataset[target] = np.log(dataset[target])
            sns.regplot(data=dataset[[col, target]], x=col, y=target)
            plt.title(f"Scatter of {col} with log transform and sale price")
            plt.xlabel(f"Transform {col}")
    else:
        for i in range(len(num_cols)):
            plt.subplot(num_rows, 4, i+1)
            col = num_cols[i]
            sns.regplot(data=dataset[[col, target]], x=col, y=target)
            plt.title(f"Scatter of {col} with no transform")
            plt.xlabel(f"{col}")
    
    fig.tight_layout(pad=1.0)
    plt.ylabel(f"SalePrice")
    plt.show()


scatter_plot(explore_data, num_cols, 0, 'SalePrice')


# In[16]:


# Then now distribution of each feature
def box_plot(data, num_cols):
    if len(num_cols)//4==0:
        num_rows = len(num_cols)//4
    else:
        num_rows = len(num_cols)//4 + 1


    fig = plt.figure(figsize=(30,30))
    for i in range(len(num_cols)):
        plt.subplot(num_rows, 4, i+1)
        sns.boxplot(data=data, x=num_cols[i])

    fig.tight_layout(pad=1.0)
    plt.show()


box_plot(explore_data, num_cols)


# In[17]:


# remove outliers in some numerical columns which in that column there are some outliers but not the major data in that column

def remove_outliers(data, columns, threshold=1.5):
    df_out = data.copy()
    for column in columns:
        q1 = df_out[column].quantile(0.25)
        q3 = df_out[column].quantile(0.75)
        iqr = q3-q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        df_out = df_out[(df_out[column] >= lower_bound) & (df_out[column] <= upper_bound)]
    return df_out

explore_data = remove_outliers(explore_data, ["LotFrontage", "LotArea", "BsmtFinSF1", "TotalBsmtSF", "GrLivArea", "MasVnrArea", "GarageArea"])
box_plot(explore_data, num_cols)
    


# In[18]:


len(explore_data)


# In[19]:


# Now consider target variable's distribution
def distribution_plot(data, col, bc_tr):
    dataset = data.copy()
    if bc_tr==1:
        #Adding a small value to prevent zero values
        dataset[col] = np.log10(dataset[col])
        sns.histplot(data=dataset[col], kde=True)
        plt.title(f"Histogram of {col} with log transform")
        plt.xlabel(f"Transform {col}")
    else:
        sns.histplot(data=dataset[col], kde=True)
        plt.title(f"Histogram of {col} with no transform")
        plt.xlabel(f"{col}")
    
    plt.ylabel(f"Count")
    plt.show()

distribution_plot(explore_data, 'SalePrice', 0)


# In[20]:


distribution_plot(explore_data, 'SalePrice', 1)


# In[21]:


#Feature Selection

# Variance Threshold for numerical columns
# remove some numerical columns which has low variance (imply less information)

explore_data_num = explore_data[num_cols]
var = VarianceThreshold(0.1) #threshold is adjustable
var.fit(explore_data_num)
boolean_selection = var.get_support()
col_names = var.feature_names_in_

remove_cols = set()
for i in range(len(boolean_selection)):
    if boolean_selection[i] == False:
        remove_cols.add(col_names[i])

remove_cols = list(remove_cols)
for j in range(len(remove_cols)):
    num_cols.remove(remove_cols[j])

explore_data = explore_data.drop(remove_cols, axis=1)
explore_data.head()


# In[22]:


#mutual information for regression for considering feature importance of each numerical features

explore_data_num = explore_data[num_cols]
mutual_info = mutual_info_regression(explore_data_num, explore_data.SalePrice)
mutual_info = pd.Series(mutual_info)
mutual_info.index = explore_data_num.columns
mutual_info.sort_values(ascending=False)


# In[23]:


#perform SelectKBest to select the K most important numerical features
select_k = SelectKBest(mutual_info_regression, k=20)
select_k.fit(explore_data_num, explore_data.SalePrice)
boolean_selection = select_k.get_support()
remove_cols = set()
for i in range(len(boolean_selection)):
    if boolean_selection[i] == False:
        remove_cols.add(num_cols[i])

remove_cols = list(remove_cols)
for j in range(len(remove_cols)):
    num_cols.remove(remove_cols[j])

explore_data = explore_data.drop(remove_cols, axis=1)
explore_data.head()


# In[24]:


# Then now we consider correlation between retained numerical features
explore_data_num = explore_data[num_cols]

corr_mat = explore_data_num.corr()

sns.heatmap(corr_mat)


# In[25]:


# remove strong correlation numerical features
def remove_strong_corr(data, neg_corr, pos_corr):
    remove_cols = set()
    corr_mat = data.corr()
    
    for i in range(len(data.columns)):
        for j in range(i):
            corr_val = corr_mat.iloc[i, j]
            if corr_val > 0 and corr_val > pos_corr:
                remove_cols.add(corr_mat.columns[j])
            elif corr_val < 0 and abs(corr_val) > neg_corr:
                remove_cols.add(corr_mat.columns[j])
    return remove_cols

remove_cols = remove_strong_corr(explore_data_num, 0.8, 0.8)
remove_cols


# In[26]:


len(num_cols)


# In[27]:


remove_cols = list(remove_cols)
for i in range(len(remove_cols)):
    num_cols.remove(remove_cols[i])

explore_data = explore_data.drop(remove_cols, axis=1)
explore_data.head()


# In[28]:


len(num_cols)


# In[29]:


#encode categorical data by OrdinalEncoder
#LabelEnocder in sklearn is for target variable not features
#If ordinal of features is crucial, then TargetEncoder() would be more appropriated

explore_data_cat = explore_data[cat_cols]
enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
enc.fit(explore_data_cat)
encoded_explored_data_cat = enc.transform(explore_data_cat)
explore_data_cat = pd.DataFrame(encoded_explored_data_cat, columns=explore_data_cat.columns)

explore_data_cat.head()


# In[30]:


#Also perform SelectKBest on categorical features
select_k = SelectKBest(mutual_info_regression, k=20)
select_k.fit(explore_data_cat, explore_data.SalePrice)
boolean_selection = select_k.get_support()
remove_cols = set()
for i in range(len(boolean_selection)):
    if boolean_selection[i] == False:
        remove_cols.add(cat_cols[i])

remove_cols = list(remove_cols)
for j in range(len(remove_cols)):
    cat_cols.remove(remove_cols[j])

    
explore_data = explore_data.drop(remove_cols, axis=1)
explore_data.head()


# In[31]:


explore_data_cat = explore_data[cat_cols]
explore_data_cat.head()


# In[ ]:





# In[ ]:





# In[32]:


#Preprocessing Data for mode

def scale_data(train_data, test_data, num_cols):
    train_data_num = train_data[num_cols]
    test_data_num = test_data[num_cols]
    std_scaler = StandardScaler()
    std_scaler.fit(train_data_num)
    train_data_num_scaled = std_scaler.transform(train_data_num)
    test_data_num_scaled = std_scaler.transform(test_data_num)
    train_data_num_scaled = pd.DataFrame(data=train_data_num_scaled, columns=train_data_num.columns)
    test_data_num_scaled = pd.DataFrame(data=test_data_num_scaled, columns=test_data_num.columns)
    
    return train_data_num_scaled, test_data_num_scaled


# In[33]:


def enc_data(train_data, test_data, cat_cols):
    train_data_cat = train_data[cat_cols]
    test_data_cat = test_data[cat_cols]
    onehot_enc = OneHotEncoder(handle_unknown='ignore')
    onehot_enc.fit(train_data_cat)
    train_data_cat_encoded = onehot_enc.transform(train_data_cat).toarray()
    test_data_cat_encoded = onehot_enc.transform(test_data_cat).toarray()
    cols = onehot_enc.get_feature_names_out()
    train_data_cat_encoded = pd.DataFrame(data=train_data_cat_encoded, columns=cols)
    test_data_cat_encoded = pd.DataFrame(data=test_data_cat_encoded, columns=cols)
    return train_data_cat_encoded, test_data_cat_encoded


# In[34]:


def combine_num_cat(data_num, data_cat):
    data = pd.concat([data_num, data_cat], axis=1)
    return data
    
def log_transform(data, col):
    for c in col:
        data[c] = np.log10(data[c])
    return data

def output_inverse_transform(data):
    data =  np.power(10, data)
    return data

def check_mismatch_cols(train_data, test_data):
    test_cols = test_data.columns
    train_cols = train_data.columns
    for i in range(len(test_cols)):
        if test_cols[i] not in train_cols:
            test_data = test_data.drop(test_cols[i], axis=1)
    return train_data, test_data


# In[39]:


# Define selected numerical features and categorical features to use in next part
num_cols_selected = num_cols
cat_cols_selected = cat_cols


# In[40]:


data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
ids = test_data['Id']

def pre_process_pipeline(train_data, test_data, num_cols_selected, cat_cols_selected):
    train_data = train_data.drop_duplicates()
    test_data = test_data.drop_duplicates()
    train_data, test_data = check_mismatch_cols(train_data, test_data)
    
    num_cols, cat_cols, tar_col = num_cat_selection(train_data)
    
    
    train_null_cols = check_null_cols(train_data)
    test_null_cols = check_null_cols(test_data)
    
    
    train_data = fill_missing_value(train_data,train_null_cols, num_cols, cat_cols)
    test_data = fill_missing_value(test_data,test_null_cols, num_cols, cat_cols)
   
    
    train_data = convert_col_type(train_data)
    test_data = convert_col_type(test_data)
    
#     train_data = remove_outliers(train_data, ["LotFrontage", "LotArea", "BsmtFinSF1", "TotalBsmtSF", "GrLivArea", "MasVnrArea", "GarageArea"])
    
    feature_engineering(train_data)
    feature_engineering(test_data)
    
#     remove_cols = find_same_val_col(explore_data, 85)
#     train_data = train_data.drop(columns=remove_colss, axis=1)
#     test_data = test_data.drop(columns=remove_colss, axis=1)
    

    train_data_target = train_data[tar_col]
    train_data_target_transformed = log_transform(train_data_target, tar_col)
    # This rest_index and drop is needed because, in scale and encoded function I rebuild DataFrame, so the index is re-run from the start
    # If don't this the indexes will be mismatched
    train_data_target_transformed.reset_index(drop=True, inplace=True)
    
      
    train_data_num_scaled, test_data_num_scaled = scale_data(train_data, test_data, num_cols_selected)
    train_data_cat_encoded, test_data_cat_encoded = enc_data(train_data, test_data, cat_cols_selected)
    
    train_data = combine_num_cat(train_data_num_scaled, train_data_cat_encoded)
    train_data = pd.concat([train_data, train_data_target_transformed], axis=1)
    test_data = combine_num_cat(test_data_num_scaled, test_data_cat_encoded)
    
    return train_data, test_data

train_data, test_data = pre_process_pipeline(data, test_data, num_cols_selected, cat_cols_selected)


# In[41]:


train_data.head()


# In[42]:


test_data.head()


# In[43]:


# train, dev set split

X_train, X_dev, y_train, y_dev = train_test_split(train_data.drop(columns=['SalePrice']), train_data['SalePrice'], test_size = 0.3, random_state=10)


# In[44]:


# Create DataFrame to correct result from models
res_cols = ['model', 'MAE', 'MSE', 'RMSE', 'r2_train', 'r2_dev']
model_result = pd.DataFrame(columns=res_cols)


# In[45]:


# Lgbm regression
lgbm_reg = LGBMRegression()

param_grid = {
    'fit_intercept': [True, False]
}

gs = GridSearchCV(estimator=lin_reg, param_grid = param_grid, cv=5)
gs.fit(X_train, y_train)

best_model = gs.best_estimator_
best_score = gs.best_score_
best_params = gs.best_params_
score = gs.cv_results_

predict_training = best_model.predict(X_train)
r2_train = r2_score(y_true=y_train, y_pred=predict_training)

predict_dev = best_model.predict(X_dev)
r2_dev = r2_score(y_true=y_dev, y_pred=predict_dev)

MAE = mean_absolute_error(y_true=y_dev, y_pred=predict_dev)
MSE = mean_squared_error(y_true=y_dev, y_pred=predict_dev)
RMSE = np.sqrt(MSE)

model_scores = { 
    'model' : 'LGBM',
    'MAE': MAE,
    'MSE': MSE,
    'RMSE' : RMSE,
    'r2_train' : r2_train,
    'r2_dev' : r2_dev
}

model_result.loc[len(model_result)] = model_scores

print(f"Best paremeter is {best_params}")
model_result.head()


# In[46]:


#Random Forest


RF = RandomForestRegressor(n_estimators= 50, max_depth=10, min_samples_leaf=5, min_samples_split=5)

RF.fit(X_train, y_train)

predict_training = RF.predict(X_train)
r2_train = r2_score(y_true=y_train, y_pred=predict_training)

predict_dev = RF.predict(X_dev)
r2_dev = r2_score(y_true=y_dev, y_pred=predict_dev)

MAE = mean_absolute_error(y_true=y_dev, y_pred=predict_dev)
MSE = mean_squared_error(y_true=y_dev, y_pred=predict_dev)
RMSE = np.sqrt(MSE)

model_scores = { 
    'model' : 'RandomForestRegressor',
    'MAE': MAE,
    'MSE': MSE,
    'RMSE' : RMSE,
    'r2_train' : r2_train,
    'r2_dev' : r2_dev
}

model_result.loc[len(model_result)] = model_scores


best_params = {
    'n_estimators': 50,
    'max_depth':10,
    'min_samples_leaf':5,
    'min_samples_split':5
    
    
}
print(f"Best paremeter is {best_params}")
model_result


# In[48]:


get_ipython().system('pip install shap')


# In[51]:


get_ipython().system('pip install lime')


# In[53]:


from shapash.explainer.smart_explainer import SmartExplainer

# Initialize SmartExplainer
xpl = SmartExplainer()
xpl.compile(
    x=X_test,  # Test features
    model=model,
    preprocessing=preprocessing_function,  # If any preprocessing is applied
    y_pred=y_pred,  # Model predictions on test set
)


# In[54]:


get_ipython().system('pip install shapash')


# In[55]:


from shapash.explainer.smart_explainer import SmartExplainer

# Initialize SmartExplainer
xpl = SmartExplainer()
xpl.compile(
    x=X_test,  # Test features
    model=model,
    preprocessing=preprocessing_function,  # If any preprocessing is applied
    y_pred=y_pred,  # Model predictions on test set
)


# In[56]:


pip install shapash


# In[57]:


from shapash.explainer.smart_explainer import SmartExplainer

# Initialize SmartExplainer
xpl = SmartExplainer()
xpl.compile(
    x=X_test,  # Test features
    model=model,
    preprocessing=preprocessing_function,  # If any preprocessing is applied
    y_pred=y_pred,  # Model predictions on test set
)


# In[62]:


import lightgbm as lgb


# Create a LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)


# Set hyperparameters for the LightGBM model
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mse',  # Mean Squared Error
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

# Train the LightGBM model
num_round = 100
lgb_model = lgb.train(params, train_data, num_round, valid_sets=[test_data], early_stopping_rounds=10)



# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


# In[ ]:




model_lgbm = LGBMRegressor(verbose = -1)
model_lgbm.fit(X_train, y_train)
predictions_lgbm = model_lgbm.predict(X_val)

