#!/usr/bin/env python
# coding: utf-8

# In[77]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 500)


# In[2]:


def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)


# In[24]:



def load_data(data,sheet_name):
    df = pd.read_excel(data)
    return df


# In[25]:




# In[6]:





# In[26]:


def create_lags(df, cols, lag_list):

    for i in cols:
        for j in lag_list:
            df[i+'_LAG_'+str(j)] = df[i].shift(j)

    return df


# In[28]:


def create_lag(df):
    df1 = df.fillna(0)
    long_lag = [1,3,5,7,10,14,21,30]
    mid_lag = [7,10,21,30]


    df1['FACEBOOK/INSTAGRAM_SPEND'] = df1['FACEBOOK_SPEND']+df1['INSTAGRAM_SPEND']
    long_lag_cols = ['FACEBOOK/INSTAGRAM_SPEND']
    mid_lag_cols = ['TV_spend']

    df1 = create_lags(df1, long_lag_cols, long_lag)
    df1 = create_lags(df1, mid_lag_cols, mid_lag)
    return(df1)


# In[29]:



# In[30]:


def generate_model_results(df, iterartion):

    x = df.drop(['OVERALL_VOLUME', 'DATE'], axis =1)
#     x.insert(0, "INTERCEPT" , 1)
    y=df[["OVERALL_VOLUME"]]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 42)
    X_train.replace([np.inf, -np.inf], 0, inplace=True)

    model = sm.OLS(y_train, X_train).fit()
    Vif = calc_vif(x)

    result_df = pd.DataFrame()
    result_df['FEATURE'] = x.columns
    result_df['COEFF'] = list(model.summary2().tables[1]['Coef.'])
    result_df['P_VALUE'] = list(round(model.pvalues, 5))

    model_iteration_new=pd.DataFrame()
    model_iteration_all=pd.DataFrame()
    model_columns_used=pd.DataFrame()
    model_score=pd.DataFrame()

    sum_df = pd.DataFrame(x.sum(axis = 0, skipna = True)).reset_index()
    sum_df.columns =  ['FEATURE', 'SUM']
    sum_df = sum_df[sum_df['FEATURE'].isin(list(result_df['FEATURE'].unique()))].reset_index(drop= True)

    result_df = result_df.merge(sum_df, left_on = 'FEATURE', right_on = 'FEATURE', how = 'left')
    result_df = result_df.merge(Vif,left_on = 'FEATURE', right_on = 'variables', how = 'left')
    result_df = result_df.drop(columns = 'variables')
    y_test['PREDICTIONS'] =  model.predict(X_test)

    print('########## RMSE ###########')
    y_test['squared_error'] = (y_test['OVERALL_VOLUME'] - y_test['PREDICTIONS'])**2
    rmse = np.sqrt(y_test['squared_error'].sum()/len(y_test))
    print(rmse)

    print('########## MAPE ###########')
    y_test['error'] = abs(y_test['OVERALL_VOLUME'] - y_test['PREDICTIONS'])
    mape = (sum(y_test['error']/y_test['OVERALL_VOLUME']))/len(y_test)
    print(mape)

    print("####### PRMSE ##############")

    prmse = rmse/y_test['OVERALL_VOLUME'].mean()
    print(prmse)

    adj_rsq = model.rsquared_adj
    rsq = model.rsquared
    print("rsquared\n",model.rsquared)
    print("rsquared_adj\n",model.rsquared_adj)

    y_test.drop(['squared_error', 'error'], axis =1, inplace = True)

    Total_vol = y['OVERALL_VOLUME'].sum()

#     result_df = pd.merge(result_df, Vif, on ='FEATURE',how='left')

#     iterartion+=1
    model_columns_used=model_columns_used.append(pd.DataFrame({'iteration':iterartion,'columns_list':list(df.columns)})).reset_index(drop='yes')
    result_df['iteration']=iterartion
    model_iteration_all=model_iteration_all.append(result_df).reset_index(drop='yes')
    model_score=model_score.append(pd.DataFrame({'iteration':iterartion,'RMSE': [rmse], 'MAPE': [mape], 'PRMSE': [prmse] , 'ADJ_RSQUARED': [adj_rsq] , 'RSQUARED': [rsq]}).T.reset_index()).reset_index(drop='yes')

    return result_df, model, x, y_test


# In[12]:


#df1[['AMAZON_WTD_DISCOUNT%', 'FLIPKART_WTD_DISCOUNT%',]].plot(kind = 'line')


# In[31]:


# new_df = df[(df['Month']>'2020-06')].reset_index(drop=True)

def create_features(df):

    new_df = df.copy()

    model_cols = [          'DATE',
                            'ISWEEKEND',

#                             'AMS_DISPLAY_SPEND',
                             'AMS_SEARCH_SPEND',
                             'AMS_SEARCH_SP_SPEND',
#                              'DISPLAY_SPEND',
#                              'FACEBOOK_SPEND',
#                              'G_ADWORDS_SPEND',
#                              'INSTAGRAM_SPEND',
#                              'YOUTUBE_SPEND',

                             'JBP_AMAZON',
                             'JBP_BIG_BASKET',

                             'PLA_SPENDS',
#                              'PCA_SPENDS',
                             'JBP_FLIPKART_TRUE',

#                               'TV_spend',

#                             'AMS_DISPLAY_SPEND_LAG_1',
                             'AMS_DISPLAY_SPEND_LAG_3',
#                              'AMS_DISPLAY_SPEND_LAG_5',
#                              'AMS_DISPLAY_SPEND_LAG_7',
#                              'AMS_DISPLAY_SPEND_LAG_10',
#                              'AMS_DISPLAY_SPEND_LAG_14',
#                              'AMS_DISPLAY_SPEND_LAG_21',
#                              'AMS_DISPLAY_SPEND_LAG_30',

#                              'AMS_SEARCH_SPEND_LAG_1',
#                              'AMS_SEARCH_SPEND_LAG_3',
#                              'AMS_SEARCH_SPEND_LAG_5',
#                              'AMS_SEARCH_SPEND_LAG_7',
#                              'AMS_SEARCH_SPEND_LAG_10',
#                              'AMS_SEARCH_SPEND_LAG_14',
#                              'AMS_SEARCH_SPEND_LAG_21',
#                              'AMS_SEARCH_SPEND_LAG_30',
#                              'AMS_SEARCH_SP_SPEND_LAG_1',
#                              'AMS_SEARCH_SP_SPEND_LAG_3',
#                              'AMS_SEARCH_SP_SPEND_LAG_5',
#                              'AMS_SEARCH_SP_SPEND_LAG_7',
#                              'AMS_SEARCH_SP_SPEND_LAG_10',
#                              'AMS_SEARCH_SP_SPEND_LAG_14',
#                              'AMS_SEARCH_SP_SPEND_LAG_21',
#                              'AMS_SEARCH_SP_SPEND_LAG_30',

#                              'DISPLAY_SPEND_LAG_1',
#                              'DISPLAY_SPEND_LAG_3',
#                              'DISPLAY_SPEND_LAG_5',
#                              'DISPLAY_SPEND_LAG_7',
#                              'DISPLAY_SPEND_LAG_10',
#                              'DISPLAY_SPEND_LAG_14',
                             'DISPLAY_SPEND_LAG_21',
#                              'DISPLAY_SPEND_LAG_30',

#                              'FACEBOOK_SPEND_LAG_1',
#                              'FACEBOOK_SPEND_LAG_3',
#                              'FACEBOOK_SPEND_LAG_5',
#                              'FACEBOOK_SPEND_LAG_7',
#                              'FACEBOOK_SPEND_LAG_10',
#                              'FACEBOOK_SPEND_LAG_14',
#                              'FACEBOOK_SPEND_LAG_21',
#                              'FACEBOOK_SPEND_LAG_30',

#                              'G_ADWORDS_SPEND_LAG_1',
#                              'G_ADWORDS_SPEND_LAG_3',
#                              'G_ADWORDS_SPEND_LAG_5',
#                              'G_ADWORDS_SPEND_LAG_7',
#                              'G_ADWORDS_SPEND_LAG_10',
#                              'G_ADWORDS_SPEND_LAG_14',
#                              'G_ADWORDS_SPEND_LAG_21',
#                              'G_ADWORDS_SPEND_LAG_30',

#                              'INSTAGRAM_SPEND_LAG_1',
#                              'INSTAGRAM_SPEND_LAG_3',
#                              'INSTAGRAM_SPEND_LAG_5',
#                              'INSTAGRAM_SPEND_LAG_7',
#                              'INSTAGRAM_SPEND_LAG_10',
#                              'INSTAGRAM_SPEND_LAG_14',
#                              'INSTAGRAM_SPEND_LAG_21',
#                              'INSTAGRAM_SPEND_LAG_30',

#                              'YOUTUBE_SPEND_LAG_1',
#                              'YOUTUBE_SPEND_LAG_3',
                             'YOUTUBE_SPEND_LAG_5',
#                              'YOUTUBE_SPEND_LAG_7',
#                              'YOUTUBE_SPEND_LAG_10',
#                              'YOUTUBE_SPEND_LAG_14',
#                              'YOUTUBE_SPEND_LAG_21',
#                              'YOUTUBE_SPEND_LAG_30',

#                              'JBP_AMAZON_LAG_15',
#                              'JBP_AMAZON_LAG_30',

#                              'JBP_BIG_BASKET_LAG_15',
#                              'JBP_BIG_BASKET_LAG_30',

#                              'PLA_SPENDS_LAG_15',
#                              'PLA_SPENDS_LAG_30',

                             'PCA_SPENDS_LAG_15',
#                              'PCA_SPENDS_LAG_30',

#                              'JBP_FLIPKART_TRUE_LAG_15',
#                              'JBP_FLIPKART_TRUE_LAG_30',

                             'FACEBOOK/INSTAGRAM_SPEND',
#                              'FACEBOOK/INSTAGRAM_SPEND_LAG_1',
#                              'FACEBOOK/INSTAGRAM_SPEND_LAG_3',
#                              'FACEBOOK/INSTAGRAM_SPEND_LAG_5',
#                              'FACEBOOK/INSTAGRAM_SPEND_LAG_7',
#                              'FACEBOOK/INSTAGRAM_SPEND_LAG_10',
#                              'FACEBOOK/INSTAGRAM_SPEND_LAG_14',
#                              'FACEBOOK/INSTAGRAM_SPEND_LAG_21',
#                              'FACEBOOK/INSTAGRAM_SPEND_LAG_30',

                              'TV_spend_LAG_7',
#                                  'TV_spend_LAG_10',
#                                  'TV_spend_LAG_21',
#                                  'TV_spend_LAG_30'

                            'AMAZON_WTD_DISCOUNT%',
                            'FLIPKART_WTD_DISCOUNT%',
#                             'FLIPKART_WTD_PPML_DISCOUNT%',

#                              'Masala Oats_MAGGI_SP_INDEX',
#                              'Masala Oats_QUAKER MASALA OATS_SP_INDEX',
                             'Masala Oats_SAFFOLA OATS_SP_INDEX',
#                              'Masala Oats_SOLIMO_SP_INDEX',
#                              'Masala Oats_SUNFEAST YIPPEE_SP_INDEX',
                             'Masala Oats_WAI WAI_SP_INDEX',
#                              'Masala Oats_YOGA BAR_SP_INDEX',

#                          'Flipkart Shop from Home Days',
#                          'Flipkart Mega Big Savings Day',
#                          'FLIPKART BIG SAVINGS DAY',
                         'FLIPKART PAY DAY SALES',
                         'FLIPKART BIG BILLION DAYS',
#                          'Amazon Freedom Sales',
                         'AMAZON_GREAT INDIA FESTIVAL',
                         'Amazon SVD',
                         'FLIPKART BIG DIWALI SALE',
#                          'FLIPKART DECEMBER SALES',
#                          'FLIPKART BIG SAVINGS Day_Republic Day',
#                          'Flipkart New Year Sale',
#                          "Flipkart Women's Day Sale",
#                          'Flipkart Big Bachath Dhamaal',
#                          'Amazon Super Value Days',
#                          'Amazon Great Republic Day Sale',
#                          "Amazon Women's Day Sales",

                            'OVERALL_VOLUME',

#                         'SPRINKLER_AMAZONIN_RATING_RATING',
#                          'SPRINKLER_FLIPKARTCOM_RATING_RATING',
#                          'SPRINKLER_NYKAACOM_RATING_RATING',

#                              'POSITIVE_MENTIONS',
#                              'NEGATIVE_MENTIONS',
#                              'NEUTRAL_MENTIONS',

#                   'PRINT_NET_SENTIMENT_MENTIONS',
#                  'WORDPRESS_NET_SENTIMENT_MENTIONS',
#                  'FACEBOOK_NET_SENTIMENT_MENTIONS',
#                  'NEWS_NET_SENTIMENT_MENTIONS',
#                  'REVIEWS_NET_SENTIMENT_MENTIONS',
# #                  'BLOGS/WEBSITES_NET_SENTIMENT_MENTIONS',
#                  'FORUMS_NET_SENTIMENT_MENTIONS',
# #                  'TUMBLR_NET_SENTIMENT_MENTIONS',
#                  'TWITTER_NET_SENTIMENT_MENTIONS',
#                  'VIDEOS_NET_SENTIMENT_MENTIONS',
#                  'YOUTUBE_NET_SENTIMENT_MENTIONS',
#                  'INSTAGRAM_NET_SENTIMENT_MENTIONS',
#                  'AUTOMATIC_ALERTS_NET_SENTIMENT_MENTIONS',
#                  'REDDIT_NET_SENTIMENT_MENTIONS',
#                  'DATA_INGESTION_NET_SENTIMENT_MENTIONS',

                             'NET_SENTIMENT_MENTIONS'


                 ]

    model_cols1 = []

    for i in model_cols:
        if i not in [ 'Masala Oats_WAI WAI_SP_INDEX', 'Masala Oats_SAFFOLA OATS_SP_INDEX',
                    ]:
            model_cols1.append(i)

    sub_df = new_df[model_cols1]
    # sub_df.rename(columns={'total': 'DATE'}, inplace=True)
    sub_df.fillna(0, inplace=True)

    return sub_df


# In[32]:





# In[36]:


def built_coefficients(df2):

    iteration=0
    iteration = iteration+1
    results_df, model, xvars, yvar = generate_model_results(df2, iteration)
    results_df
    return(model, xvars, yvar)

def run_model():
    df=load_data(r"C:\Users\SHUBHAM AGNIHOTRI\Desktop\MCA_GUI\dataset\Masala_Oats_Dataset.xlsx", sheet_name = 'Dataset')
    #df = pd.read_excel(r"C:\Users\SHUBHAM AGNIHOTRI\Desktop\MCA_GUI\Masala_Oats_Dataset.xlsx", sheet_name = 'Dataset')
    df1=create_lag(df)
    df2 = create_features(df1)
    model,xvars, yvar=built_coefficients(df2)
    #coeff.to_csv("coeff\coeff.csv")
    return(model,xvars, yvar,df1,df2,df)

model,xvars, yvar,df1,df2,df=run_model()
# In[40]:





# In[ ]:





# In[83]:


results_df=pd.read_csv("C:/Users/SHUBHAM AGNIHOTRI/Desktop/MCA_GUI/coeff/coeff.csv")


# In[84]:


def product_contribution_results(result_df):
    result_df['PRODUCT'] = result_df['SUM']*result_df['COEFF']
    result_df['CONTRIBUTION'] = round(100*result_df['PRODUCT']/result_df['PRODUCT'].sum(),3)
    result_df['ABS_CONTRIBUTION'] = abs(result_df['CONTRIBUTION'])
    result_df = result_df.sort_values('ABS_CONTRIBUTION', ascending= False)
    result_df.drop('ABS_CONTRIBUTION', axis=1, inplace = True)
    return (result_df)


# In[85]:


results_df=product_contribution_results(results_df)


# In[96]:


def create_base_list(data):
    
    data['BASE/INCREMENTAL'] = np.nan
    for j, row in data.iterrows():
#         print(row['FEATURE'])
        i = row['FEATURE']
#         print(i)
        if 'INTERCEPT' in i.upper() or 'SP_INDEX' in i.upper() or 'SENTIMENT' in i.upper() or 'NEUTRAL' in i.upper() or 'RATING' in i.upper():
            data.loc[j,'BASE/INCREMENTAL'] = 'BASE'
        else:
            data.loc[j,'BASE/INCREMENTAL'] = 'INCREMENTAL'
        
    return data[['FEATURE', 'COEFF', 'P_VALUE', 'SUM', 'PRODUCT', 'CONTRIBUTION', 'BASE/INCREMENTAL']]

results_df1 = create_base_list(results_df)


# In[97]:


dis_list = ['AMAZON_WTD_DISCOUNT%', 'FLIPKART_WTD_DISCOUNT%']

def calculate_base_discount(feat_df, data, discount_list, percentile_value):
    df= pd.DataFrame()
    copy_df = feat_df.copy()
    copy_df = copy_df[~copy_df['FEATURE'].isin(discount_list)].reset_index(drop = True)
#     print(copy_df.columns)
    for i in discount_list:
        print(i)
        discount_sum = data[i].sum()
        print(discount_sum.sum())
        base_discount = data[i].quantile(percentile_value)
        print(base_discount)
        data[i+'_NOT_BASE'] = data[i] - base_discount

        index_num = feat_df.index[feat_df['FEATURE']==i].tolist()[0]
        coeff = feat_df.loc[index_num, 'COEFF']
        p_value = feat_df.loc[index_num, 'P_VALUE']
        
        incremental_percentage = round(sum(data[i+'_NOT_BASE'])/data[i].sum(),2)
        base_percentage = 1 - incremental_percentage
        
        print('base %', base_percentage)
        print('incremental %', incremental_percentage)
#         print(incremental_percentage)
#         copy_df.loc[index_num, 'SUM'] = incremental_percentage * discount_sum
        new_df = pd.DataFrame(columns = ['FEATURE', 'COEFF', 'P_VALUE', 'SUM', 'PRODUCT', 'CONTRIBUTION',
       'BASE/INCREMENTAL'])
        
        new_df.loc[0, 'FEATURE'] = 'BASE_'+i
        new_df.loc[0, 'COEFF'] = coeff
        new_df.loc[0, 'P_VALUE'] = p_value
        new_df.loc[0, 'SUM'] = base_percentage * discount_sum
        new_df.loc[0, 'BASE/INCREMENTAL'] = 'BASE'
        
        new_df.loc[1, 'FEATURE'] = 'INCREMENTAL_'+i
        new_df.loc[1, 'COEFF'] = coeff
        new_df.loc[1, 'P_VALUE'] = p_value
        new_df.loc[1, 'SUM'] = incremental_percentage * discount_sum
        
        new_df.loc[1, 'BASE/INCREMENTAL'] = 'INCREMENTAL'
        
        copy_df = pd.concat([new_df, copy_df], axis=0).reset_index(drop = True)
#         print(new_df)

    copy_df['SUM'] = copy_df['SUM'].astype(float)
    copy_df['PRODUCT'] = copy_df['PRODUCT'].astype(float)
    copy_df['PRODUCT'] = copy_df['SUM']*copy_df['COEFF']
    copy_df['CONTRIBUTION'] = 100*copy_df['PRODUCT']/copy_df['PRODUCT'].sum()
    copy_df['ABS_CONTRIBUTION'] = abs(copy_df['CONTRIBUTION'])
    copy_df = copy_df.sort_values('ABS_CONTRIBUTION', ascending= False).reset_index(drop = True)
    copy_df.drop('ABS_CONTRIBUTION', axis=1, inplace = True)
        
    return copy_df

base_incr_df = calculate_base_discount(results_df1, df2, dis_list, 0.10) # 0.15 represents percentile


# In[98]:


index_rate = df1['IndexBPM'].sum()
base_incr_df['VALUE'] = (base_incr_df['CONTRIBUTION'] * index_rate)/100
base_incr_df['ROI_OR_VOL/DAY'] = base_incr_df['VALUE']*(10**5)/base_incr_df['SUM']
base_incr_df['Elasticity'] = base_incr_df['CONTRIBUTION']/10


# In[99]:


def actual_bpm_abs_discount(data):
    
    temp = data[['DATE', 'Month_Year', 'AMAZON_WTD_DISCOUNT%', 'FLIPKART_WTD_DISCOUNT%',
                 'Amazon_volume', 'Flipkart_volume', 'OVERALL_VOLUME', 'IndexBPM',]]

    temp['amz_contrib'] = temp['Amazon_volume']/(temp['Amazon_volume']+temp['Flipkart_volume'])
    temp['fk_contrib'] = temp['Flipkart_volume']/(temp['Amazon_volume']+temp['Flipkart_volume'])

    temp['amz_bpm'] = temp['amz_contrib']*temp['IndexBPM']
    temp['fk_bpm'] = temp['fk_contrib']*temp['IndexBPM']

    temp['amz_actual_bpm'] = temp['amz_bpm']/(1-temp['AMAZON_WTD_DISCOUNT%'])
    temp['fk_actual_bpm'] = temp['fk_bpm']/(1-temp['FLIPKART_WTD_DISCOUNT%'])

    temp['AMZ_ABS_DISCOUNT'] = temp['amz_actual_bpm']*temp['AMAZON_WTD_DISCOUNT%']
    temp['FK_ABS_DISCOUNT'] = temp['fk_actual_bpm']*temp['FLIPKART_WTD_DISCOUNT%']
    temp['Actual_BPM'] = temp['amz_actual_bpm'] + temp['fk_actual_bpm']

    return temp
    
abs_df = actual_bpm_abs_discount(df)


# In[100]:


def create_discount_variables(data):
    
    temp = data[['DATE', 'AMAZON_WTD_DISCOUNT%', 'FLIPKART_WTD_DISCOUNT%', 'Amazon_volume', 'Flipkart_volume',]]

    temp['Amazon_free_vol'] = (temp['Amazon_volume']/(1-temp['AMAZON_WTD_DISCOUNT%'])) - temp['Amazon_volume']
    temp['Flipkart_free_vol'] = (temp['Flipkart_volume']/(1-temp['FLIPKART_WTD_DISCOUNT%'])) - temp['Flipkart_volume']
    
    data['IndexRate'] = data['Actual_BPM']/data['OVERALL_VOLUME']*10**5
    
    temp1 = pd.merge(temp, data[['DATE', 'IndexRate']], on = 'DATE', how = 'left')

    temp1['Amz_free_value'] = temp1['Amazon_free_vol'] * temp1['IndexRate']
    temp1['Fk_free_value'] = temp1['Flipkart_free_vol'] * temp1['IndexRate']
    
    return temp1

sales_df = create_discount_variables(abs_df)


# In[101]:


dis_list = ['INCREMENTAL_AMAZON_WTD_DISCOUNT%', 'INCREMENTAL_FLIPKART_WTD_DISCOUNT%']

def create_discount_roi(data, dlist, amz_value, fk_value, total_bpm):
    df = data.copy()
    amz_index = df[df['FEATURE'] == dlist[0]].index.values[0]
    fk_index = df[df['FEATURE'] == dlist[1]].index.values[0]
    df.iloc[amz_index, 3] = amz_value # 'SUM' column is in 4th position (4-1 = 3)
    df.iloc[fk_index, 3] = fk_value
       
#     df['VALUE'] = (df['CONTRIBUTION'] * total_bpm)/100
    
    df.iloc[amz_index, 7] = (df.iloc[amz_index, 5] * total_bpm)/100
    df.iloc[amz_index, 8] = (df.iloc[amz_index, 7]*(10**5))/df.iloc[amz_index, 3]
    
    df.iloc[fk_index, 7] = (df.iloc[fk_index, 5] * total_bpm)/100
    df.iloc[fk_index, 8] = (df.iloc[fk_index, 7]*(10**5))/df.iloc[fk_index, 3]
    
#     df['ROI_OR_VOL/DAY'] = df['VALUE']*(10**5)/df['SUM']

    return df

final = create_discount_roi(base_incr_df, dis_list, sales_df['Amz_free_value'].sum(),
                            sales_df['Fk_free_value'].sum(), abs_df['Actual_BPM'].sum())    


# In[102]:


def cost_value(abs_df):

    cost_df = pd.DataFrame([{'Amazon_cost_per_discount' : abs_df['AMZ_ABS_DISCOUNT'].sum()/abs_df['AMAZON_WTD_DISCOUNT%'].sum(),
  'Flipkart_cost_per_discount' : abs_df['FK_ABS_DISCOUNT'].sum()/abs_df['FLIPKART_WTD_DISCOUNT%'].sum()}])
    amz_discount_contrib = results_df[results_df['FEATURE'] == 'AMAZON_WTD_DISCOUNT%']['CONTRIBUTION'].values[0]
    fk_discount_contrib = results_df[results_df['FEATURE'] == 'FLIPKART_WTD_DISCOUNT%']['CONTRIBUTION'].values[0]

    value_df = pd.DataFrame([{'Amazon_value_per_discount' : amz_discount_contrib* abs_df['Actual_BPM'].sum()/100/abs_df['AMAZON_WTD_DISCOUNT%'].sum(),
  'Flipkart_value_per_discount' : fk_discount_contrib* abs_df['Actual_BPM'].sum()/100/abs_df['FLIPKART_WTD_DISCOUNT%'].sum()}])
    cost_val_df = pd.concat([cost_df, value_df], axis=1)
    return(cost_val_df)


# In[103]:


cost_val_df=cost_value(abs_df)


# In[ ]:


final.to_csv("C:/Users/SHUBHAM AGNIHOTRI/Desktop/MCA_GUI/results/roi_elasticity.csv")


# In[ ]:





# In[ ]:




