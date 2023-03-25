#####################################################
# CLTV Prediction with BG-NBD and Gamma-Gamma Model
####################################################

###################
# Business Problem
###################

# FLO wants to set a roadmap for sales and marketing activities.
# In order for the company to make a medium-long-term plan, it is necessary to estimate the potential value that existing customers will provide to the company in the future.

#######################
# The story of dataset
#######################

# The dataset is based on the past shopping behavior of customers who made their last purchases from OmniChannel (both online and offline) in 2020 - 2021.
# consists of the information obtained.

# master_id: Unique customer number
# order_channel : Which channel of the shopping platform is used (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : The channel where the last purchase was made
# first_order_date : The date of the customer's first purchase
# last_order_date : The date of the last purchase made by the customer
# last_order_date_online : The date of the last purchase made by the customer on the online platform
# last_order_date_offline : The date of the last purchase made by the customer on the offline platform
# order_num_total_ever_online : The total number of purchases made by the customer on the online platform
# order_num_total_ever_offline : Total number of purchases made by the customer offline
# customer_value_total_ever_offline : The total price paid by the customer for offline purchases
# customer_value_total_ever_online : The total price paid by the customer for their online shopping
# interested_in_categories_12 : List of categories the customer has purchased from in the last 12 months



# TASK 1: Data Understanding
import datetime as dt
import pandas as pd
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

# 1. Read the OmniChannel.csv file
df_ = pd.read_csv("datasets/flo_data_20k.csv")
df = df_.copy()

def check_df(dataframe, head=5):
    print("INFO".center(70,'='))
    print(dataframe.info())

    print("SHAPE".center(70,'='))
    print('Rows: {}'.format(dataframe.shape[0]))
    print('Columns: {}'.format(dataframe.shape[1]))

    print("TYPES".center(70,'='))
    print(dataframe.dtypes)

    print("HEAD".center(70, '='))
    print(dataframe.head(head))

    print("TAIL".center(70,'='))
    print(dataframe.tail(head))

    print("NULL".center(70,'='))
    print(dataframe.isnull().sum())

    print("QUANTILES".center(70,'='))
    print(dataframe.describe().T)

check_df(df)


#2. Define the outlier_thresholds and replace_with_thresholds functions needed to suppress outliers.
# Note: When calculating cltv, frequency values must be integers. Therefore, round the lower and upper limits with round().

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range

    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    print(f"{variable} low_limit: {low_limit}")
    print(f"{variable} up_limit: {up_limit}")
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit)


list = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline", "customer_value_total_ever_online"]
for i in list:
    print(i,"low-up : " ,outlier_thresholds(df,i))

df[list].describe().T

# 3. Suppress if the variables "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online" have outliers.

def replace_threshold(dataframe):
    for col in list:
        replace_with_thresholds(dataframe,col)

replace_threshold(df)
df.describe().T


# 4. Omnichannel means that customers shop from both online and offline platforms.
# Create new variables for each customer's total order number and total price.

df["total_order"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_price"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]


# 5. Examine the data types

df.dtypes

df.loc[:,df.columns.str.contains("date")] = df.loc[:,df.columns.str.contains("date")].apply(pd.to_datetime)


# TASK 2: Creating the CLTV Data Structure

# 1. Take 2 days after the date of the last purchase in the data set as the date of analysis.
df["last_order_date"].max()

today_date = dt.datetime(2021, 6, 1)


# 2. Create a ne dataframe with customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg values
cltv_df = pd.DataFrame()
cltv_df["customer_id"] = df["master_id"]
cltv_df["recency_cltv_weekly"] = ((df["last_order_date"]-df["first_order_date"]).astype('timedelta64[D]')) / 7
cltv_df["T_weekly"] = ((today_date - df["first_order_date"]).astype('timedelta64[D]')) / 7
cltv_df["frequency"] = df["total_order"]
cltv_df["monetary_cltv_avg"] = df["total_price"] / df["total_order"]
cltv_df = cltv_df[(cltv_df["frequency"] > 1)]

#(df["last_order_date"] - df["first_order_date"]).dt.days / 7

cltv_df

cltv_df.describe().T


# TASK 3: BG/NBD, Establishing Gamma-Gamma Models, calculating 6-month CLTV

# 1. Build BG/NBD model
bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df["frequency"], cltv_df["recency_cltv_weekly"], cltv_df["T_weekly"])

# 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.
cltv_df["exp_sales_3_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4 * 3,
                                                        cltv_df["frequency"],
                                                        cltv_df["recency_cltv_weekly"],
                                                        cltv_df["T_weekly"])

cltv_df.sort_values(by="exp_sales_3_month", ascending=False)

# Estimate expected purchases from customers in 6 months and add exp_sales_6_month to cltv dataframe.
cltv_df["exp_sales_6_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4 * 6,
                                                        cltv_df["frequency"],
                                                        cltv_df["recency_cltv_weekly"],
                                                        cltv_df["T_weekly"])

cltv_df.sort_values(by="exp_sales_6_month", ascending=False)


# Let's review the 10 people who will make the most purchases in the 3rd and 6th months.
cltv_df.sort_values(by=["exp_sales_3_month","exp_sales_6_month"], ascending=False).head(10)


# 2. Fit the Gamma-Gamma model. Estimate the average value of the customers and add it to the cltv dataframe as exp_average_value.
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df["frequency"],cltv_df["monetary_cltv_avg"])
cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],cltv_df["monetary_cltv_avg"])

# 3. Calculate 6 months CLTV and add it to the dataframe with the name cltv.
cltv_df["cltv"] = ggf.customer_lifetime_value(bgf,
                                              cltv_df["frequency"],
                                              cltv_df["recency_cltv_weekly"],
                                              cltv_df["T_weekly"],
                                              cltv_df["monetary_cltv_avg"],
                                              time=6,
                                              freq="W",
                                              discount_rate=0.01)
cltv_df.head()


# Observe the 20 people with the highest CLTV value.
cltv_df.sort_values(by="cltv", ascending=False).head(20)


# TASK 4: Creating Segments by CLTV

# 1. Divide all your customers into 4 groups (segments) according to 6-month CLTV and add the group names to the dataset.
# Assign with the name cltv_segment.
cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4 , ["D","C","B","A"])

# 2. Examine the recency, frequency and monetary averages of the segments.
cltv_df.groupby("cltv_segment").agg({"recency_cltv_weekly":"mean",
                                     "frequency": "mean",
                                     "monetary_cltv_avg":"mean"})

