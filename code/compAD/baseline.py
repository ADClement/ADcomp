#coding:utf-8
import numpy as np
import pandas as pd
import lightgbm as lgb

"""
1.弄三个csv的demo，用来debug用
2.用户特征太大了，没办法直接加在成pandas，要做格式转换，弄成字典之后再弄成dataframe
3.特征拼接，把所有的拼到一起（这里train和test的一起弄，train的-1弄成0，然后test的标签先都打成-1，这样子到时候好区分）
4.缺失值填充成“-1”，字符型的-1
4.应该弄个数据压缩
5.one-hot编码不用弄，直接用lightgbm就行
6.这是一个二分类问题（binary）
"""

"""
广告特征（ad_feature）:
aid             int64
advertiserId    int64
campaignId      int64
creativeId      int64
creativeSize    int64
adCategoryId    int64
productId       int64
productType     int64




"""
data_path="../../preliminary_contest_data/preliminary_contest_data/"

#离散型的特征，数一数有多少个值，每个值有多少个
def discrete_count(feature_col):     #把那个特征列传进来
    print (feature_col.value_counts())



#数据有点大，这个用来读几行数据看看长啥样
#我想知道的，有哪些列，没列是离散型还是连续型特征，离散型有多少个取值
def View_ad_feature_data():     #把data（dataframe型）传进来
    ad_feature = pd.read_csv(data_path + "adFeature.csv")
    print (ad_feature)

    #这个能输出都有那些列
    columns=list(ad_feature.columns)
    print (columns)

    #这个能输出都有哪些列，这些列的数据类型都是啥
    datatype=ad_feature.dtypes
    print (datatype)

    #这个可以看离散的每一列都有哪些值，然后这些值有多少个。（注意，这个ad_feature恰好全是int64的）
    for feature in columns:
        discrete_count(ad_feature[feature])

#用来生成用户特征的csv的，4个G啊4个G（这个太大了，我的电脑内存不够，得在服务器跑）
def user_feature_csv_generate():
    # 搬砖,就是先把user_feature读成字典，然后再弄成dataframe
    userFeature_data = []
    with open(data_path + "userFeature.data", 'r') as f:
        #enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
        for i, line in enumerate(f):
            #strip用来移除前后的空格
            line = line.strip().split('|')
            userFeature_dict = {}
            for each in line:
                each_list = each.split(' ')
                userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
            userFeature_data.append(userFeature_dict)
            if i % 100000 == 0:
                print(i)
        user_feature = pd.DataFrame(userFeature_data)
        user_feature.to_csv(data_path + "userFeature.csv", index=False)




#取几行看看用户特征啥德行
def View_user_feature_data():
    print ("hello")



if __name__=="__main__":
    print ("hello")
    View_ad_feature_data()
    user_feature_csv_generate()

