#coding:utf-8
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from scipy import sparse
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



用户特征（user-feature）：
['LBS', 'age', 'appIdAction', 'appIdInstall', 'carrier', 'consumptionAbility', 'ct', 'education', 'gender', 'house', 'interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'marriageStatus', 'os', 'topic1', 'topic2', 'topic3', 'uid']
LBS                   float64
age                     int64
appIdAction            object
appIdInstall           object
carrier                 int64
consumptionAbility      int64
ct                     object
education               int64
gender                  int64
house                 float64
interest1              object
interest2              object
interest3              object
interest4              object
interest5              object
kw1                    object
kw2                    object
kw3                    object
marriageStatus         object
os                     object
topic1                 object
topic2                 object
topic3                 object
uid                     int64
dtype: object


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

#用来生成用户特征的csv
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
        #print (user_feature.iloc[0:50,:])
        user_feature.to_csv(data_path + "userFeature.csv", index=False)





#(写一堆函数是怕中间出问题就不能弄了，因为这个数据量太大)
#把广告特征，用户特征，标签都合起来，并且生成csv。train和test放一起合方便，要不到时候test也得合，太费劲了......
def train_pred_feature_concat():
    train_data=pd.read_csv(data_path+"train.csv")
    #print (train_data)
    print (1)
    test_data=pd.read_csv(data_path+"test1.csv")
    print (2)
    #print (test_data)
    #train的-1弄成0，然后test的标签先都打成-1
    train_data.loc[train_data["label"]==-1,"label"]=0
    test_data["label"]=-1
    data=pd.concat([train_data,test_data])   #这里的concat是竖着的concat，直接竖着接起来的
    print (3)
    #print (data)
    ad_feature=pd.read_csv(data_path+"adFeature.csv")
    user_feature=pd.read_csv(data_path+"userFeature.csv")
    data=pd.merge(data,ad_feature,on='aid',how='left')     #left:参与合并左侧的dataframe,就是都合到data里
    data=pd.merge(data,user_feature,on='uid',how='left')
    print (4)
    data.to_csv(data_path+"mergedData.csv",index=False)
    print (data.loc[0:50,:])
    return data


#把合起来的数据的缺失值填成-1（字符的-1）
def fillNA_data():
    mergedData=pd.read_csv(data_path+"mergedData.csv")
    mergedData.fillna("-1",inplace=True)    #缺失值填充成字符的-1，虽然我并不是很知道为啥......先有个区分吧
    print(mergedData.loc[0:50, :])
    mergedData.to_csv(data_path+"mergedDataFillna.csv",index=False)



# 这个函数没啥用，只是因为下一个函数漏写了......
# 还有测试功能
def uid_aid_out():
    data = pd.read_csv(data_path + "mergedDataFillna.csv",nrows=300)
    train = data[data.label != -1]
    print (train.size)
    train_y = train.pop('label')  # pop的作用，在train里删除label这一列，并且返回label这一列
    print (train_y.size)
    print (train_y)
    train_y.to_csv(data_path + "train_y_test.csv", index=False)
    train_y=pd.read_csv(data_path+"train_y_test.csv",header=None)    #这个header=None一定要加！否则没有列名的时候，第一行是数据，不加这个header=None就会把第一行变成列名，然后少一行！
    print (train_y)
    print (train_y.size)


def data_final():
    data=pd.read_csv(data_path+"mergedDataFillna.csv")
    #print (data)
    #遇到的一些问题，有人婚姻状况是两个，有很多，这个神奇的操作.不过这个状态少，感觉也没啥关系
    one_hot_feature = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os', 'ct','marriageStatus', 'advertiserId', 'campaignId', 'creativeId','adCategoryId', 'productId', 'productType']
    #这个特征，都是每个里面是598 872 2602 2964 1189 631 5606 5719 5859 5708......这种的，就是比如兴趣爱好标签，他可能有一大堆......这个要处理
    vector_feature = ['appIdAction', 'appIdInstall', 'interest1', 'interest2', 'interest3', 'interest4', 'interest5','kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']

    #这一段是把写出花的特征的类别弄成0，1，2这种（婚姻那个觉得也没事，顶多顶多，五六种而已......多个状态并存感觉问题也不大）
    for feature in one_hot_feature:
        try:
            data[feature]=LabelEncoder().fit_transform(data[feature].apply(int))
        except:
            data[feature]=LabelEncoder().fit_transform(data[feature])

    train=data[data.label!=-1]
    train_y=train.pop('label')     #pop的作用，在train里删除label这一列，并且返回label这一列
    test=data[data.label==-1]
    #print (test)
    res=test[['aid','uid']]
    test=test.drop('label',axis=1)
    # train_x=train[one_hot_feature]
    # test_x=test[one_hot_feature]
    # train_x["creativeSize"]=train["creativeSize"]
    # test_x["creativeSize"] = test["creativeSize"]
    train_x=train[['creativeSize']]     #之前的creativeSize没处理过，这里单独弄出来，为了以后拼接用。然后用[[]]就是dataframe格式，[]是series格式
    test_x=test[['creativeSize']]

    # 虽然lightgbm不用one-hot，但是这个稀疏矩阵的存法，也没办法在lightgbm中找列......所以还是做one-hot吧
    enc = OneHotEncoder()
    for feature in one_hot_feature:
        enc.fit(data[feature].values.reshape(-1,1))
        train_a = enc.transform(train[feature].values.reshape(-1, 1))
        test_a = enc.transform(test[feature].values.reshape(-1, 1))
        train_x=sparse.hstack((train_x,train_a))    #稀疏矩阵的拼接
        test_x=sparse.hstack((test_x,test_a))
    print ("one-hot prepared!")
    #countvectorizer的特征，感觉这个countvectorizer就是加强版的one-hot......
    cv=CountVectorizer()
    for feature in vector_feature:
        #print (data[feature])
        #print (data[feature].dtypes)
        cv.fit(data[feature])            #这里的一个启发，以前我都是train和test各自处理，就会有行列类别对不上的情况，比如这个爱好test里有，但是train里没有，然后就要再处理，非常不方便。这里就是，train和test放在一起弄，就不用怕test里有train里没有了
        train_a=cv.transform(train[feature])
        #print (train_a)    #从这个输出看，这个countvectorizer就弄成稀疏矩阵（sparse matrix）了
        test_a=cv.transform(test[feature])
        train_x=sparse.hstack((train_x,train_a))
        test_x=sparse.hstack((test_x,test_a))      #存成了稀疏矩阵可以直接用lightgbm
    print("cv prepared!")
    #print (train_x)
    #print (train_y)
    #print (test_x)
    #print (train_x)
    sparse.save_npz(data_path+"train_x.npz",train_x)
    sparse.save_npz(data_path + "test_x.npz", test_x)
    train_y.to_csv(data_path+"train_y.csv",index=False)
    res.to_csv(data_path + "res.csv", index=False)
    #tee=sparse.load_npz(data_path+"train_x.npz")
    #print (tee)




#这个用法是skicit-learn和lightgbm的版本。去lightgbm中文手册里找，scikit-learn API
def lightgbm_train_predict(train_x,train_y,test_x,res):
    clf=lgb.LGBMClassifier(
        boosting_type="gbdt",
        num_leaves=31,
        reg_alpha=0.0,       #l1正则
        reg_lambda=1,        #l2正则
        max_depth=-1,
        n_estimators=1500,    #决策树的最大的数量
        objective="binary",
        subsample=0.7,     #随机选数据，防止过拟合
        colsample_bytree=0.7,    #随机选特征，防止过拟合
        subsample_freq=1,
        learning_rate=0.05,
        min_child_weight=50,
        random_state=2018,
        n_jobs=100
    )
    print ("parameter prepared!")
    #原来的没用没用交叉验证集，用train训练，用train做交叉验证......虽然我觉得这样不太好......

    # #划交叉验证集，线下测试来定参数,然后交叉验证集定好参数之后，训练模型要用所有的train训练，这个机器学习基石里说过......
    # train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2)
    # print ("validation prepared!")

    #感觉这个early_stop_round和迭代次数两个参数完全可以只留一个......
    #clf.fit(train_x,train_y,eval_set=[(val_x,val_y)],eval_metric='auc',early_stopping_rounds=100)
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric='auc', early_stopping_rounds=100)
    joblib.dump(clf,data_path+"train_model.m")
    #clf=joblib.load("train_model.m)
    print ("model finished!")

    res["score"]=clf.predict_proba(test_x)[:,1]
    res["score"]=res["score"].apply(lambda x:float("%.6f" % x))   #保留6位小数
    res.to_csv(data_path+"submission.csv",index=False)
    print ("finish!")



def call_lightgbm():
    #这里这个header=None就这个train_y用，就它没有列名（pop操作弄出来的），然后别的都有列名
    #之前fit和train_test_split报错：ValueError: Found input variables with inconsistent numbers of samples:[4，3]，这种报错意思是前后没对上，你给的x是4行，但是y是3行这样子
    train_x=sparse.load_npz(data_path + "train_x.npz")
    train_y=pd.read_csv(data_path+"train_y.csv",header=None)
    test_x = sparse.load_npz(data_path + "test_x.npz")
    res=pd.read_csv(data_path+"res.csv")      #res是aid和uid
    print ("data loaded!")
    #print (train_x.size)
    #print (train_y.size)
    lightgbm_train_predict(train_x, train_y, test_x, res)








#下两个函数都是当小白鼠用的

#查看数据前50行，看样子这种
#还有一些代码里不确定的，可以拿出来几行数据在这里当小白鼠
def view_data():
    train=pd.read_csv(data_path+"train.csv")
    print (train.size)
    # temp=pd.read_csv(data_path+"mergedDataFillna.csv",nrows=300)
    # #print (temp)
    # #print (temp[['appIdAction', 'appIdInstall']])
    #
    # #one-hot的代码可以输出下边的东西理解一下。那个OneHotEncoder的fit里，得放一个最后reshape(-1,1)之后出来的二维array
    # print (temp["marriageStatus"])
    # print (temp["marriageStatus"].values)
    # print (temp["marriageStatus"].values.reshape(-1,1))


#取几行看看用户特征啥德行,每次的结果都记下来，否则太大了不好弄
def View_user_feature_data():
    print ("hello")
    user_feature = pd.read_csv(data_path + "userFeature.csv",nrows=50)
    print (user_feature)

    # 通过这样的代码测试，得到这个apply(int)的作用就是把各种各样的数据类型转成int
    print (user_feature["house"].fillna("-1",inplace=True))
    #print (user_feature["house"].apply(int))

    #然后这里就很神奇了。这个apply(int)的功能，是把所有的不是int都转成int的，因为LabelEncoder只能弄都是int的。
    #之前house缺失值填充的是字符的"-1"，所以这里要用apply(int)转一下。所以觉得那个人的代码缺失值完全没必要填充字符的"-1"啊，直接填充数字的-1不就可以了......
    #但是那个人的代码几个版本这个问题都没有改，为什么？
    user_feature["house"] = LabelEncoder().fit_transform(user_feature["house"].apply(int))
    print (user_feature["house"])

    #这个apply(int)还是有用，因为可能不止int和str混，还有object，想labelEncoder也得转成int
    #问题又来了，能不能不用labelEncoder(不过这个labelEncoder方便，labelEncoder的作用，不论这个类别的数字编码多大多炫酷，统统打回原型从0开始数......本质上就是个类别，就让他们变成只有0，1，2，3......这样从头开始数的基础的类别)
    user_feature["interest5"].fillna(-1, inplace=True)
    print(user_feature["interest5"])
    user_feature["interest5"] = LabelEncoder().fit_transform(user_feature["interest5"])

    # 这个能输出都有那些列
    columns = list(user_feature.columns)
    print(columns)

    # 这个能输出都有哪些列，这些列的数据类型都是啥
    datatype = user_feature.dtypes
    print(datatype)

    # 这个可以看离散的每一列都有哪些值，然后这些值有多少个。
    # for feature in columns:
    #     discrete_count(ad_feature[feature])


if __name__=="__main__":
    print ("hello")
    disperse_feature=[""]
    #View_ad_feature_data()
    #user_feature_csv_generate()
    #View_user_feature_data()
    #train_pred_feature_concat()
    #fillNA_data()
    #view_data()
    #data_final()
    #uid_aid_out()
    call_lightgbm()
