# 使用联邦学习 逐步退出
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Gaussian_class import Gaussian
from datetime import timedelta
import xgboost as xgb
import sys
from scipy.stats import norm 

#%% 重新定义param_dict的参数。同时用于Gaussian
Gaussian.dict_param(mu_transform0=None,sigma_transform0="Softplus",mu_extra_rate0=0.1,sigma_extra_rate0=0.1)

#%% 导入进行分析的数据
def load_data(path="final_dataset_multidistrict_historical.csv",exclude_district=49,):
    df=pd.read_csv(path)
    df=df[df['District_index']!=exclude_district]  #去除地区49的数据，该地区的数据被用于完全分立的测试集
    df['Date-Time']=pd.to_datetime(df['Date-Time'])
    df=df.sort_values(by='Date-Time').reset_index(drop=True)
    df.pop('Coref-Residential')
    df.pop('Coref_Commercial')
    df.pop('Coref-Transportation')
    df.pop('Coref_Industrial')
    df.pop('past_load_12')
    #df.pop('past_load_36')
    df.pop('past_load_24')
    df.pop('past_load_48')
    
    #大致确定train, eval和test的截断比例
    eval_ratio_start=0.3
    eval_ratio_rough=0.35
    train_ratio_rough=0.65
    test_num=168  #168个小时，即1周的数据
    
    def round_date(date):
        if date.hour!=0:
            return date-timedelta(hours=date.hour)+timedelta(days=1)
        return date
    time_zero_cutoff=df.iloc[int(eval_ratio_start*len(df))]['Date-Time']
    time_first_cutoff=df.iloc[int(eval_ratio_rough*len(df))]['Date-Time']
    time_first_cutoff=round_date(time_first_cutoff)
    time_second_cutoff=df.iloc[int(train_ratio_rough*len(df))]['Date-Time']
    time_second_cutoff=round_date(time_second_cutoff)
    time_third_cutoff=time_second_cutoff+timedelta(hours=test_num)
    print("Start Time:",time_zero_cutoff)
    print("Valid Set Cutoff:",time_first_cutoff)
    print("Training Set Cutoff",time_second_cutoff)
    print("Test Set Cutoff",time_third_cutoff)
    
    district_index=list(df['District_index'].unique())
    
    dtrain_list=[]  # 训练集采用简单列表的形式，[train_data0]
    dtrain_label_list=[] #训练集标签采用简单列表的形式，[train_label0]
    deval_list=[]  # 验证集采用元组组成的列表的形式，[(deval0，str(地区序号))]
    dtest_list=[]  # 测试集采用简单列表的形式，[dtest0]
    train_data=df[(df['Date-Time']>=time_first_cutoff) & (df['Date-Time']<time_second_cutoff)].copy().reset_index(drop=True)
    train_data.pop('Date-Time')    
    valid_data=df[(df['Date-Time']>=time_zero_cutoff) & (df['Date-Time']<time_first_cutoff)].copy().reset_index(drop=True)
    valid_data.pop('Date-Time')
    test_data=df[(df['Date-Time']>=time_second_cutoff) & (df['Date-Time']<time_third_cutoff)].copy().reset_index(drop=True)
    test_data.pop('Date-Time')
    valid_data_copy=valid_data.copy()
    valid_data_copy.pop('District_index')
    valid_label_copy=valid_data_copy.pop('Load')
    deval_total=(xgb.DMatrix(valid_data_copy,label=valid_label_copy),"Total")
    for IDX in district_index:
        train_data_copy=train_data[train_data['District_index']==IDX].copy()
        train_data_copy.pop('District_index')
        train_label_copy=train_data_copy.pop('Load')
        dtrain_list.append(train_data_copy)
        dtrain_label_list.append(train_label_copy)
        
        valid_data_copy=valid_data[valid_data['District_index']==IDX].copy()
        valid_data_copy.pop('District_index')
        valid_label_copy=valid_data_copy.pop('Load')
        deval_list.append((xgb.DMatrix(valid_data_copy,label=valid_label_copy),str(IDX)))
        
        test_data_copy=test_data[test_data['District_index']==IDX].copy()
        test_data_copy.pop('District_index')
        test_label_copy=test_data_copy.pop('Load')
        dtest_list.append(xgb.DMatrix(test_data_copy,label=test_label_copy))  

    return district_index,dtrain_list,dtrain_label_list,deval_list,dtest_list,deval_total

#%% 只要残留的地区数量大于1，就可以进行federated learning
district_index,dtrain_list,dtrain_label_list,deval_list,dtest_list,deval_total=load_data()
district_index_raw=[i for i in district_index]
drop_index_list=[]

district_of_interest=38
dtest_38=dtest_list[district_index.index(district_of_interest)]
dtrain_38=dtrain_list[district_index.index(district_of_interest)]
mean_38=24.781076105694652
sigma_38=11.669209709977569
eval_dict_transfer_temp=dict()
eval_transfer_dict=dict()
df_prediction_dict=dict()
df_prediction_transfer_dict=dict()
model=Gaussian()
MAX_DEPTH=3  ##!!!MAX_DEPTH=3或4都可以或多或少实现效果，请注意改动

train_data=pd.concat(dtrain_list)
train_label=pd.concat(dtrain_label_list)
dtrain=xgb.DMatrix(train_data,label=train_label)
eval_dict={}
model.train(dtrain,max_depth=MAX_DEPTH,num_boost_round=500,evals=[deval_total],evals_result=eval_dict,eta=0.2,verbose_eval=0)
res_temp=model.predict(dtest_38,normalized=False) #设定normalized为False，这样可以绘图
res_temp['mu']=res_temp['mu']*sigma_38+mean_38
res_temp['sigma']*=sigma_38
res_temp['label']=res_temp['label']*sigma_38+mean_38
for district in range(len(district_index)):
    eval_dict_transfer_temp=dict()
    IDX=district_index[district]
    if dtest_list[district].num_row():
        df_prediction_dict[IDX]=model.predict(dtest_list[district],normalized=True)
    # 寻找到底哪些参与方过拟合了，并且从district_index中删除这些参与方（进入transfer阶段）
    model.train_transfer(xgb.DMatrix(dtrain_list[district],label=dtrain_label_list[district]),IDX,max_depth=MAX_DEPTH,num_boost_round=500,evals=[deval_list[district]],
                                    evals_result=eval_dict_transfer_temp,eta=0.2,verbose_eval=0)
    if IDX==district_of_interest:
        res_temp_transfer=model.predict_transfer(dtest_38,IDX,normalized=False)
        res_temp_transfer['mu']=res_temp_transfer['mu']*sigma_38+mean_38
        res_temp_transfer['sigma']*=sigma_38
        res_temp_transfer['label']=res_temp_transfer['label']*sigma_38+mean_38
    if dtest_list[district].num_row():
        df_prediction_transfer_dict[IDX]=model.predict_transfer(dtest_list[district],IDX,normalized=True)

    eval_transfer_dict[IDX]=eval_dict_transfer_temp[str(IDX)]['NLL']

#%% 对测试集进行分析
final_list_pred=[]
for IDX in df_prediction_dict:
    df_pred=df_prediction_dict[IDX]
    final_list_pred.append(-np.nansum(norm.logpdf(x=df_pred['label'],loc=df_pred['mu'],scale=df_pred['sigma']))/df_pred.shape[0])

final_list_pred_transfer=[]
for IDX in df_prediction_transfer_dict:
    df_pred_transfer=df_prediction_transfer_dict[IDX]
    final_list_pred_transfer.append(-np.nansum(norm.logpdf(x=df_pred_transfer['label'],loc=df_pred_transfer['mu'],scale=df_pred_transfer['sigma']))/df_pred_transfer.shape[0])

#%% 对第11组数据进行分析 
def load_district_11(path="final_dataset_multidistrict_historical.csv",exclude_district=49):
    df=pd.read_csv(path)
    df=df[df['District_index']==exclude_district] #这里只保留地区49的数据
    df['Date-Time']=pd.to_datetime(df['Date-Time'])
    df=df.sort_values(by='Date-Time').reset_index(drop=True)
    df.pop('Coref-Residential')
    df.pop('Coref_Commercial')
    df.pop('Coref-Transportation')
    df.pop('Coref_Industrial')
    df.pop('past_load_12')
    #df.pop('past_load_36')
    df.pop('past_load_24')
    df.pop('past_load_48')
    df.pop('District_index')

    time_zero_cutoff=pd.to_datetime('2019-07-22 22:00:00')
    time_first_cutoff=pd.to_datetime('2019-08-10 00:00:00')
    time_second_cutoff=pd.to_datetime('2020-02-11 00:00:00')
    time_third_cutoff=pd.to_datetime('2020-02-18 00:00:00')
       
    train_data=df[(df['Date-Time']>=time_first_cutoff) & (df['Date-Time']<time_second_cutoff)].copy().reset_index(drop=True)
    train_data.pop('Date-Time')    
    valid_data=df[(df['Date-Time']>=time_zero_cutoff) & (df['Date-Time']<time_first_cutoff)].copy().reset_index(drop=True)
    valid_data.pop('Date-Time')
    test_data=df[(df['Date-Time']>=time_second_cutoff) & (df['Date-Time']<time_third_cutoff)].copy().reset_index(drop=True)
    test_data.pop('Date-Time')
    
    train_label=train_data.pop('Load')
    valid_label=valid_data.pop('Load')
    test_label=test_data.pop('Load')
    return train_data,train_label,valid_data,valid_label,test_data,test_label


train_data_11,train_label_11,valid_data_11,valid_label_11,test_data_11,test_label_11=load_district_11(exclude_district=49)

district_of_interest=38
mean_38=24.781076105694652
sigma_38=11.669209709977569
dtest_38=dtest_list[district_index.index(district_of_interest)]
dtrain_38=dtrain_list[district_index.index(district_of_interest)]

dtrain_11=xgb.DMatrix(train_data_11,label=train_label_11)
deval_11=xgb.DMatrix(valid_data_11,label=valid_label_11)
dtest_11=xgb.DMatrix(test_data_11,label=test_label_11)

df_bst_test=model.predict(dtest_11,ntree_limit=0,normalized=True)
mu_bst_test=np.array(df_bst_test['mu'])[:,None]
sigma_bst_test=Gaussian.Softplus_inv(np.array(df_bst_test['sigma']))[:,None]

df_bst_valid=model.predict(deval_11,ntree_limit=0,normalized=True)
mu_bst_valid=np.array(df_bst_valid['mu'])[:,None]
sigma_bst_valid=Gaussian.Softplus_inv(np.array(df_bst_valid['sigma']))[:,None]

df_bst_train=model.predict(dtrain_11,ntree_limit=0,normalized=True)
mu_bst_train=np.array(df_bst_train['mu'])[:,None]
sigma_bst_train=Gaussian.Softplus_inv(np.array(df_bst_train['sigma']))[:,None]

final_res_pre=-np.nansum(norm.logpdf(x=df_bst_test['label'],loc=df_bst_test['mu'],scale=df_bst_test['sigma']))/df_bst_test.shape[0]


#%% 重组
train_label_11/=model.initial_sigma
valid_label_11/=model.initial_sigma
test_label_11/=model.initial_sigma
dtrain_11=xgb.DMatrix(train_data_11,label=train_label_11)
deval_11=xgb.DMatrix(valid_data_11,label=valid_label_11)
dtest_11=xgb.DMatrix(test_data_11,label=test_label_11)
base_margin_train=np.concatenate([mu_bst_train,sigma_bst_train],axis=1).flatten()
dtrain_11.set_base_margin(base_margin_train)
base_margin_valid=np.concatenate([mu_bst_valid,sigma_bst_valid],axis=1).flatten()
deval_11.set_base_margin(base_margin_valid)
base_margin_test=np.concatenate([mu_bst_test,sigma_bst_test],axis=1).flatten()
dtest_11.set_base_margin(base_margin_test)

model2=Gaussian()
MAX_DEPTH=3  # !!!MAX_DEPTH=3或4都可以或多或少实现效果，请注意改动
eval_dict_11={}
model2.train(dtrain_11,max_depth=MAX_DEPTH,num_boost_round=500,evals=[(deval_11,"49_test")],evals_result=eval_dict_11,eta=0.2,verbose_eval=0,reset_base_margin=False)

df_bst_test=model2.predict(dtest_11,ntree_limit=0,normalized=True,reset_base_margin=False)
final_res=-np.nansum(norm.logpdf(x=df_bst_test['label'],loc=df_bst_test['mu'],scale=df_bst_test['sigma']))/df_bst_test.shape[0]

#%% 将生成的模型进行导出
directory='Model-JSON-Simultaneous/'
for IDX in model.xgb_model_individual:
    model.xgb_model_individual[IDX].dump_model(directory+"Model-District-"+str(IDX)+".json",with_stats=True,dump_format='json')

model.xgb_model.dump_model(directory+"Model-Total.json",with_stats=True,dump_format='json')


#%% 将loss进行导出
import pickle
directory='Training-Pickle-Simultaneous/'
# 导出eval_dict_list和eval_transfer_dict
with open(directory+'eval_dict.pickle', 'wb') as f:
    pickle.dump(eval_dict, f)
with open(directory+'eval_transfer_dict.pickle', 'wb') as f:
    pickle.dump(eval_transfer_dict, f)

district_index,dtrain_list,dtrain_label_list,deval_list,dtest_list,deval_total=load_data()
df_num_samples=pd.DataFrame(index=district_index,columns=["Train","Valid","Test"])
for i in range(len(district_index)):
    df_num_samples.loc[district_index[i],"Train"]=dtrain_list[i].shape[0]
    df_num_samples.loc[district_index[i],"Valid"]=deval_list[i][0].num_row()
    df_num_samples.loc[district_index[i],"Test"]=dtest_list[i].num_row()
df_num_samples.to_csv(directory+"num_samples.csv")

#%%绘图（第2部分）
directory='Training-Pickle-Simultaneous/'
with open(directory+'res_temp_simultaneous.pickle', 'wb') as f:
    pickle.dump(res_temp, f)
with open(directory+'res_temp_transfer_simultaneous.pickle', 'wb') as f:
    pickle.dump(res_temp_transfer, f)
eval_dict={}
fig,ax=plt.subplots(figsize=(10,5))

plt.plot(np.arange(len(res_temp)),res_temp['mu'],'b--',label='Mean (Global Model)')
plt.plot(res_temp_transfer['mu'],color='blue',label='Mean (Final Model)')
plt.plot(res_temp_transfer['label'],color='red',label='True Load Consumption')
plt.fill_between(np.arange(len(res_temp)), res_temp['mu']-res_temp['sigma'], res_temp['mu']+res_temp['sigma'], facecolor='pink', alpha=0.5, edgecolor='black',label='Standard Deviation (Global Model)')
plt.fill_between(np.arange(len(res_temp_transfer)), res_temp_transfer['mu']-res_temp_transfer['sigma'], res_temp_transfer['mu']+res_temp_transfer['sigma'], facecolor='orange', alpha=0.5, edgecolor='black',label='Standard Deviation (Final Model)')

ax.set_xticks([i*24 for i in range(8)])
ax.set_xticklabels(["8/1"+str(i) for i in range(8)])
ax.set_xlabel('Date')
ax.set_ylabel('Load Consumption/MW')
ax.legend(ncol=2,bbox_to_anchor=(1.05,-0.2),fontsize=20)