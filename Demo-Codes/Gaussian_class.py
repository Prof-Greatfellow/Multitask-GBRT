import numpy as np
import pandas as pd
from FederBoost import fdb
from scipy.stats import norm 
import sys
import matplotlib.pyplot as plt
from datetime import timedelta

class Gaussian:
    
    mu_transform=None
    sigma_transform="Softplus"
    mu_extra_rate=1
    sigma_extra_rate=0.1
    
    @staticmethod
    def dict_param(mu_transform0,sigma_transform0,mu_extra_rate0,sigma_extra_rate0):
        Gaussian.mu_transform=mu_transform0
        Gaussian.sigma_transform=sigma_transform0
        Gaussian.mu_extra_rate=mu_extra_rate0
        Gaussian.sigma_extra_rate=sigma_extra_rate0
    
    @staticmethod
    def Softplus(x):
        return np.log(1+np.exp(x))
    
    @staticmethod
    def Softplus_inv(x):
        return np.log(np.exp(x)-1)
    
    @staticmethod
    def gradient_mu(y,mu,sigma,types=None):
        return Gaussian.stabilize_derivative((mu-y)/(sigma**2),types)*Gaussian.mu_extra_rate
    
    @staticmethod
    def hess_mu(y,mu,sigma,types=None):
        return Gaussian.stabilize_derivative(1/(sigma**2),types)
    
    @staticmethod
    def gradient_sigma(y,mu,sigma,types=None):
        if Gaussian.sigma_transform==None:
            return Gaussian.stabilize_derivative((sigma**2-(y-mu)**2)/(sigma**3),types)*Gaussian.sigma_extra_rate      
        if Gaussian.sigma_transform=="Softplus":
            return Gaussian.stabilize_derivative((sigma**2-(y-mu)**2)/(sigma**3)*(np.exp(sigma)-1)/(np.exp(sigma)),types)*Gaussian.sigma_extra_rate
    
    @staticmethod
    def hess_sigma(y,mu,sigma,types=None):
        if Gaussian.sigma_transform==None:
            return Gaussian.stabilize_derivative((3*((y-mu)**2)-sigma**2)/(sigma**4),types)  
        if Gaussian.sigma_transform=="Softplus":
            return Gaussian.stabilize_derivative((np.exp(sigma)-1)/(np.exp(2*sigma)*(sigma**4))*(-(sigma**2)*(np.exp(sigma)-1)+2*(np.exp(sigma)-1)*((y-mu)**2)+sigma**3-sigma*(y-mu)**2),types)
    
    @staticmethod
    def stabilize_derivative(input_der,types=None):
        if types==None:
            return input_der
        if types=="L2":
            div=np.sqrt(np.nanmean(input_der**2))
            div=np.where(div<1e-4,1e-4,div)
            div=np.where(div>1e4,1e4,div)
            return input_der/div
        if types=="MAD":
            div=np.nanmedian(np.abs(input_der-np.nanmedian(input_der)))
            div=np.where(div<1e-4,1e-4,div)
            return input_der/div
    
    
    @staticmethod
    def Dist_Obj(predt,data):
        target=data.get_label()
        preds_mu=predt[:,0]
        if Gaussian.sigma_transform==None:
            preds_sigma=predt[:,1]
        elif Gaussian.sigma_transform=="Softplus":
            preds_sigma=Gaussian.Softplus(predt[:,1])
        preds_sigma=np.clip(preds_sigma,1e-7,None)
        grad=np.zeros(shape=(len(target),2))
        hess=np.zeros(shape=(len(target),2))
        types=None
        grad[:,0]=Gaussian.gradient_mu(y=target,mu=preds_mu,sigma=preds_sigma,types=types)
        grad[:,1]=Gaussian.gradient_sigma(y=target,mu=preds_mu,sigma=preds_sigma,types=types)
        hess[:,0]=Gaussian.hess_mu(y=target,mu=preds_mu,sigma=preds_sigma,types=types)
        hess[:,1]=Gaussian.hess_sigma(y=target,mu=preds_mu,sigma=preds_sigma,types=types)
        
        grad=grad.flatten()
        hess=hess.flatten()
        return grad,hess
        
    @staticmethod
    def Dist_Metric(predt,data):
        target=data.get_label()
        preds_mu=predt[:,0]
        if Gaussian.sigma_transform==None:
            preds_sigma=predt[:,1]
        elif Gaussian.sigma_transform=="Softplus":
            preds_sigma=Gaussian.Softplus(predt[:,1])
        
        nll=-np.nansum(norm.logpdf(x=target,loc=preds_mu,scale=preds_sigma))
        return "NLL",nll
    
    def __init__(self):
        self.fdb_model=None
        self.fdb_model_individual=dict()
    
    def __trainBASE__(self,dtrain,district_idx,max_depth,eta,num_boost_round,evals,maximize,early_stopping_rounds,evals_result,verbose_eval,reset_base_margin=True):
        # 对train和train_transfer的统一代码实现
        params={"objective":None,
                "base_score":0,
                "num_class":2,  #使用2 classes来模拟同时训练mu和sigma两棵决策树的方法
                "disable_default_eval_metric":True,
                "max_depth":max_depth,
                "eta":eta}
        
        if reset_base_margin:
            if evals is None:
                evals_normalized=None
            else:
                evals_normalized=[]
                for i in evals:
                    a=i[0]
                    temp_label=a.get_label()/self.initial_sigma
                    a.set_label(temp_label)
                    a.set_base_margin(((np.ones(shape=(a.num_row(),1)))*np.array([self.mu_start_values,self.sigma_RAW_start_values])).flatten())
                    evals_normalized.append((a,i[1]))
        else:
            evals_normalized=evals

        callbacks=[fdb.callback.EarlyStopping(rounds=1,data_name=i[1]) for i in evals_normalized]

        temp_model=fdb.train(params,dtrain,num_boost_round=num_boost_round,evals=evals_normalized,
                                    obj=Gaussian.Dist_Obj,feval=Gaussian.Dist_Metric,fdb_model=self.fdb_model,
                                    callbacks=callbacks,verbose_eval=verbose_eval,evals_result=evals_result,
                                    maximize=maximize,early_stopping_rounds=early_stopping_rounds)
        
        if district_idx==-1:
            self.fdb_model=temp_model
        else:
            self.fdb_model_individual[district_idx]=temp_model
        
        #最后再将initial_sigma乘回去(如果没有reset_base_margin的话)
        if reset_base_margin:
            if evals_normalized is not None:
                for i in evals_normalized:
                    a=i[0]
                    temp_label=a.get_label()*self.initial_sigma
                    a.set_label(temp_label)
    
    def train(self,dtrain,max_depth=5,eta=0.1,num_boost_round=100,evals=None,maximize=False,early_stopping_rounds=None,
              evals_result=None,verbose_eval=True,reset_base_margin=True):
        def compute_mu_start_values(label):
            assert Gaussian.mu_transform==None,"Cannot address the mu transformation right now!"
            return np.mean(label)
        
        def compute_sigma_start_values(label):
            sigma_transform=Gaussian.sigma_transform
            if sigma_transform==None:
                return np.std(label)
            if sigma_transform=="Softplus":
                return Gaussian.Softplus_inv(np.std(label))
            raise AssertionError("Cannot address the sigma transformation right now!")
        
        if reset_base_margin:
            label=dtrain.get_label()
            self.initial_sigma=np.std(label)
            
            #将label进行normalize，使其方差为1
            label/=self.initial_sigma
            dtrain.set_label(label)
            self.mu_start_values=compute_mu_start_values(label)
            self.sigma_RAW_start_values=compute_sigma_start_values(label)
            base_margin=(np.ones(shape=(dtrain.num_row(),1)))*np.array([self.mu_start_values,self.sigma_RAW_start_values])
            dtrain.set_base_margin(base_margin.flatten())
            
            self.__trainBASE__(dtrain,-1,max_depth,eta,num_boost_round,evals,maximize,early_stopping_rounds,evals_result,verbose_eval)
            dtrain.set_label(label*self.initial_sigma)  # 将label的值乘回去
        
        #这里使用reset_base_margin=False，因为考虑到地区11的重新训练需要手动添加base_margin
        # 注意这里直接不使用base_margin了！
        else:
            self.__trainBASE__(dtrain,-1,max_depth,eta,num_boost_round,evals,maximize,early_stopping_rounds,evals_result,verbose_eval,reset_base_margin=False)

    def train_transfer(self,dtrain,district_idx,max_depth=5,eta=0.1,num_boost_round=100,evals=(),maximize=False,
                       early_stopping_rounds=None,evals_result=None,verbose_eval=True):
        """
        从Gaussian_transfer复制粘贴过来，如果效果不好可以删掉
        """
        label=dtrain.get_label()
        label/=self.initial_sigma
        base_margin=(np.ones(shape=(dtrain.num_row(),1)))*np.array([self.mu_start_values,self.sigma_RAW_start_values])
        dtrain.set_base_margin(base_margin.flatten())
        
        self.__trainBASE__(dtrain,district_idx,max_depth,eta,num_boost_round,evals,maximize,early_stopping_rounds,evals_result,verbose_eval)
        dtrain.set_label(label*self.initial_sigma)  # 将label的值乘回去
    
    def __predictBASE__(self,dtest,district_idx,ntree_limit,normalized,reset_base_margin=True):
        if reset_base_margin:
            base_margin=(np.ones(shape=(dtest.num_row(),1)))*np.array([self.mu_start_values,self.sigma_RAW_start_values])
            dtest.set_base_margin(base_margin.flatten())
            dtest.set_label(dtest.get_label()/self.initial_sigma)
        #为啥要用output_margin=True？我的理解是因为我们一直在当做multi-class classification在处理
        if district_idx==-1:
            predt=self.fdb_model.predict(dtest,output_margin=True,ntree_limit=ntree_limit)
        else:
            predt=self.fdb_model_individual[district_idx].predict(dtest,output_margin=True,ntree_limit=ntree_limit)

        
        #得到最终输出的mu,sigma参数
        dist_params_predts=[]
        assert Gaussian.mu_transform==None
        dist_params_predts.append(predt[:,0])
        if Gaussian.sigma_transform==None:
            dist_params_predts.append(predt[:,1])
        elif Gaussian.sigma_transform=='Softplus':
            dist_params_predts.append(Gaussian.Softplus(predt[:,1]))
        dist_params_df=pd.DataFrame(dist_params_predts).T
        dist_params_df.columns=['mu','sigma']
        dist_params_df['label']=dtest.get_label()
        
        if reset_base_margin:
            dtest.set_label(dtest.get_label()*self.initial_sigma)  #这一步非常重要！将dtest中的label进行还原
        
        if reset_base_margin:
            if not normalized:
                dist_params_df['mu']*=self.initial_sigma
                dist_params_df['sigma']*=self.initial_sigma
                dist_params_df['label']*=self.initial_sigma
    
        return dist_params_df       
    
    def predict(self,dtest,ntree_limit=0,normalized=True,reset_base_margin=True):
        return self.__predictBASE__(dtest,-1,ntree_limit,normalized,reset_base_margin)

    def predict_transfer(self,dtest,district_idx,ntree_limit=0,normalized=True):
        return self.__predictBASE__(dtest,district_idx,ntree_limit,normalized)