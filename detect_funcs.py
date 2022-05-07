import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from variables import *
from sklearn.preprocessing import MinMaxScaler
from grid_funcs import *
import networkx as nx
import random
import os
from sklearn import *
import multiprocessing
import itertools
from scipy.stats import ks_2samp
from prophet import Prophet
from datetime import timedelta,datetime
import collections
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from glob import glob
import time

class processData:
    def build_all_raw_grid_data(all_grids):
        ''' This method must compose a table with every point per which we have grid id, node type and time point '''    
        all_raw_grid_data=pd.DataFrame(columns=synth_columns_grid_data)
        if not all_grids:
            all_grids=os.listdir(full_current_grids_data_dir)
        for grid_id in all_grids:
            aux_grid_data=pd.read_csv(f'{full_current_grids_data_dir}/{grid_id}/synth_energy_data.csv')
            aux_all_raw_grid_data=pd.DataFrame()
            for nt in np.unique(aux_grid_data['node_id']):
                aux_data_per_type=aux_grid_data[aux_grid_data['node_id']==nt].copy()
                aux_data_per_type['energy']=MinMaxScaler().fit_transform(np.array(aux_data_per_type['energy']).reshape(-1,1))[:,0]
                aux_all_raw_grid_data=aux_all_raw_grid_data.append(aux_data_per_type)
            aux_grid_data['node_type']=[n.split("_")[0] for n in aux_grid_data['node_id']]
            aux_all_raw_grid_data.columns=synth_columns_grid_data
            all_raw_grid_data=all_raw_grid_data.append(aux_all_raw_grid_data)        
        
        return all_raw_grid_data

    def build_all_raw_grid_data_per_grid(input_data):
        ''' This method is the simple reader which is responsible of read the data of a single grid. Then build_all_raw_grid_data_multi join them.'''
        grid_id=input_data
        aux_grid_data=pd.read_csv(f'{full_current_grids_data_dir}/{grid_id}/synth_energy_data.csv')
        for nt in np.unique(aux_grid_data['node_id']):
            aux_node_registers=aux_grid_data['node_id']==nt
            aux_data_per_type=aux_grid_data[aux_node_registers].copy()
            aux_grid_data['energy'].loc[aux_node_registers]=MinMaxScaler().fit_transform(np.array(aux_data_per_type['energy']).reshape(-1,1))[:,0]
        
        aux_grid_data['node_type']=[n.split("_")[0] for n in aux_grid_data['node_id']]
        return np.array(aux_grid_data)
        

    def build_all_raw_grid_data_multi(all_grids):
        ''' This method is the reader of all the synthetic data. We splitted the read of all the grids data into a more simple method. '''
        
        if len(all_grids)>5: # As we cannot use nested multiprocessing tasks we have to set when we will use a multiprocess and when a loop.
            # all_grids=os.listdir(full_current_grids_data_dir)
            config_multi_pool=all_grids
            print('multi raw_grid_data starting')
            
            with multiprocessing.Pool() as pool:
                raw_results=pool.map(processData.build_all_raw_grid_data_per_grid,config_multi_pool)
            
            pool.close()
            pool.join()
            raw_all_raw_grid_data=np.concatenate(raw_results,axis=0)
            
            all_raw_grid_data=pd.DataFrame(columns=synth_columns_grid_data,data=raw_all_raw_grid_data)
            all_raw_grid_data=all_raw_grid_data[all_raw_grid_data['node_type']=='consumptionNode'] # As the producer data is the same as the consumption but summed, we avoid inneccesary rows.
        
        else:
            all_raw_grid_data=pd.DataFrame()
            for g in all_grids:
                aux_raw_data=processData.build_all_raw_grid_data_per_grid(g)
                aux_pd_data=pd.DataFrame(columns=synth_columns_grid_data,data=aux_raw_data)
                all_raw_grid_data=all_raw_grid_data.append(aux_pd_data[aux_pd_data['node_type']=='consumptionNode'])
        
        return all_raw_grid_data
    
    def build_clean_per_time_point(input_data):
        ''' This method deletes every point from a single grid data which is out of the distribution per 80 or 30 percentile upper and lower.'''
        aux_data=input_data[2]
        ntype=input_data[0]
        t=input_data[1]
        aux_energy=aux_data[(aux_data['time']==t) & (aux_data['node_type']==ntype)]['energy']        
        aux_overpercentile=np.percentile(aux_energy,80)
        aux_underpercentile=np.percentile(aux_energy,30)
        aux_clean_energy_average=np.mean([e for e in aux_energy if (e < aux_overpercentile) and (e > aux_underpercentile)])
        aux_clean_data=[[ntype,aux_clean_energy_average,t]]
        return aux_clean_data

    def build_clean_data_multi(input_data):
        ''' This method delete every point which is an anomaly '''
        clean_data=pd.DataFrame(columns=['node_type','energy','time'])
        config_multi_pool=list(itertools.product(np.unique(input_data['node_type']),range(max_time_steps),[input_data]))
        print('multi starting')

        with multiprocessing.Pool() as pool:
            raw_results=zip(*pool.map(processData.build_clean_per_time_point,config_multi_pool))
        
        pool.close()
        pool.join()
        
        raw_clean_data=list(raw_results)[0]
        for r in raw_clean_data:
            aux_clean_data=pd.DataFrame({'node_type':[r[0]],'energy':[r[1]],'time':[r[2]]})
            clean_data=clean_data.append(aux_clean_data)

        return clean_data

    def build_expected_behaviour_data():
        ''' This is the main process which builds the expected and ideal behaviour from all grid data. This data is used for the forecasting model'''
        all_grids=os.listdir(full_current_grids_data_dir)
        all_raw_grid_data=processData.build_all_raw_grid_data_multi(all_grids)
        expected_behaviour_data=processData.build_clean_data_multi(all_raw_grid_data)
        expected_behaviour_data.reset_index(drop=True,inplace=True)
        return expected_behaviour_data


class dataModels:
    def build_train_test_data(raw_data):
        ''' This method select which nodes are going to be used as training data or as test/evaluation data '''
        n_nodes=len(raw_data[raw_data['time']==0])
        train_test_ratio
        
        n_test=int(n_nodes/train_test_ratio)
        n_train=n_nodes-n_test
        
        train_consumption,test_consumption=raw_data[0:n_train*max_time_steps].reset_index(inplace=False),raw_data[n_train*max_time_steps:n_train*max_time_steps+n_test*max_time_steps].reset_index(inplace=False)
        
        return train_consumption,test_consumption
    
    def build_multistep_shifted_data(raw_data,t_diff,eval_train='train'):
        ''' This methods transform all the grids data from a timeseries to a supervised dataset. 
        Each row is composed of last t_diff time points. Each value is the difference between the actual value and the previous. So, our input for the models is not the energy but the difference of energy consumed.'''
        raw_useful_data=raw_data[['energy','issue_type','time','node_id']]
        multistep_shifted_data=pd.DataFrame()
        multistep_shifted_data['target']=raw_useful_data['issue_type'][:-t_diff]
        multistep_shifted_data['time']=raw_useful_data['time'][:-t_diff]
        multistep_shifted_data['node_id']=raw_useful_data['node_id'][:-t_diff]
        
        for t in range(0,t_diff+1):
            multistep_shifted_data[f't-{t}']=raw_useful_data.shift(t)['energy']
            if t!=0:
                multistep_shifted_data[f't-{t}']=raw_useful_data.shift(t-1)['energy']-raw_useful_data.shift(t)['energy']
        
        multistep_shifted_data.drop(['t-0'],axis=1,inplace=True)
        multistep_shifted_data.dropna(inplace=True)
        
        if eval_train=='train':
            targ_freq=dict(collections.Counter(multistep_shifted_data['target']))
            targ_freq_vals=list(targ_freq.values())
            red_percentage=1-targ_freq_vals[1]/targ_freq_vals[0]
            aux_ind_drop_regs=multistep_shifted_data[multistep_shifted_data.target==list(targ_freq.keys())[0]].sample(frac=red_percentage).index
            multistep_shifted_data.drop(aux_ind_drop_regs,inplace=True)
        
        multistep_shifted_data.reset_index(inplace=True,drop=True)
        
        return multistep_shifted_data
    
    def build_targets_input_data(raw_data):
        ''' This method is only a '''
        print(raw_data)
        input_columns=[i for i in raw_data.columns if 't-' in i]
        input_data=raw_data[input_columns].values
        targets=raw_data['target'].values
        return input_data,targets

    def build_detection_time_eval(preds,real_data):
        ''' This method calculates how long the models lasted to detect an issue. '''
        pd_preds=pd.DataFrame({'preds':preds,'time':real_data['time']})
        pd_preds.reset_index(inplace=True)
        real_data.reset_index(inplace=True)
        
        # STARTING TIME PREDICTION
        aux_starting_time=[0]*len(pd_preds)
        aux_acumulative_detection_time=[]
        aux_acumulative_detection_time_abs=[]
        for i in range(1,len(pd_preds)-1):
            if  pd_preds['preds'][i]=='broken' and pd_preds['preds'][i-1]=='broken':
                aux_starting_time[i]=aux_starting_time[i-1]
            elif pd_preds['preds'][i]=='broken' and not pd_preds['preds'][i-1]=='broken':
                aux_starting_time[i]=pd_preds['time'][i]
                aux_acumulative_detection_time.append(aux_starting_time[i]-real_data['starting_time'][i])
                aux_acumulative_detection_time_abs.append(abs(aux_starting_time[i]-real_data['starting_time'][i]))
        
        detection_time_eval=np.sum(np.array(aux_acumulative_detection_time))
        
        return detection_time_eval

class trainingModels:
    def do_train_all_models(train_data,best_default='default'):
        ''' This method trains all our type of models '''
        if best_default=='best':
            trainingModels.do_search_best_trained_rf_issue_model(train_data,classification_detection='detection')
            trainingModels.do_search_best_trained_rf_issue_model(train_data,classification_detection='classification')
        if best_default=='default':
            trainingModels.do_trained_rf_issue_model(train_data,classification_detection='detection')
            trainingModels.do_trained_rf_issue_model(train_data,classification_detection='classification')
        
        trainingModels.do_trained_lr_issue_detection_model(train_data)
        trainingModels.build_trained_prophet_forecast_model()
    
    def do_trained_rf_issue_model(train_data,aux_max_depth=None,aux_max_leaf_nodes=None,n_estimators=100,classification_detection='detection'):
        ''' This method trains sklear models '''
        rf = RandomForestClassifier(n_jobs=-1,verbose=1,random_state=42, \
            max_depth=aux_max_depth,max_leaf_nodes=aux_max_leaf_nodes,n_estimators=n_estimators)
        
        aux_train_data=train_data.copy()
        t_diff=len([c for c in aux_train_data.columns if 't-' in c])
        
        if classification_detection=='detection':
            aux_train_data['target']=[0 if t=='none' else 1 for t in aux_train_data['target']]        
        
        if classification_detection=='classification':
            aux_train_data=aux_train_data[aux_train_data['target']!='none']
        
        input_data,targets=dataModels.build_targets_input_data(aux_train_data)
        
        rf.fit(input_data, targets)
        
        pickle.dump(rf,open(f'{models_folder}/rf_issue_{classification_detection}_model_leafs_{aux_max_leaf_nodes}_estimators_{n_estimators}_depth_{aux_max_depth}_t-{t_diff}.sav','wb'))
        return rf
    
    def do_search_best_trained_rf_issue_model(train_data,aux_min_max_depth=1,aux_max_max_depth=4,aux_min_max_leaf_nodes=2,aux_max_max_leaf_nodes=5,aux_min_n_estimators=10,aux_max_n_estimators=100,classification_detection='detection'):
        ''' This method trains sklear models '''
        rf = RandomForestClassifier(n_jobs=-1,verbose=1,random_state=42)
        parameters = {'max_depth':np.arange(aux_min_max_depth,aux_max_max_depth,int((aux_max_max_depth-aux_min_max_depth)/10 + 1))
                        , 'max_leaf_nodes':np.arange(aux_min_max_leaf_nodes,aux_max_max_leaf_nodes,int((aux_max_max_leaf_nodes-aux_min_max_leaf_nodes)/10 + 1))
                        , 'n_estimators':np.arange(aux_min_n_estimators,aux_max_n_estimators,int((aux_max_n_estimators-aux_min_n_estimators)/5 + 1))
                        }
        clf=GridSearchCV(rf,parameters)
        aux_train_data=train_data.copy()
        t_diff=len([c for c in aux_train_data.columns if 't-' in c])
        if classification_detection=='detection':
            aux_train_data['target']=[0 if t=='none' else 1 for t in aux_train_data['target']]
        
        input_data,targets=dataModels.build_targets_input_data(aux_train_data)
        clf.fit(input_data,targets)
        pickle.dump(clf,open(f'{models_folder}/rf_issue_{classification_detection}_model_t-{t_diff}.sav','wb'))
        return clf
    
    def do_trained_lr_issue_detection_model(train_data):
        ''' This method trains sklear models '''
        aux_train_data=train_data.copy()
        t_diff=len([c for c in aux_train_data.columns if 't-' in c])
        aux_train_data['target']=[0 if t=='none' else 1 for t in aux_train_data['target']]
        input_data,targets=dataModels.build_targets_input_data(aux_train_data)
        lr = LogisticRegression(n_jobs=-1,random_state=42)
        lr.fit(input_data,targets)
        pickle.dump(lr,open(f'{models_folder}/lr_issue_detection_model_t-{t_diff}.sav','wb'))
        return lr
    
    def do_search_best_trained_lr_issue_detection_model(train_data):
        ''' This method trains sklear models '''
        train_data['target']=[0 if t=='none' else 1 for t in train_data['target']]
        t_diff=len([c for c in train_data.columns if 't-' in c])
        input_data,targets=dataModels.build_targets_input_data(train_data)
        logreg = LogisticRegression(n_jobs=-1,random_state=42)
        lr.fit(input_data,targets)
        pickle.dump(lr,open(f'{models_folder}/lr_issue_detection_model_t-{t_diff}.sav','wb'))
        return lr
    
    def build_trained_prophet_forecast_model():
        ''' This method trains prophet models on our data framework. '''
        train_data=processData.build_expected_behaviour_data()
        
        prophet_model=Prophet(yearly_seasonality=False,weekly_seasonality=False,daily_seasonality=True)
        
        raw_train_data=train_data['energy'].copy()
        
        now_now=datetime.now()
        start_time=now_now-timedelta(days=max_days)
        full_ds=[start_time+timedelta(minutes=minutes_per_point*t) for t in range(len(raw_train_data))]
        
        pd_train=pd.DataFrame({'ds':full_ds,'y':raw_train_data})
        pd_train.index=pd_train.ds
        pd_train=pd_train.drop('ds',axis=1)
        hourly_train_data=pd_train.resample(f'{minutes_per_point*60}s').sum()
        
        prophet_model.fit(pd.DataFrame({'ds':hourly_train_data.index,'y':hourly_train_data.y}))
        
        pickle.dump(prophet_model,open(f'{models_folder}/prophet_forecasting_model.sav','wb'))
        
        return prophet_model
        
    def build_prophet_forecast(prophet_model,predict_date):
        ''' This method makes prophet predictions on our data framework. '''
        future_ds=pd.DataFrame(columns=['ds'])
        future_ds['ds']=[predict_date+timedelta(seconds=minutes_per_point*60)]
        raw_predicts=prophet_model.predict(future_ds)
        prediction=raw_predicts['yhat'].values[-1]
        return prediction

class evaluateModels:
    
    def build_prophet_evaluation():
        ''' This method evaluates prophet models on our data framework. '''
        all_grid_filenames=list(os.listdir(full_current_grids_data_dir))
        all_raw_data=processData.build_all_raw_grid_data_multi(all_grid_filenames)
        start_date=datetime.now()
        os.chdir(f'{models_folder}')
        model_filename=glob(f'*prophet*')[0]
        os.chdir('../')
        prophet_model=pickle.load(open(f'{models_folder}/{model_filename}', 'rb'))
        evaluation=pd.DataFrame(columns=['node_id','time','issue_type','grid_name'])
        
        for g in all_grid_filenames:
            raw_data=processData.build_all_raw_grid_data_multi([g])
            for t in range(len(raw_data)):
                aux_node_id=raw_data['node_id'][t]
                aux_predict_date=raw_data['time'][t]
                predict_date=start_date+timedelta(minutes=aux_predict_date*minutes_per_point)
                aux_energy=raw_data['time'][t]
                prediction=trainingModels.build_prophet_forecast(prophet_model,predict_date)
                diff_pred=aux_energy/prediction
                
                if diff_pred>underconsume_min and diff_pred<underconsume_max:
                    aux_issue_type='underconsume'
                
                elif diff_pred>overconsume_min:
                    aux_issue_type='overconsume'
                    
                elif diff_pred<underconsume_min:
                    aux_issue_type='broken'
                
                else:
                    aux_issue_type='none'
                
                aux_evaluation=pd.DataFrame({'node_id':[aux_node_id],'time':[aux_predict_date],'grid_name':[g],'issue_type':aux_issue_type})
                evaluation=evaluation.append(aux_evaluation)
        
        evaluation['confusion_matrix']=confusion_matrix(all_raw_data['issue_type'],evaluation['issue_type'],labels=np.unique(aux_evaluation['issue_type']))
        
        issue_types=np.unique(aux_evaluation['issue_type'])
        
        for i in range(len(issue_types)):
            cm=evaluation['confusion_matrix'].copy()
            real_total=sum(cm[i,:])
            accurated_preds=cm[i,i]
            evaluation[f'accuracy_{i}']=accurated_preds/real_total        
        
        brief_evaluation=pd.DataFrame()
        for a in [c for c in evaluation.columns if 'accuracy' in c]:
            brief_evaluation[f'avg_{a}']=np.mean(evaluation[a])
        
        
        try:
            os.system(f'rm -fr ./{eval_folder}/{model_filename}')
        except:
            os.makedirs(f'{eval_folder}/{model_filename}')
            pass
        
        brief_evaluation.to_csv(f'./{eval_folder}/{model_filename}/pd_brief_evaluation.csv')
        
        return brief_evaluation,evaluation
    
    def build_model_evaluation_multi(evaluating_grids=[]):
        ''' This method evaluates sklearn models on our data framework. '''
        model_filenames=[i for i in list(os.listdir(f'{models_folder}')) if 'prophet' not in i]
        #evaluateModels.build_prophet_evaluation()
        
        for m in model_filenames:
            if len(evaluating_grids)>0:
                evaluateModels.build_model_evaluation_multigrid_per_model(m,evaluating_grids)
            else:
                evaluateModels.build_model_evaluation_multigrid_per_model(m)
    
    def build_model_evaluation_per_grid(config_multi_pool):
        ''' This method evaluates sklearn models on our data framework. '''
        grid_name=config_multi_pool[0]
        model_name=config_multi_pool[1]
        raw_eval_data=processData.build_all_raw_grid_data_multi([grid_name])
        loaded_model = pickle.load(open(f'{models_folder}/{model_name}', 'rb'))
        try:
            t_diff=loaded_model.n_features_
        except:
            t_diff=len(loaded_model.coef_[0])
        
        eval_data=dataModels.build_multistep_shifted_data(raw_eval_data,t_diff,eval_train='eval')
        
        if 'detection' in model_name:
            eval_data['target']=[0 if t=='none' else 1 for t in eval_data['target']]
        
        elif 'classification' in model_name:
            eval_data=eval_data[eval_data['target']!='none']
        
        input_data,targets=dataModels.build_targets_input_data(eval_data)
        
        evaluation={}
        evaluation['score']=loaded_model.score(input_data,targets)
        evaluation['predict_proba']=loaded_model.predict_proba(input_data)
        evaluation['predict']=loaded_model.predict(input_data)
        
        if 'detection' in model_name:
            evaluation['detection_time']=dataModels.build_detection_time_eval(evaluation['predict'],raw_eval_data[t_diff:-t_diff])
        
        evaluation['confusion_matrix']=confusion_matrix(targets,evaluation['predict'],labels=np.unique(targets))
        issue_types=np.unique(eval_data['target'])
        for i in range(len(issue_types)):
            cm=evaluation['confusion_matrix'].copy()
            real_total=sum(cm[i,:])
            accurated_preds=cm[i,i]
            evaluation[f'accuracy_{i}']=accurated_preds/real_total
        return evaluation
    
    
    def build_model_evaluation_multigrid_per_model(model_name,evaluating_grids=[]):
        ''' This method evaluates sklearn models on our data framework. '''
        if len(evaluating_grids)>0:
            config_multi_pool=[(g,model_name) for g in evaluating_grids]
        else:
            config_multi_pool=[(g,model_name) for g in list(os.listdir(full_current_grids_data_dir))]
        
        print('multi model_evaluation starting')
        
        with multiprocessing.Pool() as pool:
            raw_results=pool.map(evaluateModels.build_model_evaluation_per_grid,config_multi_pool)
        
        raw_all_raw_grid_eval=list(raw_results)
        pool.close()
        pool.join()
        
        full_evaluation=pd.DataFrame(columns=list(raw_all_raw_grid_eval[0].keys()),data=raw_all_raw_grid_eval)
        os.makedirs(f'./{eval_folder}/{model_name}',exist_ok=True)
        full_evaluation.to_pickle(f'./{eval_folder}/{model_name}/pd_full_evaluation.pkl')
        brief_evaluation=pd.DataFrame({'avg_score':[np.mean(full_evaluation['score'])]})
        for a in [c for c in full_evaluation.columns if 'accuracy' in c]:
            brief_evaluation[f'avg_{a}']=np.mean(full_evaluation[a])
        brief_evaluation.to_csv(f'./{eval_folder}/{model_name}/pd_brief_evaluation.csv')
        return full_evaluation
    
    def build_evaluation_report(loaded_model,aux_raw_data,aux_eval_data,classification_detection='detection',model_grid_evaluation='grid'):
        ''' This method evaluates sklearn models on our data framework. '''
        try:
            t_diff=loaded_model.n_features_
        except:
            t_diff=len(loaded_model.coef_[0])
        
        input_data,targets=dataModels.build_targets_input_data(aux_eval_data)
        
        evaluation,brief_evaluation={},{}
        evaluation['predict']=loaded_model.predict(input_data)
        
        # If this method is called to check the accuracy of the model other fields are added to the evaluation report
        if model_grid_evaluation=='model':
            evaluation['score']=loaded_model.score(input_data,targets)
            evaluation['predict_proba']=loaded_model.predict_proba(input_data)
            evaluation['confusion_matrix']=confusion_matrix(targets,evaluation['predict'],labels=np.unique(targets))
            issue_types=np.unique(aux_eval_data['target'])
            
            for i in range(len(issue_types)):
                cm=evaluation['confusion_matrix'].copy()
                real_total=sum(cm[i,:])
                accurated_preds=cm[i,i]
                evaluation[f'accuracy_{i}']=accurated_preds/real_total
            
            evaluation=pd.DataFrame(columns=list(evaluation.keys()),data=[evaluation])
            brief_evaluation=pd.DataFrame({'avg_score':[np.mean(evaluation['score'])]})
            for a in [c for c in evaluation.columns if 'accuracy' in c]:
                brief_evaluation[f'avg_{a}']=np.mean(evaluation[a])

            if classification_detection=='detection':
                evaluation['detection_time']=dataModels.build_detection_time_eval(evaluation['predict'],aux_raw_data[t_diff:-t_diff])
        
        else:
            evaluation['time']=aux_eval_data['time'].values
            evaluation['node_id']=aux_eval_data['node_id'].values
            evaluation=pd.DataFrame(columns=list(evaluation.keys()),data=evaluation)
            
            return evaluation
        
        return brief_evaluation, evaluation
    
    def build_evaluation_report_detection_and_classification(evaluating_grids=[]):
        ''' This method will take each grid data and detect the issues happening, classifying each one and reporting the root cause. '''
        if len(evaluating_grids)>0:
            all_grid_filenames=evaluating_grids
        else:
            all_grid_filenames=list(os.listdir(full_current_grids_data_dir))
        
        raw_data=processData.build_all_raw_grid_data_multi(all_grid_filenames)
        os.chdir(f'{models_folder}')
        if len(glob('*detection*')+glob('*classification*'))==2:
            detection_model_name=glob('*detection*')[0]
            classification_model_name=glob('*classification*')[0]
            
            detection_loaded_model = pickle.load(open(f'./{detection_model_name}', 'rb'))
            classification_loaded_model = pickle.load(open(f'./{classification_model_name}', 'rb'))
            os.chdir('../')
            
            try:
                t_diff=detection_loaded_model.n_features_
            except:
                t_diff=len(detection_loaded_model.coef_[0])
            
            aux_raw_data=raw_data.copy()
            eval_data=dataModels.build_multistep_shifted_data(aux_raw_data,t_diff,eval_train='eval')
            
            aux_eval_data=eval_data.copy()
            aux_detection_data=aux_eval_data.copy()
            
            aux_detection_data['target']=[0 if t=='none' else 1 for t in aux_eval_data['target']]
            aux_classification_data=aux_eval_data[aux_eval_data['target']!='none'].copy()
            
            # DETECT ISSUES
            detection_brief_evaluation, detection_evaluation=evaluateModels.build_evaluation_report(detection_loaded_model,aux_raw_data,aux_detection_data,'detection','model')
            # CLASSIFY ISSUES        
            classification_brief_evaluation, classification_evaluation=evaluateModels.build_evaluation_report(classification_loaded_model,aux_raw_data,aux_classification_data,'classification','model')
            
            return detection_brief_evaluation, classification_brief_evaluation
        else:
            print('There must be only 2 models, one for issue detection and another for classification.')

    def build_rootcause_evaluation(grid_data,grid,model_grid_evaluation='grid'):
        ''' This method gives you the data about rootcause of the major issues.'''
        aux_grid=grid.copy()
        t_issues=np.unique(grid_data[grid_data['issue_type']=='broken']['time'])
        t_issues.sort()
        t_issues=[t_issues[i] for i in range(len(t_issues)) if t_issues[i-1]!=(t_issues[i]-1) and i!=0]
        production_nodes=[i for i in aux_grid.nodes if 'production' in i]
        all_prueba=[]
        most_likely_rootcause_nodes=[]
        rootcauses_dict={}
        
        for t in t_issues:
            grid_snapshot=grid_data[grid_data['time']==t]
            if model_grid_evaluation=='model':
                true_rootcauses=list(np.unique(grid_snapshot[grid_snapshot['issue_type']=='broken']['rootcause']))
            affected_nodes=grid_snapshot[grid_snapshot['issue_type']=='broken']['node_id']
            not_affected_nodes=grid_snapshot[grid_snapshot['issue_type']!='broken']['node_id']
            
            rootcauses_dict[t]={}
            for n in affected_nodes:
                temp_grid=aux_grid.copy()
                temp_grid.remove_node(n)
                other_affected_nodes=[aux_n for aux_n in affected_nodes if aux_n!=n]
                
                if len(other_affected_nodes)>=1:
                    rootcauses_dict[t][n]={}
                    rootcauses_dict[t][n]['other_affected_nodes']={}
                    
                    for oan in other_affected_nodes:
                        rootcauses_dict[t][n]['other_affected_nodes'][oan]=gridAnalysis.get_broken_node_is_connected(temp_grid,oan,t,grid_data,production_nodes)
                    
                    rootcauses_dict[t][n]['prob_rootcause']=np.sum([1 for oan_connected in list(rootcauses_dict[t][n]['other_affected_nodes'].values()) if oan_connected==False])/len(other_affected_nodes)
                    
                    if rootcauses_dict[t][n]['prob_rootcause']==1:
                        break
                else:
                    rootcauses_dict[t][n]={}
                    rootcauses_dict[t][n]['other_affected_nodes']={}
                    rootcauses_dict[t][n]['prob_rootcause']=1
            
            if model_grid_evaluation=='model':
                max_prob_rootcause=max([rootcauses_dict[t][i]['prob_rootcause'] for i in rootcauses_dict[t]])
                aux_most_likely_rootcause_nodes=[i for i in rootcauses_dict[t] if rootcauses_dict[t][i]['prob_rootcause']==max_prob_rootcause]
                most_likely_rootcause_nodes.append([t,true_rootcauses,aux_most_likely_rootcause_nodes])
                rootcause_analysis=pd.DataFrame(columns=['starting_time','true_rootcauses','predicted_rootcauses'],data=most_likely_rootcause_nodes)
            
            elif model_grid_evaluation=='grid':
                max_prob_rootcause=max([rootcauses_dict[t][i]['prob_rootcause'] for i in rootcauses_dict[t]])
                aux_most_likely_rootcause_nodes=[i for i in rootcauses_dict[t] if rootcauses_dict[t][i]['prob_rootcause']==max_prob_rootcause]
                most_likely_rootcause_nodes.append([t,aux_most_likely_rootcause_nodes,max_prob_rootcause])
                rootcause_analysis=pd.DataFrame(columns=['starting_time','predicted_rootcauses','prob_rootcause'],data=most_likely_rootcause_nodes)
        
        return rootcause_analysis

class gridAnalysis:
    def get_broken_node_is_connected(grid,node,t,grid_w_issues,production_nodes):
        ''' This method tell us if a broken node was connected before or not.'''
        # Which nodes are broken?
        paths=[]
        
        for pn in production_nodes:
            try:
                aux_paths=list(nx.shortest_path(grid,pn,node))
                paths.append(aux_paths)
            except:
                continue
        
        if len(paths)>0:
            node_is_connected=True
        else:
            node_is_connected=False
        
        return node_is_connected
    
    def do_full_grid_analysis(training=False):
        ''' This method is the controller which read the grid data, train the models, evaluate them and analise the data given. You can choose to not train new models.'''
        all_grid_filenames=list(os.listdir(full_current_grids_data_dir))
        evaluating_grids=np.unique(random.choices(all_grid_filenames, k=int(len(all_grid_filenames)/train_test_ratio)))
        training_grids=[i for i in all_grid_filenames if i not in evaluating_grids]
        t_diff=6
        
        # TRAINING THE MODEL
        raw_data=processData.build_all_raw_grid_data_multi(training_grids)
        raw_train_data,raw_test_data=dataModels.build_train_test_data(raw_data)
        train_data=dataModels.build_multistep_shifted_data(raw_train_data,t_diff,'train')
        
        if training:
            trainingModels.do_train_all_models(train_data)
        
        # EVALUATING THE MODEL
        model_evaluation=evaluateModels.build_model_evaluation_multi(evaluating_grids)
        
        # ISSUE ANALYSIS OF THE GRIDS
        all_classification_evaluation, all_rootcause_analysis=gridAnalysis.do_full_issue_detection_report(evaluating_grids)
        gridAnalysis.do_brief_issues_report_for_humans(all_classification_evaluation, all_rootcause_analysis)
        gridAnalysis.do_brief_trends_report_for_humans(all_classification_evaluation, all_rootcause_analysis)
        
    def do_full_issue_detection_report(evaluating_grids=[]):
        ''' This method will take each grid data and detect the issues happening, classifying each one and reporting the root cause. '''
        if len(evaluating_grids)>0:
            all_grids=evaluating_grids
        else:
            all_grids=os.listdir(full_current_grids_data_dir)
        
        os.chdir(f'{models_folder}')
        
        detection_model_name=glob('*detection*')[0]
        classification_model_name=glob('*classification*')[0]
        
        detection_loaded_model = pickle.load(open(f'./{detection_model_name}', 'rb'))
        classification_loaded_model = pickle.load(open(f'./{classification_model_name}', 'rb'))
        
        os.chdir('../')
        all_rootcause_analysis,all_classification_evaluation=pd.DataFrame(),pd.DataFrame()
        for grid_name in all_grids:
            #raw_eval_data=pd.read_csv(f'{full_current_grids_data_dir}/{grid_name}/{filename_synth_data}')
            raw_eval_data=processData.build_all_raw_grid_data_multi([grid_name])
            # DETECT ISSUES
            try:
                t_diff=detection_loaded_model.n_features_
            except:
                t_diff=len(detection_loaded_model.coef_[0])
            
            eval_data=dataModels.build_multistep_shifted_data(raw_eval_data,t_diff,eval_train='eval')
            
            aux_detection_data=eval_data.copy()
            aux_detection_data['target']=[0 if t=='none' else 1 for t in aux_detection_data['target']]
            
            # DETECT ISSUES
            detection_evaluation=evaluateModels.build_evaluation_report(detection_loaded_model,raw_eval_data,aux_detection_data,'detection','grid')
            
            # CLASSIFY ISSUES        
            aux_classification_data=eval_data.copy()
            aux_classification_data=aux_classification_data[detection_evaluation['predict'].values==1]
            classification_evaluation=evaluateModels.build_evaluation_report(classification_loaded_model,raw_eval_data,aux_classification_data,'classification','grid')
            
            # ROOTCAUSE ANALYSIS
            aux_grid_relations=pd.read_csv(f'{full_current_grids_data_dir}/{grid_name}/nodes_relationships.csv')
            aux_grid=gridTasks.build_grid_from_csv(aux_grid_relations)
            
            predict_grid_data=raw_eval_data[t_diff:-t_diff].copy()
            predict_grid_data=classification_evaluation.copy()
            predict_grid_data=predict_grid_data.rename(columns={'predict':'issue_type'})
            
            rootcause_analysis=evaluateModels.build_rootcause_evaluation(predict_grid_data,aux_grid,'grid')
            
            if not os.path.exists(f'{grid_analysis_folder}/{grid_name}'):
                os.makedirs(f'{grid_analysis_folder}/{grid_name}')
            
            rootcause_analysis.to_csv(f'{grid_analysis_folder}/{grid_name}/rootcause_analysis.csv',index=False)
            classification_evaluation.to_csv(f'{grid_analysis_folder}/{grid_name}/classification_evaluation.csv')
            
            classification_evaluation['grid_name']=[grid_name]*len(classification_evaluation)
            all_classification_evaluation=all_classification_evaluation.append(classification_evaluation)
            
            rootcause_analysis['grid_name']=[grid_name]*len(rootcause_analysis)
            all_rootcause_analysis=all_rootcause_analysis.append(rootcause_analysis)
        
        all_classification_evaluation.reset_index(drop=True,inplace=True)
        t_issues=all_classification_evaluation['time']
        new_t_issues=t_issues.copy()
        for t in range(1,len(new_t_issues)):
            if t_issues[t]==t_issues[t-1]+1 and all_classification_evaluation['predict'][t]==all_classification_evaluation['predict'][t-1] and all_classification_evaluation['grid_name'][t]==all_classification_evaluation['grid_name'][t-1]:
                new_t_issues[t]=new_t_issues[t-1]
        
        all_classification_evaluation['starting_time']=new_t_issues
        
        for i in range(1,len(all_classification_evaluation)):
            now_predict,last_predict=all_classification_evaluation['predict'][i],all_classification_evaluation['predict'][i-1]
            now_node_id,last_node_id=all_classification_evaluation['node_id'][i],all_classification_evaluation['node_id'][i-1]
            now_grid_name,last_grid_name=all_classification_evaluation['grid_name'][i],all_classification_evaluation['grid_name'][i-1]
            
            if now_predict=='underconsume' and last_predict=='overconsume' and last_node_id==now_node_id and last_grid_name==now_grid_name:
                all_classification_evaluation['predict'][i]='overconsume'
            elif last_predict=='underconsume' and now_predict=='overconsume' and last_node_id==now_node_id and last_grid_name==now_grid_name:
                all_classification_evaluation['predict'][i]='underconsume'
        
        
        all_classification_evaluation.to_csv(f'{full_report_folder}/issues_detection.csv',index=False)
        all_rootcause_analysis.to_csv(f'{full_report_folder}/root_cause_analysis.csv',index=False)
        
        return all_classification_evaluation, all_rootcause_analysis
    
    def do_brief_issues_report_for_humans(all_classification_evaluation, all_rootcause_analysis):
        ''' This print a summary of the issues per grid of every node.'''
        print('This is a really long task because it will show you every grid performance of the last days. It is only visual, you have all the data on the right folders. If you want to skip it you can:')
        wrong_answer=True
        while wrong_answer:
            skip_it=input('Do you want to skip it? It could be useful but sometimes it is too much because there might be some false issues and all of them are shown here with human reading speed in mind [Skip/Keep]')
            if skip_it.lower()=='skip':
                wrong_answer=False
                return 0
            
            elif skip_it.lower()=='keep':
                now_now=datetime.now()
                start_date=now_now-timedelta(days=max_days)
                all_classification_evaluation['day']=[datetime.date(start_date+timedelta(minutes=t*minutes_per_point)) for t in all_classification_evaluation['starting_time']]
                all_rootcause_analysis['day']=[datetime.date(start_date+timedelta(minutes=t*minutes_per_point)) for t in all_rootcause_analysis['starting_time']]
                
                all_classification_evaluation['complete_date']=[start_date+timedelta(minutes=t*minutes_per_point) for t in all_classification_evaluation['starting_time']]
                all_rootcause_analysis['complete_date']=[start_date+timedelta(minutes=t*minutes_per_point) for t in all_rootcause_analysis['starting_time']]
                
                all_classification_evaluation['count']=1
                all_rootcause_analysis['count']=1
                
                unique_classification_evaluation=all_classification_evaluation[['predict','grid_name','day','count','complete_date','node_id']].drop_duplicates()
                
                issue_dates=np.unique(list(all_rootcause_analysis['day'])+list(unique_classification_evaluation['day']))
                all_rootcause_analysis['str_predicted_rootcauses']=[",".join(pr) for pr in all_rootcause_analysis['predicted_rootcauses']]
                
                for grid in np.unique(unique_classification_evaluation['grid_name']):
                    all_rootcause_analysis_per_grid=all_rootcause_analysis[all_rootcause_analysis['grid_name']==grid]
                    unique_classification_evaluation_per_grid=unique_classification_evaluation[unique_classification_evaluation['grid_name']==grid]
                    print(f'THIS IS THE REPORT ABOUT GRID {grid}!')
                    print(f'\n \n \n')
                    time.sleep(3)
                    for d in issue_dates:
                        time.sleep(0.5)
                        day_rootcauses_data=all_rootcause_analysis_per_grid[all_rootcause_analysis_per_grid['day']==d]
                        day_classification_evaluation_data=unique_classification_evaluation_per_grid[unique_classification_evaluation_per_grid['day']==d]
                        
                        # BROKE
                        for n in np.unique(day_rootcauses_data['str_predicted_rootcauses']):
                            time.sleep(1)
                            day_rootcauses_data_per_node=day_rootcauses_data[day_rootcauses_data['str_predicted_rootcauses']==n]
                            aux_all_issue_times=[':'.join(str(datetime.time(cd)).split(':')[0:2]) for cd in day_rootcauses_data_per_node['complete_date']]
                            aux_all_issue_times.sort()
                            aux_complete_times=', '.join(aux_all_issue_times)
                            print(f'The node/s {n} suffered a total shut down on {d} at these times: {aux_complete_times}')
                            
                        # MINOR ISSUES
                        for n in np.unique(day_classification_evaluation_data['node_id']):
                            time.sleep(1)
                            day_classification_evaluation_data_node=day_classification_evaluation_data[day_classification_evaluation_data['node_id']==n]
                            aux_all_issue_times=[':'.join(str(datetime.time(cd)).split(':')[0:2]) for cd in day_classification_evaluation_data_node['complete_date']]
                            aux_all_issue_times.sort()
                            aux_complete_times=', '.join(aux_all_issue_times)
                            aux_each_issue=day_classification_evaluation_data_node['predict']
                            for aux_issue in np.unique(aux_each_issue):
                                count_issues=len(day_classification_evaluation_data_node[day_classification_evaluation_data_node['predict']==aux_issue])
                                print(f'The node/s {n} suffered {count_issues} {aux_issue} behaviour on {d} at these times: {aux_complete_times}')
                print('If there is to much information here, we remind you there is two csv on the folders which are named in the intructions of use.')
                wrong_answer=False
            else:
                print('You did not write Skip nor Keep as an answer. Please take into account, you must write Skip or Keep only.')
                print('\n \n \n')
                continue
    
    
    def do_brief_trends_report_for_humans(all_classification_evaluation, all_rootcause_analysis):
        ''' This makes some plots about the grids following the predictions of the models.'''
        now_now=datetime.now()
        start_date=now_now-timedelta(days=max_days)
        all_classification_evaluation['hour']=[(start_date+timedelta(minutes=t*minutes_per_point)).hour for t in all_classification_evaluation['starting_time']]
        all_rootcause_analysis['hour']=[(start_date+timedelta(minutes=t*minutes_per_point)).hour for t in all_rootcause_analysis['starting_time']]
        
        all_classification_evaluation['count']=1
        all_rootcause_analysis['count']=1
        
        pd_issues_list=all_classification_evaluation[['predict','grid_name','starting_time','hour','count']].drop_duplicates()

        issue_by_hour=pd_issues_list.groupby(['predict','hour'],as_index=False).sum(['count'])
        total_issues_by_hour=pd_issues_list.groupby(['hour'],as_index=False).sum(['count'])
        #issue_dates=np.unique(list(all_rootcause_analysis['day'])+list(all_classification_evaluation['day']))
        
        # Issue type per day hour
        bottom_bar=0*len(issue_by_hour)
        fig, ax = plt.subplots()
        try:
            for p in np.unique(issue_by_hour['predict']):
                aux_data=issue_by_hour[issue_by_hour['predict']==p]
                ax.bar(aux_data['hour'],aux_data['count'],bottom=bottom_bar,label=p)
                if len(aux_data['count'])!=24:
                    lacking_hours=[i for i in range(0,24) if i not in aux_data['hour'].values]
                    for l in lacking_hours:
                        aux_data=aux_data.append(pd.DataFrame({'hour':[l],'count':[0],'predict':['none']}))
                aux_data.reset_index(inplace=True,drop=True)
                aux_bottom_bar=aux_data['count'].copy()
                bottom_bar=bottom_bar+aux_bottom_bar.values
            
            ax.set_ylabel('Number of incidences')
            plt.xticks(aux_data['hour'])
            ax.set_title(f'Number of issues splitted by type detected each hour on the last {max_days} days')
            ax.legend()
            plt.savefig(f'{full_report_folder}/issues_count_per_hour.png')
            plt.close()
            
            # Issue distribution per hour
            bottom_bar=0*len(issue_by_hour)
            fig, ax = plt.subplots()
            
            for p in np.unique(issue_by_hour['predict']):
                aux_data=issue_by_hour[issue_by_hour['predict']==p]
                aux_percentage_issue=aux_data['count'].values/total_issues_by_hour['count'].values
                ax.bar(aux_data['hour'],aux_percentage_issue,bottom=bottom_bar,label=p)
                bottom_bar=bottom_bar+aux_percentage_issue
            
            ax.set_ylabel('Percentage of incidences')
            plt.xticks(aux_data['hour'])
            ax.set_title(f'Distribution of issues splitted by type detected each hour on the last {max_days} days')
            ax.legend()
            plt.savefig(f'{full_report_folder}/issues_distribution_per_hour.png')
            plt.close()
        except:
            print('There is not enough data to build the plot.')



