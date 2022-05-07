from grid_funcs import *
from detect_funcs import *
from variables import *
import os
import sys

class sanemMain:
    def do_full_execution():
        os.chdir('full_execution')
        os.system(f'rm -fr {full_current_grids_data_dir}')
        os.makedirs(f'{full_current_grids_data_dir}')
        
        print('Your simulated grids are being built:')
        gridTasks.do_multiple_main_full_flow(num_grids)
        print('Your simulated grids are already built!')
        
        os.system(f'rm -fr {models_folder}')
        os.makedirs(f'{models_folder}')
        
        os.system(f'rm -fr {grid_analysis_folder}')
        os.makedirs(f'{grid_analysis_folder}')
        
        print('Now starts the model training, evaluating and grid analysis.')
        gridAnalysis.do_full_grid_analysis(training=True)
        print('All is finish and you can check the result in the folders which are named in the instructions.')
    
    def do_custom_flow():
        os.chdir('custom_flow')
        print('This product will let you create a grid, train models and detect issues. You must choose to do all of those or just some of thems.')
        print("Let's begin!")
        
        wrong_answer=True
        while wrong_answer:
            added_grid_data=input('Did you add your grids data? [Yes/No]')
            if added_grid_data.lower()=='yes':
                wrong_answer=False
                continue
            
            elif added_grid_data.lower()=='no':
                os.system(f'rm -fr {grid_data_folder_name}')
                os.system(f'cp -r {grid_data_folder_name}_default {grid_data_folder_name}')
                pass
            else:
                print('You must write exactly one of these options: \n \t - Yes \n \t - No \n')
                continue
            
            print('You can put some grid data on the grids_data folder as it is said on the documentation. If you have not...')
            do_create_grid_data=input('Do you want to create some grids or you want to take the 10 we give you by default? [Yes/No]')
            if do_create_grid_data.lower()=='yes':
                os.system(f'rm -fr {grid_data_folder_name}/*')
                sanemSubMain.main_grid_synthetic_data_with_issues()
                wrong_answer=False
            elif do_create_grid_data.lower()=='no':
                wrong_answer=False
            else:
                print('You must write exactly one of these options: \n \t - Yes \n \t - No \n')
                continue
        
        wrong_answer=True
        while wrong_answer:
            print('You can put some trained models on the models_trained folder as it is said on the documentation. \n')
            added_trained_models=input('Did you add your own trained models? [Yes/No]')
            if added_trained_models.lower()=='yes':
                wrong_answer=False
                continue
            elif added_trained_models.lower()=='no':
                
                print("If you didn't add any trained model and you created less than a hundred of grids, we recommend you to use our trained models by default. \n")
                print('Do you want to train some new models? [Yes/No] \n')
                
                do_train_models=input()
                
                if do_train_models.lower()=='yes':
                    os.system(f'rm -fr {models_folder}')
                    os.mkdir(f'{models_folder}')
                    sanemSubMain.main_train_models()
                    wrong_answer=False
                
                elif do_train_models.lower()=='no':
                    os.system(f'rm -fr {models_folder}')
                    os.system(f'cp -r {models_folder}_default {models_folder}')
                    wrong_answer=False
                
                else:
                    print('You must write exactly one of these options: \n \t - Yes \n \t - No \n')
                    continue
            
            else:
                print('You must write exactly one of these options: \n \t - Yes \n \t - No \n')
                continue
        
        
        wrong_answer=True
        while wrong_answer:
            print('Do you want to analize the issues which happened on the grids? [Yes/No] \n')
            do_detect_issues=input()
            if do_detect_issues.lower()=='yes':
                print("Your grids are being analize...")
                all_classification_evaluation, all_rootcause_analysis=gridAnalysis.do_full_issue_detection_report()
                print("Your grids were analized! You have the report of issues and root cause in the folder of each grid.")
                sanemSubMain.main_detect_issues(all_classification_evaluation, all_rootcause_analysis)
                #os.system('python main_detect_issues.py')
                wrong_answer=False
            elif do_detect_issues.lower()=='no':
                wrong_answer=False
            else:
                print('You must write exactly one of these options: \n \t - Yes \n \t - No \n')
                continue
        
        
class sanemSubMain:
    def main_grid_synthetic_data_with_issues():
        wrong_answer=True
        while wrong_answer:
            print('You must set a number of grids you want to make.')
            try:
                n_grids=int(input('How many grids simulations you want to make?'))
                wrong_answer=False
            except:
                print('Write only an integer number number.')
                continue
        
        print("Your grids are starting to be created...")
        gridTasks.do_multiple_main_full_flow(n_grids)
        print("Your simulated grids are already created!")
    
    def main_train_models():
        import os
        import sys
        
        wrong_answer=True
        while wrong_answer:
            print('Do you want to Classify issues, Detect issues or both? [Classify/Detect/Both] \n')
            try:
                analysis_type=(input())
                if analysis_type not in ('Classify','Detect','Both'):
                    print('You must write exactly one of these options: \n \t - Classify \n \t - Detect \n \t - Both \n')
                    continue
                wrong_answer=False
            except:
                print('You must write exactly one of these options: \n \t - Classify \n \t - Detect \n \t - Both \n')
                continue
        
        wrong_answer=True
        while wrong_answer:
            try:
                t_diff_asked=int(input('How many time points you want to have into account to detect changes on the energy consumption trend? Usually 3 points, so 30 minutes, is enough. \n'))
                wrong_answer=False
                continue
            except:
                print('You must write only an integer!')
                continue
        
        wrong_answer=True
        while wrong_answer:
            if analysis_type=='Detect' or analysis_type=='Both':
                print('You must decide if you want a RandomForest or a LogisticRegression for the detection model. [RandomForest/LogisticRegression] \n')
            else:
                break
            try:
                detect_model_type=input()
                if detect_model_type not in ('RandomForest','LogisticRegression'):
                    print('You must write exactly one of these options: \n \t - RandomForest \n \t - LogisticRegression \n')
                    continue
                wrong_answer=False
            except:
                print('You must write exactly one of these options: \n \t - RandomForest \n \t - LogisticRegression \n')
                continue
        
        print('We are reading the grid data...')
        all_grid_filenames=list(os.listdir(full_current_grids_data_dir))
        evaluating_grids=np.unique(random.choices(all_grid_filenames, k=int(len(all_grid_filenames)/train_test_ratio)))
        training_grids=[i for i in all_grid_filenames if i not in evaluating_grids]
        #raw_data=processData.build_all_raw_grid_data_multi(training_grids)
        raw_data=pd.DataFrame()
        for tg in training_grids:
            aux_raw_data=pd.read_csv(f'{grid_data_folder_name}/{tg}/synth_energy_data.csv')
            for nt in np.unique(aux_raw_data['node_id']):
                aux_node_registers=aux_raw_data['node_id']==nt
                aux_data_per_type=aux_raw_data[aux_node_registers].copy()
                aux_raw_data['energy'].loc[aux_node_registers]=MinMaxScaler().fit_transform(np.array(aux_data_per_type['energy']).reshape(-1,1))[:,0]
            aux_raw_data['node_type']=[n.split("_")[0] for n in aux_raw_data['node_id']]
            raw_data=raw_data.append(aux_raw_data)
            print(f'We read {tg} grid data.')
        
        raw_train_data,raw_test_data=dataModels.build_train_test_data(raw_data)
        
        print(f'We are starting to build the training data from the simulated grid data...')
        train_data=dataModels.build_multistep_shifted_data(raw_train_data,t_diff_asked,'train')
        print(f'We have finished creating the training data ...')
        
        wrong_answer=True
        while wrong_answer:
            if analysis_type=='Detect' or analysis_type=='Both':
                print('We are going to train the detection model.')
                if detect_model_type=='RandomForest':
                    print('Now you must define some configuration for the training.')
                    try:
                        aux_max_leaf_nodes=int(input('Which are the maximum leaves per node? The default value is 2.'))
                    except:
                        aux_max_leaf_nodes=2
                        print('The value of maximum leaves per node was set to 2. If you wrote a value, it had have to be an integer.')
                        pass
                    
                    try:
                        aux_n_estimators=int(input('Which are the estimators? The default value is 50.'))
                    except:
                        aux_n_estimators=50
                        print('The value of estimators was set to 50. If you wrote a value, it had have to be an integer.')
                        pass
                    
                    try:
                        aux_max_depth=int(input('Which are the maximum depth? The default value is 4.'))
                    except:
                        aux_max_depth=4
                        print('The value of maximum depth was set to 4. If you wrote a value, it had have to be an integer.')
                        pass
                    
                    trainingModels.do_trained_rf_issue_model(train_data,aux_max_depth,aux_n_estimators,aux_max_leaf_nodes,classification_detection='detection')
                
                elif detect_model_type=='LogisticRegression':
                    trainingModels.do_trained_lr_issue_detection_model(train_data)
                
                else:
                    print('You must write exactly one of these options: \n \t - RandomForest \n \t - LogisticRegression \n')
                    continue
            
            if analysis_type=='Classify' or analysis_type=='Both':
                print('Now we need a random forest classification model for figuring out if the node is suffering a complete shut down issue or it is only behaving strangely.')
                try:
                    aux_max_leaf_nodes=int(input('Which are the maximum leaves per node? The default value is 2.'))
                except:
                    aux_max_leaf_nodes=2
                    print('The value of maximum leaves per node was set to 2. If you wrote a value, it had have to be an integer.')
                    pass
                
                try:
                    aux_n_estimators=int(input('Which are the estimators? The default value is 50.'))
                except:
                    aux_n_estimators=50
                    print('The value of estimators was set to 50. If you wrote a value, it had have to be an integer.')
                    pass
                
                try:
                    aux_max_depth=int(input('Which are the maximum depth? The default value is 4.'))
                except:
                    aux_max_depth=4
                    print('The value of maximum depth was set to 4. If you wrote a value, it had have to be an integer.')
                    pass
                
                trainingModels.do_trained_rf_issue_model(train_data,aux_max_depth,aux_n_estimators,aux_max_leaf_nodes,classification_detection='classification')
            
            wrong_answer=False
    
    def main_detect_issues(all_classification_evaluation, all_rootcause_analysis):
        gridAnalysis.do_brief_issues_report_for_humans(all_classification_evaluation, all_rootcause_analysis)
        gridAnalysis.do_brief_trends_report_for_humans(all_classification_evaluation, all_rootcause_analysis)


    