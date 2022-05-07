import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from variables import *
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
from networkx.algorithms.traversal.depth_first_search import dfs_edges
import random
import os
from time import time
import itertools

class consumerNode:
    def get_model_consumer_energy_data(consumer_model_name='sinus_signal_4'):
        ''' Calculates the ideal behaviour of a consumer node '''
        if consumer_model_name=='sinus_signal_4':
            all_points_sin=np.linspace(0,max_days*2*np.pi,max_time_steps) # The full time range in radians
            raw_model_consumer_energy_data_ideal=np.sin(all_points_sin)+abs(0.2*np.sin(4*all_points_sin+np.pi))-0.2*np.sin(4*all_points_sin+0.75*np.pi)
            raw_model_consumer_energy_data=raw_model_consumer_energy_data_ideal*np.random.normal(1,noise_percentage,len(raw_model_consumer_energy_data_ideal))
            model_consumer_energy_data=raw_model_consumer_energy_data+abs(min(raw_model_consumer_energy_data))
        return model_consumer_energy_data
    
    def build_issue_data():
        ''' This method creates completely random issues on nodes considering each node independent from the rest in the grid. The method build_grid_w_issues consider that the grid is actually connected and a severe issue must affect the rest of nodes.'''
        raw_data=consumerNode.get_model_consumer_energy_data()
        issue_types=['overconsume','broken','underconsume','none']
        issue_weights=[overconsume_population,broken_population,underconsume_population]
        issue_weights.append(100-sum(issue_weights))
        step_points=int(issue_duration_mean*points_per_min)
        aux_max_steps=int(len(raw_data)/step_points)
        raw_issue_property=pd.DataFrame({'issue_type':['none']*len(raw_data),'starting_time':[0]*len(raw_data)})
        
        for s in range(aux_max_steps):        
            issue_type=random.choices(issue_types, weights=issue_weights, k=1)[0]
            
            if s*step_points>=(aux_max_steps-1):
                issue_duration=step_points
            else:
                issue_duration=int(np.random.normal(step_points,int(step_points*0.3),1))
            
            aux_issue_data=raw_data[s*step_points:s*step_points+issue_duration].copy()
            
            if issue_type=='overconsume':
                raw_data[s*step_points:s*step_points+issue_duration]=aux_issue_data*np.random.uniform(overconsume_min, overconsume_max,issue_duration)
                raw_issue_property.loc[s*step_points:s*step_points+issue_duration-1,:]=pd.DataFrame({'issue_type':[issue_type]*issue_duration,'starting_time':[s*step_points]*issue_duration}).values
            elif issue_type=='broken':
                raw_data[s*step_points:s*step_points+issue_duration]=aux_issue_data*np.random.uniform(0, 0.001,issue_duration)
                raw_issue_property.loc[s*step_points:s*step_points+issue_duration-1,:]=pd.DataFrame({'issue_type':[issue_type]*issue_duration,'starting_time':[s*step_points]*issue_duration}).values
            elif issue_type=='underconsume':
                raw_data[s*step_points:s*step_points+issue_duration]=aux_issue_data*np.random.uniform(underconsume_min, underconsume_max,issue_duration)
                raw_issue_property.loc[s*step_points:s*step_points+issue_duration-1,:]=pd.DataFrame({'issue_type':[issue_type]*issue_duration,'starting_time':[s*step_points]*issue_duration}).values
            else:
                raw_data[s*step_points:s*step_points+issue_duration]=raw_data[s*step_points:s*step_points+issue_duration]
        
        issue_data=raw_data
        return raw_issue_property, issue_data 

    def build_consumer_ideal_energy_data(init_energy_data,consumer_nodes):
        ''' Calculates the consumption in all the time simulated for every consumer node '''
        consumer_ideal_energy_data=init_energy_data.copy()
        for i in consumer_nodes:
            raw_energy_model_data=consumerNode.get_model_consumer_energy_data('sinus_signal_4')
            raw_energy=MinMaxScaler().fit_transform(raw_energy_model_data.reshape(-1,1))[:,0] # This is used to bring the scale to [0,1] so we can multiple the value to the max of each consumer
            aux_consumer_ideal_energy_data=pd.DataFrame(data=list(zip([i]*len(raw_energy),raw_energy,all_steps)),columns=init_energy_data.columns)
            consumer_ideal_energy_data=consumer_ideal_energy_data.append(aux_consumer_ideal_energy_data,ignore_index=True)
        return consumer_ideal_energy_data


    def build_consumer_energy_data(init_energy_data,consumer_nodes):
        ''' Calculates the consumption in all the time simulated for every consumer node '''
        consumer_energy_data=init_energy_data.copy()
        for i in consumer_nodes:
            #raw_energy_model_data=consumerNode.get_model_consumer_energy_data('sinus_signal_4')
            raw_issue_property,raw_energy_model_data=consumerNode.build_issue_data()
            raw_energy=MinMaxScaler().fit_transform(raw_energy_model_data.reshape(-1,1))[:,0] # This is used to bring the scale to [0,1] so we can multiple the value to the max of each consumer
            aux_energy_data=pd.DataFrame(data=list(zip([i]*len(raw_energy),raw_energy,all_steps)),columns=init_energy_data.columns)
            aux_consumer_energy_data=pd.concat([aux_energy_data,raw_issue_property],axis=1)
            consumer_energy_data=consumer_energy_data.append(aux_consumer_energy_data,ignore_index=True)
        return consumer_energy_data

class producerNode:
    def get_model_producer_energy_data(producer,consumer_energy_data,consumers_to_supply, producer_model_name='add_all_consumers'):
        ''' Calculates the ideal behaviour of a consumer node '''
        model_producer_energy_data=consumer_energy_data.copy()
        if producer_model_name=='add_all_consumers':
            model_producer_energy_data=consumer_energy_data[[n in consumers_to_supply for n in consumer_energy_data['node_id']]].groupby(by='time',as_index=False).sum()
            model_producer_energy_data['node_id']=[producer]*len(model_producer_energy_data)
            model_producer_energy_data['issue_type']=['none']*len(model_producer_energy_data)
            model_producer_energy_data['starting_time']=[0]*len(model_producer_energy_data)
        return model_producer_energy_data
    
    def build_producer_energy_data(consumer_energy_data,production_nodes,*aux_grid):
        ''' Calculates the production in all the time simulated for every producer node '''
        producer_energy_data=consumer_energy_data.copy()
        for producer in production_nodes:
            # Get the consumers nodes related to the producer
            raw_consumers_to_supply=relations_data[relations_data['node_id_origin']==producer]['node_id_destination'].values
            if isinstance(aux_grid, tuple):
                aux_grid=aux_grid[0]
            
            if aux_grid:
                first_consumer=[i for i in aux_grid.successors(producer)][0]
                consumers_to_supply=list(np.unique([ed[0] for ed in dfs_edges(aux_grid, first_consumer)]))
            else:
                consumers_to_supply=list(set(raw_consumers_to_supply).difference(set(production_nodes)))
            
            # Calculate the producer energy value:
            aux_energy_data=producerNode.get_model_producer_energy_data(producer,consumer_energy_data,consumers_to_supply,'add_all_consumers')
            aux_energy_data['issue_type']='none'
            aux_energy_data['starting_time']='0'
            aux_energy_data['rootcause']='0'
            producer_energy_data=producer_energy_data.append(aux_energy_data,ignore_index=True)
        
        return producer_energy_data


class processingEnergyData:
    def build_real_data(energy_data):
        ''' This method transform from the model to real consuming values.'''
        for cn in property_data[property_data['node_type']=='consumptionNode']['node_id']:
            max_value=property_data[property_data['node_id']==cn]['max_value'].iloc[0]
            aux_node_registers=(energy_data['node_id']==cn)
            aux_unmodified_data=energy_data.loc[aux_node_registers]['energy']
            energy_data['energy'].loc[aux_node_registers]=aux_unmodified_data*max_value
        return energy_data

class gridTasks:
    def build_grid_from_csv(relations_data):
        ''' This method is essential for building a standard network as networkx object from my csv property standard'''
        grid_from_csv=nx.DiGraph()
        # Networkx GRID CONSTRUCTION
        for index,edge in relations_data.iterrows():
            grid_from_csv.add_edge(edge[0],edge[1])
            grid_from_csv[edge[0]][edge[1]]['relation_type']=edge[2]
        return grid_from_csv
    
    def get_node_is_connected(grid,node,t,grid_w_issues,production_nodes):
        # Which nodes are broken?
        broken_nodes=grid_w_issues[(grid_w_issues['issue_type']=='broken') & (grid_w_issues['time']==t)]['node_id']
        if node not in broken_nodes:
            temp_grid=grid.copy()
            temp_grid.remove_nodes_from(broken_nodes)
            paths=[]
            
            for pn in production_nodes:
                try:
                    aux_paths=list(nx.shortest_path(temp_grid,pn,node))
                    paths.append(aux_paths)
                except:
                    continue
            
            if len(paths)>0:
                node_is_connected=True
            else:
                node_is_connected=False
        else:
            node_is_connected=False
        
        return node_is_connected

    def build_nodes_w_issues(aux_grid_w_issues,issue_type,issue_duration,n,t):
        ''' This method modifies the node data depending on the issue is suffering. '''
        step_points=int(issue_duration_mean*points_per_min)
        
        aux_affected_registers = (aux_grid_w_issues['node_id']==n) & (aux_grid_w_issues['time']>=t) & (aux_grid_w_issues['time']<t+issue_duration)
        aux_issue_data=aux_grid_w_issues[aux_affected_registers]['energy'].values
        
        if len(aux_issue_data)< issue_duration:
            issue_duration=len(aux_issue_data)
        
        if issue_type=='overconsume':
            aux_grid_w_issues['energy'].loc[aux_affected_registers]=aux_issue_data*np.random.uniform(overconsume_min, overconsume_max,issue_duration)
            aux_grid_w_issues['issue_type'].loc[aux_affected_registers] = issue_type
            aux_grid_w_issues['starting_time'].loc[aux_affected_registers] = t
            aux_grid_w_issues['rootcause'].loc[aux_affected_registers] = n
        
        elif issue_type=='underconsume':
            aux_grid_w_issues['energy'].loc[aux_affected_registers]=aux_issue_data*np.random.uniform(underconsume_min, underconsume_max,issue_duration)
            aux_grid_w_issues['issue_type'].loc[aux_affected_registers] = issue_type
            aux_grid_w_issues['starting_time'].loc[aux_affected_registers] = t
            aux_grid_w_issues['rootcause'].loc[aux_affected_registers] = n
        
        nodes_w_issues=aux_grid_w_issues.copy()
        return nodes_w_issues
    
    
    def build_nodes_w_broken_issues(aux_grid_w_issues,grid,node,t,consumption_nodes,production_nodes,issue_duration):
        ''' This method changes the all the grid data in the case that a node is completely broken so the rest of the nodes which cannot connect to any other production source is also disconnected.
        It means, this method generates spread the broken issue to all the nodes really affected as in reality.
        '''
        aux_consumption_nodes=consumption_nodes.copy()
        grid.remove_nodes_from(node)
        
        aux_affected_registers = (aux_grid_w_issues['node_id']==node) & (aux_grid_w_issues['time']>=t) & (aux_grid_w_issues['time']<t+issue_duration)
        aux_unmodified_data = aux_grid_w_issues[aux_affected_registers]['energy'].values
        
        if len(aux_unmodified_data)< issue_duration:
            issue_duration=len(aux_unmodified_data)
        
        aux_grid_w_issues['energy'].loc[aux_affected_registers] = aux_unmodified_data*np.random.uniform(0, 0.001,issue_duration)
        aux_grid_w_issues['issue_type'].loc[aux_affected_registers] = 'broken'
        aux_grid_w_issues['starting_time'].loc[aux_affected_registers] = t
        aux_grid_w_issues['rootcause'].loc[aux_affected_registers] = node
        aux_consumption_nodes=np.delete(aux_consumption_nodes,np.where(aux_consumption_nodes==node))
        for cn in aux_consumption_nodes:
            aux_node_is_connected=gridTasks.get_node_is_connected(grid,cn,t,aux_grid_w_issues,production_nodes)
            if not aux_node_is_connected:
                #aux_consumption_nodes=np.delete(aux_consumption_nodes,np.where(aux_consumption_nodes==cn)) #REVISAR
                aux_affected_registers = (aux_grid_w_issues['node_id']==cn) & (aux_grid_w_issues['time']>=t) & (aux_grid_w_issues['time']<t+issue_duration)
                
                aux_unmodified_data = aux_grid_w_issues[aux_affected_registers]['energy'].values
                
                aux_grid_w_issues['energy'].loc[aux_affected_registers] = aux_unmodified_data*np.random.uniform(0, 0.001,issue_duration)
                aux_grid_w_issues['issue_type'].loc[aux_affected_registers] = 'broken'
                aux_grid_w_issues['starting_time'].loc[aux_affected_registers] = t
                aux_grid_w_issues['rootcause'].loc[aux_affected_registers] = node
        
        nodes_w_broken_issues=aux_grid_w_issues.copy()
        return nodes_w_broken_issues
    
    def build_grid_w_issues(aux_grid,energy_data):
        ''' This is the main method which modify the data to implement some issues. '''
        aux_grid_w_issues=energy_data.copy()
        aux_grid_w_issues['issue_type']='none'
        aux_grid_w_issues['starting_time']=0
        aux_grid_w_issues['rootcause']='0'
        
        production_nodes=np.unique([n for n in aux_grid.nodes if 'productionNode' in n])
        consumption_nodes=np.unique([n for n in aux_grid.nodes if 'consumptionNode' in n])
        
        # The creation of issues was thought to do it checking every time point and then calculate by a chance of issue happening.
        # Instead, we will randomly select 1% of time points over the total time points and then check to add or not an issue.
        starting_issue_time_points=np.unique(np.random.randint(0,max_time_steps,int(max_time_steps*0.01)+1))
        consumption_nodes_chance_issue=np.unique(random.choices(consumption_nodes, k=5))
        issues_pairs=sorted(random.sample(set(itertools.product(consumption_nodes_chance_issue,starting_issue_time_points)),len(starting_issue_time_points)), key=lambda tup: tup[1])
        step_points=int(issue_duration_mean*points_per_min)
        
        for pair in issues_pairs:
            n,t=pair
            issue_duration=int(np.random.normal(step_points,int(step_points*0.3),1))
            
            if max_time_steps - t < issue_duration:
                issue_duration=step_points
            
            node_time_data=aux_grid_w_issues[(aux_grid_w_issues['node_id']==n) & (aux_grid_w_issues['time']==t)].copy()
            node_is_connected=gridTasks.get_node_is_connected(aux_grid,n,t,aux_grid_w_issues,production_nodes)
            if ((node_time_data['issue_type']=='none').values[0]) and (node_is_connected):
                issue_types=['overconsume','broken','underconsume','none']
                issue_weights=[overconsume_population,broken,underconsume_population]
                issue_weights.append(100-sum(issue_weights))
                issue_type=random.choices(issue_types, weights=issue_weights, k=1)[0]
                if issue_type!='none' and issue_type!='broken':
                    aux_grid_w_issues=gridTasks.build_nodes_w_issues(aux_grid_w_issues,issue_type,issue_duration,n,t)
                
                elif issue_type=='broken':
                    aux_grid_w_issues=gridTasks.build_nodes_w_broken_issues(aux_grid_w_issues,aux_grid,n,t,consumption_nodes,production_nodes,issue_duration)
        
        grid_w_issues=aux_grid_w_issues.copy()
        return grid_w_issues

    def do_report_effect_broken_issues(energy_data):
        ''' This method is useful for the simulation side in order to see the most severe issues and how it affected to rest of the network.
        It gives you a csv with the impact as a ratio of nodes broken and nodes which got disconnected because those were broken.'''
        report_effect_broken_issues=pd.DataFrame(columns=['starting_time','rootcauses','node_ratio_effect','affected_other_nodes'])
        broken_issues_data=energy_data[energy_data['issue_type']=='broken']
        starting_time_issues=np.unique(broken_issues_data['starting_time'].values)
        for st in starting_time_issues:
            affected_nodes=np.unique(broken_issues_data[broken_issues_data['starting_time']==st]['node_id'])
            rootcauses=np.unique(broken_issues_data[broken_issues_data['starting_time']==st]['rootcause'])
            affected_other_nodes=[i for i in affected_nodes if i not in rootcauses]
            node_ratio_effect=len(affected_other_nodes)/len(rootcauses)
            aux_report_effect_broken_issues=pd.DataFrame({'starting_time':[st],'rootcauses':[rootcauses],'node_ratio_effect':[node_ratio_effect],'affected_other_nodes':[affected_other_nodes]})
            report_effect_broken_issues=report_effect_broken_issues.append(aux_report_effect_broken_issues)
        
        report_effect_broken_issues.to_csv(filename_report_effect_broken_issues,index=False)
        
    def do_synth_data_creator_real_issues(energy_data):
        ''' This method controls the flow from initial energy data to the end'''
        # The simulated data follows two phases. First we get the ideal model data which shows the behaviour over time per node. Then we use that ideal data to real data. The model data values are on [0,1] range and real data might be on [0,max_value] per node. 
        aux_grid=gridTasks.build_grid_from_csv(relations_data)
        production_nodes=[n for n in aux_grid.nodes if 'production' in n]
        consumer_nodes=[n for n in aux_grid.nodes if 'consumption' in n]
        
        ## ADD MODEL DATA BEHAVIOUR FOR EVERY CONSUMER NODE 
        starting=time()
        energy_data = consumerNode.build_consumer_ideal_energy_data(energy_data,consumer_nodes)
        #print(f'time for creating ideal_data: {time()-starting}')
        
        starting=time()
        energy_data = gridTasks.build_grid_w_issues(aux_grid,energy_data)
        #print(f'time for creating issue_data: {time()-starting}')        
        
        # TRANSFORM FROM IDEAL BEHAVIOUR TO REAL DATA
        starting=time()
        energy_data=processingEnergyData.build_real_data(energy_data)
        #print(f'time for creating real transform data: {time()-starting}')
        
        # ADD MODEL DATA BEHAVIOUR FOR EVERY PRODUCTION NODE
        
        energy_data = producerNode.build_producer_energy_data(energy_data,production_nodes,aux_grid)
        #print(f'time for creating producer data: {time()-starting}')
        
        gridTasks.do_report_effect_broken_issues(energy_data)
        
        # SAVE DATA
        starting=time()        
        energy_data.to_csv(filename_synth_data,index=False)
        
        #print(f'time for creating save csv data: {time()-starting}')
        
        return energy_data
        
        
    def do_synth_data_creator(energy_data):
        ''' This method controls the flow from initial energy data to the end. This doesn't add real issue behaviour.'''
        # The simulated data follows two phases. First we get the ideal model data which shows the behaviour over time per node. Then we use that ideal data to real data. The model data values are on [0,1] range and real data might be on [0,max_value] per node. 
        aux_grid=gridTasks.build_grid_from_csv(relations_data)
        production_nodes=[n for n in aux_grid.nodes if 'production' in n]
        consumer_nodes=[n for n in aux_grid.nodes if 'consumption' in n]
        
        ## ADD MODEL DATA BEHAVIOUR FOR EVERY CONSUMER NODE 
        print('build_consumer_energy_data')
        energy_data = consumerNode.build_consumer_energy_data(energy_data,consumer_nodes)

        # TRANSFORM FROM IDEAL BEHAVIOUR TO REAL DATA
        print('build_real_data')
        energy_data['energy']=processingEnergyData.build_real_data(energy_data)
        
        # ADD MODEL DATA BEHAVIOUR FOR EVERY PRODUCTION NODE
        print('build_producer_energy_data')
        energy_data = producerNode.build_producer_energy_data(energy_data,production_nodes,aux_grid)
        
        # SAVE DATA
        print('to_csv')
        energy_data.to_csv(filename_synth_data,index=False) 
    
    def do_draw_grid_map():
        ''' It draws the map of the grid simulated.'''
        # Networkx GRID CONSTRUCTION
        GD=gridTasks.build_grid_from_csv(relations_data)
        
        # Networkx GRID DRAW
        pos = nx.kamada_kawai_layout(GD)  # positions for all nodes
        
        # List each type of node
        consumptionNode_IDs=property_data[property_data['node_type']=='consumptionNode']['node_id'].values
        productionNode_IDs=property_data[property_data['node_type']=='productionNode']['node_id'].values
        
        # Draw nodes and edges
        nx.draw_networkx_nodes(GD, pos, nodelist=productionNode_IDs, node_color="tab:blue",node_size=500)
        nx.draw_networkx_nodes(GD, pos, nodelist=consumptionNode_IDs, node_color="tab:red",node_size=400)
        nx.draw_networkx_edges(GD, pos,width=1.5, alpha=0.8,arrowstyle='->',arrowsize=20)
        
        labels={}
        for n in list(GD.nodes): labels[n]=n.split("_")[1] # This is the label inside the nodes
        
        nx.draw_networkx_labels(GD, pos, labels, font_size=12, font_color="whitesmoke")
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('grid_map.png',dpi=300)
        plt.close()
        
        nx.write_gexf(GD,'grid_map.gexf')
    
    def do_multiple_main_full_flow(n_grids,mode='real_issues'):
        ''' This method does to create and share the information about the folders used to leave the grid_data per grid. '''
        
        for grid_id in range(n_grids):
            starting=time()
            os.chdir(full_current_grids_data_dir)
            grid_dir='grid_id_'+grid_id.__str__()
            os.mkdir(grid_dir)
            
            os.chdir(grid_dir)          
            gridTasks.do_create_grid_csvs()
            
            starting=time()
            gridTasks.main_full_flow(mode)
            os.chdir('../../.')
            
    def build_new_grid():
        ''' Thi method creates a random network given some properties such as number of consumers and producers, the degree of connection between one and another, etc.'''
        n_consumers,n_producers=random.randint(min_consumer_nodes,max_consumer_nodes),random.randint(min_producer_nodes,max_producer_nodes)
        
        for i in range(100000):
            in_sequence_consumers=[random.randint(min_degree,max_degree) for i in range(n_consumers)]
            in_sequence_producers=[0 for i in range(n_producers)]
            in_sequence=in_sequence_consumers+in_sequence_producers
            
            out_sequence_consumers=[random.randint(min_degree,max_degree) for i in range(n_consumers)]
            out_sequence_producers=[1 for i in range(n_producers)]
            out_sequence=out_sequence_consumers+out_sequence_producers
            
            if sum(in_sequence)==sum(out_sequence):
                break
            else:
                continue
        
        new_grid=nx.generators.degree_seq.directed_havel_hakimi_graph(in_sequence,out_sequence)
        return new_grid
    
    def build_connected_grid(clean_grid):
        ''' This method ensures any consumptionNode is connected with a Production node even if it is indirectly'''
        connected_grid=clean_grid.copy()
        connected_grid_undirected=connected_grid.to_undirected().copy()
        consumers=[i for i in list(connected_grid.nodes) if 'consumptionNode' in i]
        producers=[i for i in list(connected_grid.nodes) if 'productionNode' in i]
        connected_nodes=[]
        for n_con in consumers:
            paths=[]
            for n_prod in producers:
                try:
                    aux_paths=list(nx.shortest_path(connected_grid,n_prod,n_con))
                    paths.append(aux_paths)
                except:
                    continue                
            if len(paths)==0:
                if len(connected_nodes)>2:
                    connected_grid.add_edge(random.choices(connected_nodes, k=1)[0],n_con)
                else:
                    connected_grid.add_edge(random.choices(producers, k=1)[0],n_con)
            connected_nodes.append(n_con)
        
        if len(producers)>1:
            for n_prod in producers:
                paths=[]
                other_producers=[n for n in producers if n!=n_prod]
                for n_other_prod in other_producers:
                    try:
                        aux_paths=list(nx.shortest_path(connected_grid_undirected,n_other_prod,n_prod))
                        paths.append(aux_paths)
                    except:
                        continue                
                if len(paths)==0:
                    try:
                        prod_to_connect=random.choices(other_producers, k=1)[0]
                        prod_to_connect_edges=connected_grid.edges(prod_to_connect)
                        first_nodes_to_connect=list(itertools.chain(*[[n for n in i if 'consumptionNode' in n] for i in prod_to_connect_edges]))
                        first_node_to_connect=random.choices(first_nodes_to_connect, k=1)[0]
                        second_node_edges=connected_grid.edges(first_node_to_connect)
                        second_nodes_to_connect=list(itertools.chain(*[[n for n in i if 'consumptionNode' in n] for i in second_node_edges]))
                        connected_grid.add_edge(n_prod,random.choices(second_nodes_to_connect, k=1)[0])
                    except:
                        continue
        return connected_grid
    
    def do_create_grid_csvs():
        ''' This method creates the csv properties and relations which enable to simulate energy data'''
        new_grid=gridTasks.build_new_grid()
        tagged_nodes_grid=gridTasks.build_tagged_nodes_grid(new_grid)
        clean_grid=gridTasks.build_clean_grid(tagged_nodes_grid)
        connected_grid=gridTasks.build_connected_grid(clean_grid)
        gridTasks.do_create_properties_per_node_csv(connected_grid)
        gridTasks.do_create_relationships_per_node_csv(connected_grid)
    
    def do_create_relationships_per_node_csv(new_network):
        ''' This method creates only the relation csv from each node'''
        nodes_id_origin,nodes_id_destination=[i[0] for i in new_network.edges()],[i[1] for i in new_network.edges()]
        relation_type=[]
        
        for r in new_network.edges():
            if 'consumption' in str(r[0]) and 'consumption' in str(r[1]):
                relation_type.append('distributionRelation')
            else:
                relation_type.append('productionRelation')
        
        nodes_relationships_pd=pd.DataFrame({'node_id_origin':nodes_id_origin,'node_id_destination':nodes_id_destination,'relation_type':relation_type})
        nodes_relationships_pd.to_csv('nodes_relationships.csv',index=False)
    
    def do_create_properties_per_node_csv(new_network):
        ''' This method creates only the property csv from each node'''
        consumption_nodes=[n for n in new_network.nodes() if 'consum' in n]
        production_nodes=[n for n in new_network.nodes() if 'product' in n]
        
        energy_consumption_nodes=[random.randint(min_consumer_energy,max_consumer_energy) for i in range(len(consumption_nodes))]
        energy_production_nodes=[]
        
        properties_per_node_pd=pd.DataFrame({'node_id':consumption_nodes,'node_type':['consumptionNode']*len(consumption_nodes),'max_value':energy_consumption_nodes})
        
        for pn in production_nodes:
            all_cn=np.unique([i[0] for i in list(nx.dfs_successors(new_network,pn).values())])
            aux_total_energy=sum([properties_per_node_pd[properties_per_node_pd['node_id']==i]['max_value'].values[0] for i in all_cn])
            energy_production_nodes.append(aux_total_energy)
        
        properties_per_node_pd=properties_per_node_pd.append(pd.DataFrame({'node_id':production_nodes,'node_type':['productionNode']*len(production_nodes),'max_value':energy_production_nodes}))
        properties_per_node_pd.to_csv('nodes_properties.csv',index=False)
    
    def build_unique_relation_grid(new_network):
        ''' This method ensure that each node has only a relation between them. It means that there is no path forward from node A and node B and another forward path from node B to node A.
        This methods unify duplicates.'''
        raw_relations_per_node=list(new_network.edges())
        clean_relations=[]
        
        for r in raw_relations_per_node:
            if r not in clean_relations:
                other_r=[i for i in raw_relations_per_node if i!=r]
                bool_unique_r=all([(i[0] not in r or i[1] not in r) for i in other_r])
                if bool_unique_r:
                    clean_relations.append(r)
            else:
                continue
        
        raw_duplicated_r=[r for r in raw_relations_per_node if r not in clean_relations]
        duplicated_r=[tuple(edge) for edge in set(map(frozenset, raw_duplicated_r))]
        unique_relation_grid=new_network.copy()
        print('\n \n')
        unique_relation_grid.remove_edges_from(duplicated_r)
        return unique_relation_grid
    
    def build_one_degree_connection_grid(new_network):
        ''' This method makes that every producer only has access to the grid to ONE node. That node does the role of a electrical substation.'''
        aux_grid=new_network.copy()
        all_degrees=dict(aux_grid.degree)
        big_degree_nodes=[dn for dn in all_degrees if all_degrees[dn]>2 and 'productionNode' not in dn]
        for n in big_degree_nodes:
            aux_edges=list(aux_grid.in_edges(n))+list(aux_grid.out_edges(n))
            #aux_edges=list(aux_grid.edges(n))
            if len(aux_edges)>2:
                aux_edges=[e for e in aux_edges if 'productionNode' not in e[0] and 'productionNode' not in e[1]]
                aux_degree=random.randint(min_degree,max_degree)
                deleted_edges=random.choices(aux_edges,k=len(aux_edges)-aux_degree)
                aux_grid.remove_edges_from(deleted_edges)
            else:
                continue
        one_degree_connection_grid=aux_grid
        return one_degree_connection_grid
    
    def build_clean_grid(new_network):
        ''' As we wanted a unstable grid with a low degree of connection, we added some methods for that purpose. This is the flow controller.'''
        aux_grid=new_network.copy()
        unique_relation_grid=gridTasks.build_unique_relation_grid(aux_grid)
        one_degree_connection_grid=gridTasks.build_one_degree_connection_grid(unique_relation_grid)
        clean_grid=one_degree_connection_grid
        return clean_grid
    
    def build_tagged_nodes_grid(new_network):
        ''' This method tags every node following their properties. Every node with really few connections must be producer but the number of producers is limited, so it is difficult as we cannot create type nodes from the begining.'''
        dict_nodes_degree=dict(new_network.degree())
        producer_nodes=[i for i in dict(dict_nodes_degree) if dict_nodes_degree[i]==1] # Extract the nodes with only a connection
        consumer_nodes=[i for i in dict_nodes_degree.keys() if i not in producer_nodes]
        producer_id,consumer_id=0,0
        modified_labels_dict={}
        for node_id in dict_nodes_degree.keys():
            if node_id in producer_nodes:
                producer_id=producer_id+1
                modified_labels_dict[node_id]=f'productionNode_{producer_id}'
            
            elif node_id in consumer_nodes:
                consumer_id=consumer_id+1
                modified_labels_dict[node_id]=f'consumptionNode_{consumer_id}'
        
        tagged_nodes_grid=nx.relabel.relabel_nodes(new_network,modified_labels_dict)
        return tagged_nodes_grid
    
    def do_plot_every_node(node_filter='none'):
        ''' This is an auxiliary method to create a plot for every node in every grid to show how they behaviours and how many issues there are.'''
        try:
            e_data=pd.read_csv('synth_energy_data.csv')
            aux_nodes=np.unique(e_data['node_id'])
            
            if node_filter!='none':
                aux_nodes=[n for n in aux_nodes if node_filter in n]
            
            for n in aux_nodes:
                aux_time=e_data[e_data['node_id']==n]['time']
                aux_energy=e_data[e_data['node_id']==n]['energy']
                plt.plot(aux_time,aux_energy,linewidth=0.05)
            
            plt.legend(aux_nodes)
            os.chdir(f'{root_dir}/{version_folder}')
            plt.savefig('all_nodes.png',dpi=1500)
            plt.close()
        except:
            print('You are not in the right folder! Go to one folder of grids_data')
            return 0
    
    
    def do_fast_plot_every_node(node_filter='none'):
        ''' This method is as the last one but doing it only with a grid.'''
        os.chdir(f'{root_dir}/{version_folder}/grids_data/grid_id_0')
        gridTasks.do_plot_every_node(node_filter)
    
    def main_full_flow(mode='real_issues'):
        ''' This method does every step of the flow. '''
        # INIT PROPERTY AND STRUCTURE DATA ABOUT THE GRID
        global property_data,relations_data,all_node_id,consumer_nodes,production_nodes
        
        property_data=pd.read_csv('nodes_properties.csv')
        relations_data=pd.read_csv('nodes_relationships.csv')
        
        if mode=='random':
            gridTasks.do_synth_data_creator(init_energy_data)
        
        if mode=='real_issues':
            gridTasks.do_synth_data_creator_real_issues(init_energy_data)
        
        gridTasks.do_draw_grid_map()
    
