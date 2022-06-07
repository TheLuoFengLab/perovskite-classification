import argparse
import os,sys
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import math
import copy
from pandas import read_excel
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
#import matplotlib.pyplot as plt
#import seaborn as sns

# Author: Fei Ding, feid@g.clemson.edu
# Usage: python main.py -step 0.05 -As Ba -Bs Ti,Ce,Zr,Y,Yb


class Compound:
    """Define a compound using data from a txt file."""
    def __init__(self, As, Bs, step = 0.05):
        self.rO = 1.4
        default_opt=False
        parameter_file = './data/parameters.xlsx'

        self.feature_name = [r'$O_A$', r'$O_B$', r'$r_A$', r'$r_B$', r'$d_{AO}$', r'$d_{BO}$', \
            r'$M_A$', r'$M_B$', r'$r_{A\_max}/r_O$', r'$r_{B\_max}/r_O$', r'$r_{A\_min}/r_O$', \
            r'$r_{B\_min}/r_O$', r'$r_{A\_max}/r_{A\_min}$', r'$r_{B\_max}/r_{B\_min}$',  r'$\sigma^2(A)$', \
            r'$\sigma^2(B)$', r'$r_A/r_O$', r'$r_B/r_O$', 't', r'$(O_A + O_B)/6$', r'$\tau$']
        
        self.site_A_dict, self.site_B_dict = self.read_parameters(parameter_file, default_opt)

        codes = {'negative':0, 'positive':1}
        data_set =  read_excel('./data/Fractional_Perovsktie_Oxides_Data.xlsx', sheet_name='Dataset')
        data_set.insert(0, 'label', data_set['True label'].map(codes))
        print(data_set['True label'].value_counts())
        data_set = data_set.drop(columns=['Chemical Compositions', 'True label']).values

        exp_samples =  read_excel('./data/Fractional_Perovsktie_Oxides_Data.xlsx', sheet_name='Experimental data')
        exp_samples.insert(0, 'label', exp_samples['True label'].map(codes))
        exp_label = exp_samples[['label']].values
        exp_samples = exp_samples.drop(columns=['Chemical Compositions', 'True label']).values

        gen_data = self.generate_compunds(As, Bs, step)
        gen_samples = self.combine_samples(gen_data)
        self.gen_samples_all = copy.deepcopy(gen_samples[:, 1:])
        self.gen_samples_all = np.round(self.gen_samples_all, 6)

        #self.plot_correlations(data_set[:, 1:], data_set[:, 0])

        ##############################################
        # choose top 5 features
        remove_list = [0, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20]  #top5
        rm_list = [i + 1 for i in remove_list] #y is at the beginning
        idxs = [i for i in range(0, 22) if i not in rm_list]
        data_set = data_set[:, idxs]
        exp_samples = exp_samples[:, idxs]
        gen_samples = gen_samples[:, idxs]
        ##############################################        
        
        #self.compounds = np.concatenate((negative_samples, positive_samples), axis=0)
        self.compounds = data_set
        self.experimental_samples = exp_samples[:, 1:] 
        self.experimental_labels = np.squeeze(exp_label)
        self.gen_samples = gen_samples[:, 1:]
        self.gen_data = np.array(gen_data)
                
        # construct data sets
        self.X = []
        self.y = []
        for (index, values) in enumerate(self.compounds.tolist()):
            self.X.append(values[1:])
            self.y.append(int(values[0])) 
        
    def generate_compunds(self, As, Bs, step):
        As_list = As.split(',')
        Bs_list = Bs.split(',')
        numbers = np.arange(step, 1+step, step)
        
        def subset_sum(cnt, partial, permutations):
            s = sum(partial)
            # check if the partial sum is equals to target
            if math.isclose(s, 1.0) and cnt == 0:
                permutations.append(",".join([str(round(i, 2)) for i in partial]))
            if s >= 1.0 + step or cnt < 0:
                return  # if we reach the number why bother to continue
            for num in numbers:
                subset_sum(cnt-1, partial + [num], permutations)
        
        As_perm, Bs_perm = [], []
        subset_sum(len(As_list), [], As_perm)
        subset_sum(len(Bs_list), [], Bs_perm)
      
        As_fraction = [As + "," + item for item in As_perm]
        Bs_fraction = [Bs + "," + item for item in Bs_perm]
        permutations = [[a,b] for a in As_fraction for b in Bs_fraction]
        return permutations
    
    def read_parameters(self, parameter_file, default_opt=True):
        site_A_1 = read_excel(parameter_file, sheet_name='A site-CN=12').iloc[:42]
        site_A_1 = site_A_1[['Element Symbol','Oxidation state','Ionic radius-A site(CN=12)', \
                             'dAO(CN=12)', 'Medeleev number']]
        #site_A_1['Medeleev number'] /= 86  # 86 is maximum, normalized
        site_B_1 = read_excel(parameter_file, sheet_name='B site-CN=6').iloc[:42]
        site_B_1 = site_B_1[['Element Symbol','Oxidation state','Ionic radius-B site(CN=6)', \
                             'dBO', 'Medeleev number']]
        #site_B_1['Medeleev number'] /= 86  # 86 is maximum, normalized
        
        site_A_2 = read_excel(parameter_file, sheet_name='A site-CN=12 from ouyang').iloc[:42]
        site_A_2 = site_A_2[['Element Symbol','Oxidation state','Ionic radius-A site(CN=12)']]
        site_B_2 = read_excel(parameter_file, sheet_name='B site-CN=6 from ouyang').iloc[:42]
        site_B_2 = site_B_2[['Element Symbol','Oxidation state','Ionic radius-B site(CN=6)']]     
        # site_A_1 and site_A_2 share same dAO(CN=12), Medeleev number
        site_A_2 = pd.concat([site_A_2, site_A_1[['dAO(CN=12)', 'Medeleev number']]],axis=1)
        # site_B_1 and site_B_2 share same dBO, Medeleev number
        site_B_2 = pd.concat([site_B_2, site_B_1[['dBO', 'Medeleev number']]],axis=1)   
        
        if default_opt:
            site_A_dict = site_A_1.set_index('Element Symbol').T.to_dict('list')
            site_B_dict = site_B_1.set_index('Element Symbol').T.to_dict('list')
        else:
            site_A_dict = site_A_2.set_index('Element Symbol').T.to_dict('list')
            site_B_dict = site_B_2.set_index('Element Symbol').T.to_dict('list')  
        
        return site_A_dict, site_B_dict
            
    def combine_samples(self, site_list, label = 2):
        sample_list = []
        for i, (A, B) in enumerate(site_list):
            A_list = A.split(',')
            B_list = B.split(',')
            A_list = ' '.join(A_list).split()
            B_list = ' '.join(B_list).split()  

            A_pair = list(zip(A_list[0:len(A_list)//2], A_list[-len(A_list)//2:]))
            B_pair = list(zip(B_list[0:len(B_list)//2], B_list[-len(B_list)//2:]))
            A_pair = [ (ele.replace(' ', ''), frac) for (ele, frac) in A_pair]
            B_pair = [ (ele.replace(' ', ''), frac) for (ele, frac) in B_pair]
            A_pair = [ (ele.replace('\t', ''), frac) for (ele, frac) in A_pair]
            B_pair = [ (ele.replace('\t', ''), frac) for (ele, frac) in B_pair]       
            
            # Oxidation state_A, rA, dAO, MA, rA_max, rA_min, rA_max_min, rA_variance
            A_features = self.calculate_features(A_pair, True)
            # Oxidation state_B, rB, dBO, MB, rB_max, rB_min, rB_max_min, rB_variance
            B_features = self.calculate_features(B_pair, False)
            
            rA_rO = round(A_features[1]/self.rO, 4)
            rB_rO = round(B_features[1]/self.rO, 4)
            t = round((A_features[1]+self.rO)/(math.sqrt(2)*(B_features[1]+self.rO)),4)

            combine_features = [[a, b] for (a,b) in zip(A_features, B_features)] # 8+8
            
            oxidation_state = combine_features[0]
            n_A = A_features[0]       
            t_new = self.rO / B_features[1] - n_A * (n_A - (A_features[1] / B_features[1])/(np.log(A_features[1] / B_features[1])))

            combine_features.append([rA_rO, rB_rO, t, np.sum(oxidation_state)/6, t_new]) # 8+8+5
            combine_features.insert(0, [label])
            combine_features = sum(combine_features, [])                       
            sample_list.append(combine_features)

        sample_list = np.array(sample_list)
        sample_list = sample_list[~np.isnan(sample_list).any(axis=1)]        
        return sample_list

    def calculate_features(self, pair_list, is_A=True):
        element_list = [e for (e, f) in pair_list]
        fraction_list =  [float(f) for (e, f) in pair_list]
        
        if is_A:
            # 'Oxidation state','Ionic radius-A site(CN=12)', 'dAO(CN=12)', 'Medeleev number'
            feature_matrix = [self.site_A_dict[e] for e in element_list]
        else:
            # 'Oxidation state','Ionic radius-B site(CN=6)', 'dBO', 'Medeleev number'
            feature_matrix = [self.site_B_dict[e] for e in element_list]
        feature_matrix = np.array(feature_matrix)
              
        radius_list = feature_matrix[:, 1]
        radius_max = np.max(radius_list)/self.rO
        radius_min = np.min(radius_list)/self.rO
        radius_max_min = np.max(radius_list)/np.min(radius_list)
        
        features = np.dot(np.array(fraction_list), feature_matrix)
        
        r_i_2 = feature_matrix[:,1]**2
        variance = [a*b for a,b in zip(fraction_list, r_i_2)]  
        variance = sum(variance) - features[1]**2
        features = np.concatenate((features, np.array([radius_max, radius_min, radius_max_min, variance])))
        return features

    def plot_correlations(self, train_X, train_y):
        train_data = np.concatenate((train_X, np.expand_dims(train_y, axis=1)), axis=1)
        train_pd = pd.DataFrame(train_data)
        train_pd.columns = self.feature_name + ['y']
        # correlations
        corrMatrix = train_pd.corr()
        #Using Pearson Correlation
        #sns.set_style("whitegrid")
        #sns.set(font_scale=2) 
        plt.figure(figsize=(12,10))
        sns.heatmap(corrMatrix, annot=True, cmap=plt.cm.Reds, fmt='.2f')
        plt.savefig("./data/pearson_correlation_matrix.png", dpi=300)        

parser = argparse.ArgumentParser(description='Compound data',
        formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-step', help='The step of fraction', type=float, default=0.05)
parser.add_argument('-As', required=True, help='A1 site and A2 site, such as: Sr,La', type=str)
parser.add_argument('-Bs', required=True, help='B1 site and B2 site, such as: Ti,Fe', type=str)

if __name__ == '__main__':
    args = parser.parse_args()
    compound = Compound(args.As, args.Bs, args.step)
    
    GBDT = GradientBoostingClassifier(n_estimators=200, min_samples_split=4, min_samples_leaf=3, max_depth=4, random_state=1)
    train_X = np.array(compound.X).astype(float)
    train_y = np.array(compound.y).astype(int)
    GBDT.fit(train_X, train_y)
    y_predict = GBDT.predict(compound.experimental_samples)
    y_true = compound.experimental_labels

    print()
    print('experimental samples: ', compound.experimental_samples.shape[0]) 
    print('True label: ')
    print(compound.experimental_labels)
    print('Predict label: ')
    print(y_predict) 
    assert len(y_true) == len(y_predict), "Oh no! This assertion failed!"   
    print('correct: ', len([i for i, j in zip(y_predict, y_true) if i == j]), 'total: ', len(y_true))

    gen_prediction = GBDT.predict(compound.gen_samples)
    gen_all = np.concatenate([compound.gen_data, np.expand_dims(gen_prediction, axis=1), compound.gen_samples_all], 
                             axis=1)
    ## convert your array into a dataframe
    df = pd.DataFrame (gen_all, columns = ['As', 'Bs', 'Prediction'] + compound.feature_name)
    print()
    print('generate_compunds:')
    print(df)
    ## save to xlsx file
    df.to_excel(args.As + '_' + args.Bs + '_' + str(args.step) + '.xlsx', index=False)
