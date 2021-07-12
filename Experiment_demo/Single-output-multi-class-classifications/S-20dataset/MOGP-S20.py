
            ## This is the main function for runing modul ##

### import Necessary packages
import sys
sys.path.append('YourPath')
#e.g., sys.path.append('./MOGP-AR')
import numpy as np
import tensorflow as tf
import time
import warnings
import pandas as pd
import argparse
from sklearn.model_selection import StratifiedKFold ## for c-v
warnings.filterwarnings('ignore')  # ignore DeprecationWarnings from tensorflow

import gpflow
from gpflow.ci_utils import ci_niter ## for the number of training
from MOGP_convolutionalKernel.utils import run_adam, Performance_measure, Performance_list ## Import optimization
from MOGP_convolutionalKernel.Building_Models import Build_GP_AR_RBF, Build_GPC_RBF,Build_MOGP_AR_RBF, Build_MOGP_RBF



# reproducibility
import random
random.seed(3)
np.random.seed(1)
tf.random.set_seed(24)

# Performance metric
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.datasets import make_classification as mc
import math
## Cross-validation
from sklearn.model_selection import train_test_split

                                                        #######################
                                                        ###    Data set     ###
                                                        #######################

def Dataset(Data_name):
    '''
    This is used for importing the dataset and change it into our required format.
    Note: we only choose one dataset in one time

    Args:
        Data_name: the Name of dataset

    Returns:
        X: Input
        Y: Output
    '''
    if Data_name == 'S-20.data':
        X1, Y1 = mc(n_samples=2000, n_classes=20, n_features=5, n_redundant=0, n_informative=5, n_clusters_per_class=1,
                    random_state=1)
        X = X1.copy()
        Y = Y1[:, None].copy()
    elif Data_name == 'CNAE-9.data':
        Importdata = pd.read_table(args.pathin + 'CNAE-9.data', sep=',', header=None)
        npdata = np.array(Importdata)
        Data = npdata.astype('float64')
        X = Data[:, 1:]
        Y = Data[:, 0][:, None].astype('int64') - 1
    elif Data_name == 'balance-scale.data':
        Importdata = pd.read_table(args.pathin +'balance-scale.data', sep=',', header=None)
        npdata = np.array(Importdata)
        npdata = np.where(npdata == 'L', 0, npdata)
        npdata = np.where(npdata == 'R', 1, npdata)
        npdata = np.where(npdata == 'B', 2, npdata)
        Data = npdata.astype('float64')
        X = Data[:, 1:]
        Y = Data[:, 0][:, None].astype('int64')
    elif Data_name == 'Mediamill.data':
        Data = np.loadtxt(args.pathin+'MediaMill/Mediamill_data.txt')
        X = Data[:, :Data.shape[1] - 1]
        Y = Data[:, Data.shape[1] - 1][:, None].astype('int64')
    else:
        print('Please use existed dataset')
        X = None
        Y = None
    return X, Y
                                                        ####################
                                                        #### Run Model #####
                                                        ####################

def Run_model(X,Y):
    '''
    We run all the models together
    Args:
        X: the training data for input
        Y: the training data for output
        DataName: the Name of dataset
    Returns:
        MOGP_AR: The MOGP_AR model
        data_MOGP_AR: All the training data for MOGP_AR
        N: The number of training data
    '''
    sfolder = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

                            ######################################
                            #####  Performacne Metric Set up #####
                            ######################################
    ##### MOGP Performacne Metric#####
    ## Performance Metric
    time_training_MOGP,Prediction_error_MOGP,Precision_weighted_MOGP,\
    Recall_weighted_MOGP,F1_weighted_MOGP=Performance_list()

    MOGP_U = []
    MOGP_whole_time = []

    for train_index, test_index in sfolder.split(X, Y):
        Xtrain, Xtest = X[train_index], X[test_index]
        Ytrain, Ytest = Y[train_index], Y[test_index]

        ### minibatch_size
        minibatch_size = args.Size_Minibatch

        ### maxiter interation
        maxiter = ci_niter(args.Number_train)


        ###  Build our model and run times and optimization
        ### MOGP
        a_MOGP_whole_time = time.time()
        X_train_cv, X_val, Y_train_cv, Y_val = train_test_split(Xtrain, Ytrain, test_size=0.2, random_state=0)
        CV_error = []
        for N_U in args.Num_U_CV:
            MOGP_cv, data_MOGP_cv, N_MOGP_cv = Build_MOGP_RBF(X_train_cv, Y_train_cv,Num_U=N_U, Num_classes=args.Num_classes)
            logf_mogp_cv = run_adam(MOGP_cv, maxiter, data_MOGP_cv, N_MOGP_cv, minibatch_size)
            X_pre_MOGP_cv = np.hstack((X_val, np.zeros_like(X_val[:, 0][:, None])))
            mu_y_cv, _ = MOGP_cv.predict_y_categorical(X_pre_MOGP_cv)
            if args.cv_measure == 'Accuracy':
                Test_error_cv = np.mean(np.argmax(mu_y_cv[0], axis=1) == Y_val.squeeze())
            elif args.cv_measure == 'Recall':
                Test_error_cv = recall_score(Y_val, np.argmax(mu_y_cv[0], axis=1)[:, None], average='weighted')
            else:
                print('Please print the correct the performance measure')
                break
            CV_error.append(Test_error_cv)
        Optimal_Num_u_MOGP = args.Num_U_CV[CV_error.index(max(CV_error))]
        MOGP, data_MOGP, N_MOGP = Build_MOGP_RBF(Xtrain, Ytrain, Num_U=Optimal_Num_u_MOGP, Num_classes=args.Num_classes)
        a_MOGP = time.time()
        logf_mogp = run_adam(MOGP, maxiter, data_MOGP, N_MOGP, minibatch_size)
        b_MOGP = time.time()
        print('runing time:', b_MOGP - a_MOGP)
        time_training_MOGP.append(b_MOGP - a_MOGP)
        MOGP_whole_time.append(b_MOGP - a_MOGP_whole_time)
        # MOGP
        X_pre_MOGP = np.hstack((Xtest, np.zeros_like(Xtest[:, 0][:, None])))
        mu_y_MOGP, _ = MOGP.predict_y_categorical(X_pre_MOGP)
        MOGP_U.append(Optimal_Num_u_MOGP)

        ####################################
        ####### Performance Measure ########
        ####################################
        ## MOGP ##
        Test_error_MOGP, P_w_MOGP, R_w_MOGP,\
        F1_w_MOGP = Performance_measure(mu=np.argmax(mu_y_MOGP[0], axis=1), Yest=Ytest)

        ### Prediciton error
        # MOGP
        Prediction_error_MOGP.append(Test_error_MOGP)

        ### Precision
        # MOGP
        Precision_weighted_MOGP.append(P_w_MOGP)

        ### Recall
        # MOGP
        Recall_weighted_MOGP.append(R_w_MOGP)

        ### F1
        # MOGP
        F1_weighted_MOGP.append(F1_w_MOGP)


    ## Include MOGP result
    Result = pd.DataFrame({'Model': [ 'MOGP'],
                           '0': [MOGP_U],
                           'Training Time mean': [np.mean(time_training_MOGP)],
                           'Training Time std': [np.std(time_training_MOGP)],
                           '1': [ np.mean(MOGP_whole_time)],
                           'Mean Prediciton error': [np.mean(Prediction_error_MOGP)],
                           '2': [np.std(MOGP_whole_time)],
                           'Std prediction error': [np.std(Prediction_error_MOGP)],
                           '3': ['&'],
                           'Mean Precision_weighted': [np.mean(Precision_weighted_MOGP)],
                           '4': ['&'],
                           'Std Precision_weighted': [np.std(Precision_weighted_MOGP)],
                           '5': ['&'],
                           'Mean Recall_weighted': [np.mean(Recall_weighted_MOGP)],
                           '6': ['&'],
                           'Std Recall_weighted': [np.std(Recall_weighted_MOGP)],
                           '7': ['&'],
                           'Mean F1_weighted': [np.mean(F1_weighted_MOGP)],
                           '8': ['&'],
                           'Std F1_weighted': [np.std(F1_weighted_MOGP)]
                           })

    return Result

if __name__ == '__main__':
    print("starting to run")
    ## Inputs for the main function
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--MOGP_AR_model', type=str, default='MOGP_AR(1)',help='The name of MOGP_AR')
    parser.add_argument('--Name_report', type=str, default='Result_Balance-Accuracy.csv',help='The Name of output report')
    parser.add_argument('--Name_data',type= str,default='balance-scale.data',help='The name of the dataset')
    # parser.add_argument('--Name_data', type=str, default='balance-scale.data')
    parser.add_argument('--Size_Minibatch', type=int, default=200,help='The number of minibatch')
    parser.add_argument('--Number_train', type=int, default=30, help='The number of training times')
    parser.add_argument('--Num_classes', type=int, default=3, help='The number of class for single output')
    parser.add_argument('--cv_measure',type= str,default='Accuracy')
    ## Cross-validation parameters
    parser.add_argument('--Num_U_CV', type = int, nargs='+', default=[1,2,3], help='The number of U for Cross validation')
    ## Path in and out
    ## Path in and out
    parser.add_argument('--pathin', type=str, default='YourPath/')
    ## e.g., parser.add_argument('--pathin', type= str, default='../MOGP_AR_Dataset/')
    parser.add_argument('--pathout', type= str, default='YourPath/')
    ## e.g., parser.add_argument('--pathout', type= str, default='Result/')
    args = parser.parse_args()
    ## Dataset
    X,Y = Dataset(Data_name=args.Name_data)
    ## Run the model
    Result = Run_model(X,Y)
    ## Save the Result
    Result.to_csv(args.pathout + args.Name_report, index=False, sep=',',
                  float_format='%.4f')





