
            ## This is the main function for runing modul ##

### import Necessary packages
import sys
sys.path.append('YourPath')
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
from sklearn.datasets import make_classification as mc
import math
## Cross-validation
from sklearn.model_selection import train_test_split

                                                        ####################
                                                        #### Run Model #####
                                                        ####################

def Run_model(Data,train_index,test_index):
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
                            ######################################
                            #####  Performacne Metric Set up #####
                            ######################################

    ##### GPAR Performacne Metric#####
    ## Performance Metric
    time_training_GPAR,Prediction_error_GPAR,Precision_weighted_GPAR,\
    Recall_weighted_GPAR,F1_weighted_GPAR=Performance_list()

    ##### GPC Performacne Metric#####
    ## Performance Metric
    time_training_GPC,Prediction_error_GPC,Precision_weighted_GPC,\
    Recall_weighted_GPC,F1_weighted_GPC=Performance_list()

    #### MOGP_AR Performacne Metric#####
    # Performance Metric
    time_training_MOGP_AR, Prediction_error_MOGP_AR, Precision_weighted_MOGP_AR, \
    Recall_weighted_MOGP_AR, F1_weighted_MOGP_AR = Performance_list()

    MOGP_AR_U = []
    MOGP_AR_whole_time =[]

    for i in range(3):
        ## since the index from 0 in python
        train_index_0 = train_index[:, i].astype('int64').copy() - 1
        test_index_0 = test_index[:, i].astype('int64').copy() - 1

        ## the train data
        Xtrain = Data[train_index_0, :Data.shape[1] - 1]
        Ytrain = Data[train_index_0, Data.shape[1] - 1][:, None].astype('int64')

        ## the test data
        Xtest = Data[test_index_0, :Data.shape[1] - 1]
        Ytest = Data[test_index_0, Data.shape[1] - 1][:, None].astype('int64')

        ### minibatch_size
        minibatch_size = args.Size_Minibatch

        ### maxiter interation
        maxiter = ci_niter(args.Number_train)


        ###  Build our model and run times and optimization
        ### GPAR
        a_GPAR = time.time()
        GPAR, data_GPAR, N_GPAR = Build_GP_AR_RBF(Xtrain, Ytrain, Num_classes=args.Num_classes)
        logf_GPAR = run_adam(GPAR, maxiter, data_GPAR, N_GPAR, minibatch_size)
        b_GPAR = time.time()
        print('runing time:', b_GPAR - a_GPAR)
        time_training_GPAR.append(b_GPAR - a_GPAR)
        # Prediction
        mu_y_GPAR, _ = GPAR.predict_y(Xtest)
        mu_y_GPAR = mu_y_GPAR.numpy()


        ### GPC
        a_GPC = time.time()
        GPC, data_GPC, N_GPC = Build_GPC_RBF(Xtrain, Ytrain, Num_classes=args.Num_classes)
        logf_GPC = run_adam(GPC, maxiter, data_GPC, N_GPC, minibatch_size)
        b_GPC = time.time()
        print('runing time:', b_GPC - a_GPC)
        time_training_GPC.append(b_GPC - a_GPC)
        # Prediction
        mu_y_GPC, _ = GPC.predict_y(Xtest)
        mu_y_GPC = mu_y_GPC.numpy()


        X_train_cv, X_val, Y_train_cv, Y_val = train_test_split(Xtrain, Ytrain, test_size=0.2, random_state=0)

        ### MOGP_AR
        a_MOGP_AR_whole_time = time.time()
        CV_error_AR = []
        for N_U in args.Num_U_CV:
            MOGP_AR_cv, data_MOGP_AR_cv, N_MOGP_AR_cv = Build_MOGP_AR_RBF(X_train_cv, Y_train_cv, Num_U=N_U, Num_sub=args.ns,
                                                                            Whole_f=args.whole_latent_f,
                                                                            num_subsample=args.num_sample,
                                                                            num_latent_f=args.num_latent_f_list,
                                                                            num_output=args.num_output,
                                                                            Size_minibatch=minibatch_size)
            logf_mogp_ar_cv = run_adam(MOGP_AR_cv, maxiter, data_MOGP_AR_cv, N_MOGP_AR_cv, minibatch_size)
            X_pre_MOGP_AR_cv = np.hstack((X_val, np.zeros_like(X_val[:, 0][:, None])))
            mu_y_MOGP_AR_cv, _ = MOGP_AR_cv.predict_y_categorical(X_pre_MOGP_AR_cv)
            if args.cv_measure == 'Accuracy':
                Test_error_AR_cv = np.mean(np.argmax(mu_y_MOGP_AR_cv[0], axis=1) == Y_val.squeeze())
            elif args.cv_measure == 'Recall':
                Test_error_AR_cv = recall_score(Y_val, np.argmax(mu_y_MOGP_AR_cv[0], axis=1)[:, None], average='weighted')
            else:
                print('Please print the correct the performance measure')
                break
            CV_error_AR.append(Test_error_AR_cv)
        Optimal_Num_u_MOGP_AR = args.Num_U_CV[CV_error_AR.index(max(CV_error_AR))]
        # Optimal_Num_u_MOGP_AR =3
        MOGP_AR, data_MOGP_AR, N_MOGP_AR = Build_MOGP_AR_RBF(Xtrain, Ytrain, Num_U=Optimal_Num_u_MOGP_AR, Num_sub=args.ns,
                                                                           Whole_f = args.whole_latent_f,
                                                                           num_subsample = args.num_sample,
                                                                           num_latent_f = args.num_latent_f_list,
                                                                            num_output = args.num_output,
                                                                            Size_minibatch=minibatch_size)
        a_MOGP_AR = time.time()
        logf_mogp_AR = run_adam(MOGP_AR, maxiter, data_MOGP_AR, N_MOGP_AR, minibatch_size)
        b_MOGP_AR = time.time()
        print('runing time:', b_MOGP_AR - a_MOGP_AR)
        time_training_MOGP_AR.append(b_MOGP_AR - a_MOGP_AR)
        MOGP_AR_whole_time.append(b_MOGP_AR - a_MOGP_AR_whole_time)
        ## Prediction
        X_pre_MOGP_AR = np.hstack((Xtest, np.zeros_like(Xtest[:, 0][:, None])))
        mu_y_MOGP_AR, _ = MOGP_AR.predict_y_categorical(X_pre_MOGP_AR)
        MOGP_AR_U.append(Optimal_Num_u_MOGP_AR)


        ####################################
        ####### Performance Measure ########
        ####################################
        ## GPAR
        Test_error_GPAR, P_w_GPAR, R_w_GPAR, \
        F1_w_GPAR = Performance_measure(mu=np.argmax(mu_y_GPAR, axis=1), Yest=Ytest)
        ## GPC ##
        Test_error_GPC, P_w_GPC, R_w_GPC,\
        F1_w_GPC = Performance_measure(mu=np.argmax(mu_y_GPC, axis=1), Yest=Ytest)

        ## MOGP_AR ##
        Test_error_MOGP_AR, P_w_MOGP_AR, R_w_MOGP_AR,\
        F1_w_MOGP_AR = Performance_measure(mu=np.argmax(mu_y_MOGP_AR[0], axis=1), Yest=Ytest)

        ### Prediciton error
        # GPAR
        Prediction_error_GPAR.append(Test_error_GPAR)
        # GPC
        Prediction_error_GPC.append(Test_error_GPC)
        # MOGP_AR
        Prediction_error_MOGP_AR.append(Test_error_MOGP_AR)
        ### Precision
        # GPAR
        Precision_weighted_GPAR.append(P_w_GPAR)
        # GPC
        Precision_weighted_GPC.append(P_w_GPC)
        # MOGP_AR
        Precision_weighted_MOGP_AR.append(P_w_MOGP_AR)
        ### Recall
        # GPAR
        Recall_weighted_GPAR.append(R_w_GPAR)
        # GPC
        Recall_weighted_GPC.append(R_w_GPC)
        # MOGP_AR
        Recall_weighted_MOGP_AR.append(R_w_MOGP_AR)
        ### F1
        # GPAR
        F1_weighted_GPAR.append(F1_w_GPAR)
        # GPC
        F1_weighted_GPC.append(F1_w_GPC)
        # MOGP_AR
        F1_weighted_MOGP_AR.append(F1_w_MOGP_AR)


    ## Include GPC, GP_AR, MOGP result
    Result = pd.DataFrame({'Model': ['GPAR', 'GPC'],
                           '0': ['&', '&'],
                           'Training Time mean': [np.mean(time_training_GPAR), np.mean(time_training_GPC)],
                           'Training Time std': [np.std(time_training_GPAR), np.std(time_training_GPC)],
                           '1': ['&', '&'],
                           'Mean Prediciton error': [np.mean(Prediction_error_GPAR), np.mean(Prediction_error_GPC)],
                           '2': ['&', '&'],
                           'Std prediction error': [np.std(Prediction_error_GPAR), np.std(Prediction_error_GPC)],
                           '3': ['&', '&'],
                           'Mean Precision_weighted': [np.mean(Precision_weighted_GPAR),
                                                       np.mean(Precision_weighted_GPC)],
                           '4': ['&', '&'],
                           'Std Precision_weighted': [np.std(Precision_weighted_GPAR), np.std(Precision_weighted_GPC)],
                           '5': ['&', '&'],
                           'Mean Recall_weighted': [np.mean(Recall_weighted_GPAR), np.mean(Recall_weighted_GPC)],
                           '6': ['&', '&'],
                           'Std Recall_weighted': [np.std(Recall_weighted_GPAR), np.std(Recall_weighted_GPC)],
                           '7': ['&', '&'],
                           'Mean F1_weighted': [np.mean(F1_weighted_GPAR), np.mean(F1_weighted_GPC)],
                           '8': ['&', '&'],
                           'Std F1_weighted': [np.std(F1_weighted_GPAR), np.std(F1_weighted_GPC)]
                           })

    ## Add MOGP_AR result in to result
    Result.loc[2] = [args.MOGP_AR_model,MOGP_AR_U,np.mean(time_training_MOGP_AR),np.std(time_training_MOGP_AR),np.mean(MOGP_AR_whole_time),
                            np.mean(Prediction_error_MOGP_AR),np.std(MOGP_AR_whole_time),np.std(Prediction_error_MOGP_AR),
                            '&',np.mean(Precision_weighted_MOGP_AR),'&',np.std(Precision_weighted_MOGP_AR), '&',np.mean(Recall_weighted_MOGP_AR),'&',
                            np.std(Recall_weighted_MOGP_AR), '&',np.mean(F1_weighted_MOGP_AR), '&',np.std(F1_weighted_MOGP_AR)
                        ]
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
    ## Model parameters
    parser.add_argument('--ns',type=int, nargs='+', default=[1], help='The number of subsampling for each output')
    parser.add_argument('--num_sample', type=int, nargs='+', default=[2],help='The number of sample in each output')
    parser.add_argument('--num_latent_f_list',type=int,nargs='+', default=[3], help='The number of latent parameter functions f in each output')
    parser.add_argument('--num_output', type=int, default=1, help='The number of outputs')
    parser.add_argument('--whole_latent_f', type=int, default=3, help='The number of all the latent parameter functions f')
    ## Cross-validation parameters
    parser.add_argument('--Num_U_CV', type = int, nargs='+', default=[1,2,3], help='The number of U for Cross validation')
    ## Path in and out
    parser.add_argument('--pathin', type= str, default='YourPath')
    parser.add_argument('--pathout', type= str, default='YourPath')
    args = parser.parse_args()
    ## Dataset
    ## Dataset
    Data = np.loadtxt('/YourPath/Bibtex/Bibtex_data.txt')
    train_index = np.loadtxt('/YourPath/Bibtex/bibtex_trSplit.txt')
    test_index = np.loadtxt('/YourPath/Bibtex/bibtex_tstSplit.txt')
    ## Run the model
    Result = Run_model(Data,train_index,test_index)
    ## Save the Result
    Result.to_csv(args.pathout + args.Name_report, index=False, sep=',',
                  float_format='%.4f')





