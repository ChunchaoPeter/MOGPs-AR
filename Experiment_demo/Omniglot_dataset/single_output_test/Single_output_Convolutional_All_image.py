
            ## This is the main function for runing modul ##

### import Necessary packages
import sys
sys.path.append('YourPath/MOGP-AR')
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
from MOGP_convolutionalKernel.Building_Models import Build_GP_AR_Conv, Build_GPC_Conv
from MOGP_convolutionalKernel.Load_data import Load_Sinlge_Omniglot_Dataset
# reproducibility
import random


random.seed(30)
np.random.seed(10)
tf.random.set_seed(24)



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

    ##### GPAR Performacne Metric#####
    ## Performance Metric
    time_training_GPAR,Prediction_error_GPAR,Precision_weighted_GPAR,\
    Recall_weighted_GPAR,F1_weighted_GPAR=Performance_list()

    ##### GPC Performacne Metric#####
    ## Performance Metric
    time_training_GPC,Prediction_error_GPC,Precision_weighted_GPC,\
    Recall_weighted_GPC,F1_weighted_GPC=Performance_list()

    for train_index, test_index in sfolder.split(X, Y):
        Xtrain, Xtest = X[train_index], X[test_index]
        Ytrain, Ytest = Y[train_index], Y[test_index]

        ### minibatch_size
        minibatch_size = args.Size_Minibatch

        ### maxiter interation
        maxiter = ci_niter(args.Number_train)


        ###  Build our model and run times and optimization
        ### GPAR
        a_GPAR = time.time()
        GPAR, data_GPAR, N_GPAR = Build_GP_AR_Conv(Xtrain, Ytrain, Num_classes=args.Num_classes,patch1 = args.patch1,
                                                   patch2 = args.patch2,Maximum_induc_patch= args.Maximum_induc_patch)
        logf_GPAR = run_adam(GPAR, maxiter, data_GPAR, N_GPAR, minibatch_size)
        b_GPAR = time.time()
        print('runing time:', b_GPAR - a_GPAR)
        time_training_GPAR.append(b_GPAR - a_GPAR)
        # Prediction
        mu_y_GPAR, _ = GPAR.predict_y(Xtest)
        mu_y_GPAR = mu_y_GPAR.numpy()

        ### GPC
        a_GPC = time.time()
        GPC, data_GPC, N_GPC = Build_GPC_Conv(Xtrain, Ytrain, Num_classes=args.Num_classes, patch1 = args.patch1,
                                              patch2 = args.patch2,Maximum_induc_patch= args.Maximum_induc_patch)
        logf_GPC = run_adam(GPC, maxiter, data_GPC, N_GPC, minibatch_size)
        b_GPC = time.time()
        print('runing time:', b_GPC - a_GPC)
        time_training_GPC.append(b_GPC - a_GPC)
        # Prediction
        mu_y_GPC, _ = GPC.predict_y(Xtest)
        mu_y_GPC = mu_y_GPC.numpy()


        ####################################
        ####### Performance Measure ########
        ####################################
        ## GPAR
        Test_error_GPAR, P_w_GPAR, R_w_GPAR, \
        F1_w_GPAR = Performance_measure(mu=np.argmax(mu_y_GPAR, axis=1), Yest=Ytest)
        ## GPC ##
        Test_error_GPC, P_w_GPC, R_w_GPC,\
        F1_w_GPC = Performance_measure(mu=np.argmax(mu_y_GPC, axis=1), Yest=Ytest)

        ### Prediciton error
        # GPAR
        Prediction_error_GPAR.append(Test_error_GPAR)
        # GPC
        Prediction_error_GPC.append(Test_error_GPC)

        ### Precision
        # GPAR
        Precision_weighted_GPAR.append(P_w_GPAR)
        # GPC
        Precision_weighted_GPC.append(P_w_GPC)

        ### Recall
        # GPAR
        Recall_weighted_GPAR.append(R_w_GPAR)
        # GPC
        Recall_weighted_GPC.append(R_w_GPC)

        ### F1
        # GPAR
        F1_weighted_GPAR.append(F1_w_GPAR)
        # GPC
        F1_weighted_GPC.append(F1_w_GPC)


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
                           'Mean Precision_weighted': [np.mean(Precision_weighted_GPAR),np.mean(Precision_weighted_GPC)],
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

    return Result


if __name__ == '__main__':
    print("starting to run")
    ## Inputs for the main function
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--Name_report', type=str, default='Result_Balance-Accuracy.csv',help='The Name of output report')
    parser.add_argument('--Size_Minibatch', type=int, default=200,help='The number of minibatch')
    parser.add_argument('--Number_train', type=int, default=30, help='The number of training times')
    parser.add_argument('--Num_classes', type=int, default=7, help='The number of class for single output')
    ## Convolutional kernel parameters
    parser.add_argument('--patch1', type=int, default=19, help='The length size of patch')
    parser.add_argument('--patch2', type=int, default=20, help='The width size of patch')
    parser.add_argument('--Maximum_induc_patch', type=int, default=7, help='The maximum of inducing patches')
    ## Path in and out
    parser.add_argument('--pathin', type=str, default='YourPath/MOGP_AR_Dataset/')
    ## e.g.,     parser.add_argument('--pathin', type= str, default='./images/images_background/Greek/')
    parser.add_argument('--pathout', type= str, default='YourPath/')
    ## e.g., parser.add_argument('--pathout', type= str, default='Result/')
    args = parser.parse_args()
    ## Dataset
    X,Y = Load_Sinlge_Omniglot_Dataset(pathin=args.pathin)
    ## Run the model
    Result = Run_model(X,Y)
    ## Save the Result
    Result.to_csv(args.pathout + args.Name_report, index=False, sep=',',
                  float_format='%.4f')





