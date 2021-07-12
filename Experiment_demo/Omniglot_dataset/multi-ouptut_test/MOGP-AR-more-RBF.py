## This is the main function for runing modul ##

### import Necessary packages
import sys
import os
sys.path.append('YourPath')
#e.g., sys.path.append('./MOGP-AR')
import numpy as np
import tensorflow as tf
import time
import warnings
import pandas as pd
import argparse
from sklearn.model_selection import StratifiedKFold  ## for c-v
warnings.filterwarnings('ignore')  # ignore DeprecationWarnings from tensorflow

### Import form our model
from gpflow.ci_utils import ci_niter  ## for the number of training
from MOGP_convolutionalKernel.utils import  run_adam ## Import optimization
from MOGP_convolutionalKernel.utils import Performance_list, Performance_measure, Transform_data_for_training, Combining_X_with_Index ## Import the performance measure
from MOGP_convolutionalKernel.Building_Models import Build_MOGPAR_RBF_outputs ## Model
from MOGP_convolutionalKernel.Load_data import Save_result_more_output,prepare_data_list ## Dataset
from MOGP_convolutionalKernel.Load_data import Find_optimal_U_MOGP_AR, Load_Omniglot_data_from_directory
# reproducibility
import random
random.seed(3)
np.random.seed(1)
tf.random.set_seed(24)



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

    ######################################
    #####  Performacne Metric Set up #####
    ######################################

    #### MOGP_AR Performacne Metric#####
    # Performance Metric for each outputs and we make a list for thos performance metric
    # E.g we have two outputs. The Performance metric have results for the first output and second output separately.
    # E.g Performance_measure(mu=np.argmax(mu_y_MOGP_AR[0], axis=1), Yest=Y_test_data[0])
    # and Performance_measure(mu=np.argmax(mu_y_MOGP_AR[0], axis=1), Yest=Y_test_data[0])
    # Further, for each output, it also includes the all result for each cross-validation.
    time_training_MOGP_AR_d, Prediction_error_MOGP_AR_d, Precision_weighted_MOGP_AR_d,\
    Recall_weighted_MOGP_AR_d,F1_weighted_MOGP_AR_d = Performance_list()

    ## The Performance Metric for whole output. Some performance measure does not work since we compute the error by using
    ## all the predictived result together. # E.g Performance_measure(mu=np.argmax([mu_y_MOGP_AR[0],mu_y_MOGP_AR[1]], axis=1), Yest=[Y_test_data[0],Y_test_data[1]])
    ## In this kind of calculation, the performance metric Recall_weighted_MOGP_AR and Prediction_error_MOGP_AR works. We ignore those performance now.
    ## Frankly, we may igore all the perfromance in whole outputs later.
    ## By the way, the performance include the all the result for each cross-validation
    time_training_MOGP_AR, Prediction_error_MOGP_AR, Precision_weighted_MOGP_AR, \
    Recall_weighted_MOGP_AR, F1_weighted_MOGP_AR = Performance_list()
    MOGP_AR_U = []
    MOGP_AR_whole_time = []

    ############################
    ## set up some parameters ##
    ############################
    Num_cv = args.num_cv ## number of cross-validation
    Num_class = args.num_latent_f_list ## number of class in each output
    Num_output = args.num_output ## number of outputs

    minibatch_size = args.Size_Minibatch ## minibatch_size
    maxiter = ci_niter(args.Number_train) ## maxiter interation for training

    ###########################################
    #####  Prepare training and test data #####
    ###########################################
    Xtrain_all,Xtest_all,Ytrain_all,Ytest_all = prepare_data_list(X,Y,Num_output,Num_cv)

    #################################################################
    #### We do cross-validataion for all training and test data #####
    #################################################################
    ## The loop is for cross-validation, e.g there are 2 cross-validation datasets
    for i in range(Num_cv):
        a_MOGP_AR_whole_time = time.time()
        ### Find the optmial number of U in MOGP-AR
        Optimal_Num_u_MOGP_AR= Find_optimal_U_MOGP_AR(Num_output, Xtrain_all, Ytrain_all, i, args.Num_U_CV, args.ns,
                               args.whole_latent_f, args.num_sample, args.num_latent_f_list, args.num_output,
                               minibatch_size, Num_class, maxiter, args.cv_measure)
        # Optimal_Num_u_MOGP_AR = 3

        #########################################################
        #### Prepare all output data for one cross-validation ###
        #########################################################
        X_train_data = [] # X_train_data : the training dataset for all 50 outputs for one c-v
        X_test_data = [] # X_test_data : the test dataset for all 50 outputs for one c-v
        Y_train_data = [] # Y_train_data : the training dataset for all 50 outputs for one c-v
        Y_test_data = [] # Y_test_data : the test dataset for all 50 outputs for one c-v
        for m in range(Num_output):
            X_train_data.append(Xtrain_all[m][i])
            X_test_data.append(Xtest_all[m][i])
            Y_train_data.append(Ytrain_all[m][i])
            Y_test_data.append(Ytest_all[m][i])

        ###########################
        #### Building our model ###
        ###########################
        MOGP_AR = Build_MOGPAR_RBF_outputs(X_train_data,
                                           Num_U=Optimal_Num_u_MOGP_AR, Num_sub=args.ns,
                                           Whole_f=args.whole_latent_f,
                                           num_subsample=args.num_sample,
                                           num_latent_f=args.num_latent_f_list,
                                           num_output=args.num_output,
                                           minibatch=minibatch_size,
                                           Num_inducing_point= args.Num_inducing_point)
        ###########################
        #### Training our model ###
        ###########################
        ## Transform_data_for_training and optimizaing the model by adam
        Data_opti = Transform_data_for_training(X_train_data, Y_train_data, Num_class, Num_output)
        a_MOGP_AR = time.time()
        logf_mogp_AR = run_adam(MOGP_AR, maxiter, Data_opti,100, minibatch_size,Moreoutput=True)
        b_MOGP_AR = time.time()
        print('runing time:', b_MOGP_AR - a_MOGP_AR)
        time_training_MOGP_AR_d.append(b_MOGP_AR - a_MOGP_AR)
        MOGP_AR_whole_time.append(b_MOGP_AR - a_MOGP_AR_whole_time)
        MOGP_AR_U.append(Optimal_Num_u_MOGP_AR)

        ###########################
        #### Making prediction ####
        ###########################
        ## prepare the prediction dataset
        ## X_pre_MOGP_AR is a matrix that include for all the outputs
        X_pre_MOGP_AR = Combining_X_with_Index(X_test_data,Num_output)

        # mu_y_MOGP_AR is a list that include all the prediction for all the outputs
        mu_y_MOGP_AR = []
        for task_index in tf.range(Num_output):
            mu_y_MOGP_AR_initial,_ = MOGP_AR.predict_y_one_output(Task=task_index,Xnew=X_pre_MOGP_AR[task_index])
            mu_y_MOGP_AR.append(mu_y_MOGP_AR_initial)
        # mu_y_MOGP_AR, _ = MOGP_AR.predict_y_categorical(np.vstack(X_pre_MOGP_AR))

        ## Combine all prediction result together
        Y_pre_all = []
        for q in range(args.num_output):
            Y_pre_all.append(np.argmax(mu_y_MOGP_AR[q], axis=1))
        mu_y_whole = np.hstack(Y_pre_all)
        Ytest = np.vstack(Y_test_data)


                                ####################################
                                ####### Performance Measure ########
                                ####################################

        ###################################
        ## Check each output performance ##
        ###################################
        time_training_MOGP_AR_output1, Prediction_error_MOGP_AR_output1, \
        Precision_weighted_MOGP_AR_output1, Recall_weighted_MOGP_AR_output1,\
        F1_weighted_MOGP_AR_output1 = Performance_list()
        for k in range(Num_output):
            ## The preformance measure for all output in one cross-validation dataset
            Test_error_MOGP_AR_output1, P_w_MOGP_AR_output1, R_w_MOGP_AR_output1,\
            F1_w_MOGP_AR_output1 = Performance_measure(mu=np.argmax(mu_y_MOGP_AR[k], axis=1), Yest=Y_test_data[k])

            Prediction_error_MOGP_AR_output1.append(Test_error_MOGP_AR_output1)
            Precision_weighted_MOGP_AR_output1.append(P_w_MOGP_AR_output1)
            Recall_weighted_MOGP_AR_output1.append(R_w_MOGP_AR_output1)
            F1_weighted_MOGP_AR_output1.append(F1_w_MOGP_AR_output1)
        ## The preformance measure for all output and all the cross-validation dataset
        ## We all each output result into the below list
        Prediction_error_MOGP_AR_d.append(Prediction_error_MOGP_AR_output1)
        Precision_weighted_MOGP_AR_d.append(Precision_weighted_MOGP_AR_output1)
        Recall_weighted_MOGP_AR_d.append(Recall_weighted_MOGP_AR_output1)
        F1_weighted_MOGP_AR_d.append(F1_weighted_MOGP_AR_output1)

        ##############################################
        ## Check whole outputs performance together ##
        ##############################################
        # whole output
        Test_error_MOGP_AR, P_w_MOGP_AR, R_w_MOGP_AR,\
        F1_w_MOGP_AR = Performance_measure(mu=mu_y_whole, Yest=Ytest)
        Prediction_error_MOGP_AR.append(Test_error_MOGP_AR)  ### Prediciton error
        Precision_weighted_MOGP_AR.append(P_w_MOGP_AR) ### Precision
        Recall_weighted_MOGP_AR.append(R_w_MOGP_AR) ### Recall
        F1_weighted_MOGP_AR.append(F1_w_MOGP_AR) ### F1

    ###################################
    ## Save all the results together ##
    ###################################
    ## We get all the results
    Result = Save_result_more_output(Num_output, args.num_cv, time_training_MOGP_AR_d,MOGP_AR_whole_time,Prediction_error_MOGP_AR,
                                     Precision_weighted_MOGP_AR,Recall_weighted_MOGP_AR,F1_weighted_MOGP_AR,Prediction_error_MOGP_AR_d,
                                     Precision_weighted_MOGP_AR_d,Recall_weighted_MOGP_AR_d,F1_weighted_MOGP_AR_d)

    return Result


def Dataset(Dataset_name):
    '''
    This is used for importing the dataset and change it into our required format.

    :param Data_name:
    :return:
    '''
    if Dataset_name == 'images_evaluation':
        validation_path = os.path.join(args.pathin, 'images_evaluation')
        val_output = Load_Omniglot_data_from_directory(path=validation_path)
        ####### We splite all dataset separately
        X_all = []  ### all the X for different outputs
        Y_all = []  ### all the corrsponding label for the different outputs
        Name_output = []  ### all name for the different outputs
        Num_class = []  ### all the number of classes for the different outputs
        for i in range(20):
            X_all.append(val_output[i][0])
            Y_all.append(val_output[i][1])
            Name_output.append(val_output[i][2])
            Num_class.append(val_output[i][3])
    elif Dataset_name == 'images_background':
        train_path = os.path.join(args.pathin, 'images_background')
        train_output = Load_Omniglot_data_from_directory(path=train_path)
        ####### We splite all dataset separately
        X_all = []  ### all the X for different outputs
        Y_all = []  ### all the corrsponding label for the different outputs
        Name_output = []  ### all name for the different outputs
        Num_class = []  ### all the number of classes for the different outputs
        for i in range(30):
            X_all.append(train_output[i][0])
            Y_all.append(train_output[i][1])
            Name_output.append(train_output[i][2])
            Num_class.append(train_output[i][3])

    elif Dataset_name == 'Ominiglot_all':
        train_path = os.path.join(args.pathin, 'images_background')
        validation_path = os.path.join(args.pathin, 'images_evaluation')
        train_output = Load_Omniglot_data_from_directory(path=train_path)
        val_output = Load_Omniglot_data_from_directory(path=validation_path)
        ####### We splite all dataset separately
        X_all = []  ### all the X for different outputs
        Y_all = []  ### all the corrsponding label for the different outputs
        Name_output = []  ### all name for the different outputs
        Num_class = []  ### all the number of classes for the different outputs
        for i in range(30):
            X_all.append(train_output[i][0])
            Y_all.append(train_output[i][1])
            Name_output.append(train_output[i][2])
            Num_class.append(train_output[i][3])
        for i in range(20):
            X_all.append(val_output[i][0])
            Y_all.append(val_output[i][1])
            Name_output.append(val_output[i][2])
            Num_class.append(val_output[i][3])
    else:
        print('Please use the exieted dataset')
        X_all = None
        Y_all = None
    return X_all, Y_all





if __name__ == '__main__':
    print("starting to run")
    #########################
    ### Set up parameters ###
    #########################
    ## Inputs for the main function
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--Name_report', type=str, default='Result_MOGP_AR_test.csv', help='The Name of output report')
    ## Training parameters
    parser.add_argument('--Size_Minibatch', type=int, default=50, help='The number of minibatch')
    parser.add_argument('--Number_train', type=int, default=50, help='The number of training times')
    parser.add_argument('--cv_measure', type=str, default='Recall')
    ## Model parameters
    parser.add_argument('--ns', type=int, nargs='+', default=[1, 2], help='The number of subsampling for each output')
    parser.add_argument('--num_sample', type=int, nargs='+', default=[2, 3], help='The number of sample in each output')
    parser.add_argument('--num_latent_f_list', type=int, nargs='+', default=[6, 7],
                        help='The number of latent parameter functions f in each output')
    parser.add_argument('--num_output', type=int, default=2, help='The number of outputs')
    parser.add_argument('--whole_latent_f', type=int, default=13,
                        help='The number of all the latent parameter functions f')
    parser.add_argument('--Num_inducing_point', type=int, default=3, help='The number of cross-validation')

    ## Cross-validation parameters
    parser.add_argument('--Num_U_CV', type=int, nargs='+', default=[2,3,4,5,6], help='The number of U for Cross validation')
    parser.add_argument('--num_cv', type=int, default=3, help='The number of cross-validation')
    ## Path in and out
    parser.add_argument('--pathin', type=str, default='YourPath/')
    ## e.g., parser.add_argument('--pathin', type= str, default='../MOGP_AR_Dataset/images')
    parser.add_argument('--pathout', type= str, default='YourPath/')
    ## e.g., parser.add_argument('--pathout', type= str, default='Result/')
    parser.add_argument('--Name_setdata',type= str,default='images_background',help='The name of the dataset')

    args = parser.parse_args()

    ########################
    ## Import our Dataset ##
    ########################
    X_all,Y_all = Dataset(Dataset_name=args.Name_setdata)

    ###################
    ## Run the model ##
    ###################
    Result = Run_model(X_all, Y_all)

    #####################
    ## Save the Result ##
    #####################
    Result.to_csv(args.pathout + args.Name_report, index=False, sep=',',
                  float_format='%.4f')





