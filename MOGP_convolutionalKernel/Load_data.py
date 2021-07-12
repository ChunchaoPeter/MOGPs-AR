

                                #######################################################
                                #################### Load data set ####################
                                #######################################################
import gpflow
import numpy as np
## This is for image dataset
import os
from PIL import Image
import glob
import pandas as pd
from sklearn.model_selection import StratifiedKFold  ## for c-v
from MOGP_convolutionalKernel.utils import Transform_data_for_training,run_adam,Combining_X_with_Index
from sklearn.model_selection import train_test_split
from MOGP_convolutionalKernel.Building_Models import Build_MOGPAR_RBF_outputs, Build_MOGPAR_Conv_outputs
from sklearn.metrics import recall_score
import tensorflow as tf

def Load_Omniglot_data_from_directory(path):
    '''
    We load the Omniglot dataset and resize it into 20 *20.
    Output: A list that contains different outputs.
    Each output is (X,Y,Name,Num_classes)
    where X is the we resize the data into 20 * 20, then we change it into a vector e.g 400
    Y is the classification corresponding to the X
    Name: the name of the dataset alphabet
    Num_classes: The number of characters in that alphabet or the number of classes in that output.
    '''
    all_output = []
    path_order = sorted(os.listdir(path))
    # for alphabet in os.listdir(path):
    for alphabet in path_order:
        print("loading alphabet: " + alphabet)
        alphabet_path = os.path.join(path, alphabet)

        ## Each character in alphabet is  in a separate folder
        X_alphabet = []
        for letter in sorted(os.listdir(alphabet_path)):
            letter_path = os.path.join(alphabet_path, letter)
            Data_path = os.path.join(letter_path,'*g')

            Data_set = []
            Files_data = glob.glob(Data_path)
            Files_data.sort()

            ### The for below is for each character that include 20 image
            for f in Files_data:
                ## change the data formate and resize the image to 20 * 20
                img = np.array(Image.open(f).resize((20, 20))) + 0
                Data_set.append(img)
            ## change the each character into vector format
            X_data = np.array(Data_set).reshape(20, 400).astype(gpflow.config.default_float())
            X_alphabet.append(X_data)
        ## We combine all characters in the same alphabet into a same output
        X_output = np.vstack(X_alphabet)
        Y_output = np.vstack([np.arange(int(X_output.shape[0]/20))]*20).T.reshape(-1,1).astype(gpflow.config.default_int())
        data_alphabet = (X_output, Y_output,alphabet,int(X_output.shape[0]/20))
        all_output.append(data_alphabet)
    return all_output


def Load_Sinlge_Omniglot_Dataset(pathin):
    '''
    We load the Omniglot dataset and resize it into 20 *20.
    Note: we only choose one dataset in one time
    X_output is the we resize the data into 20 * 20, then we change it into a vector e.g 400
    Y_output is the classification corresponding to the X_output
    Args:
        Data_name: the Name of dataset
    Returns:
        X_output: Input
        Y_output: Output
    '''
    ### The alphabet_path that we need the path for each data set
    alphabet_path = pathin
    X_alphabet = []

    ## We also make the data in order
    for letter in sorted(os.listdir(alphabet_path)):
        letter_path = os.path.join(alphabet_path, letter)
        Data_path = os.path.join(letter_path, '*g')
        Data_set = []
        Files_data = glob.glob(Data_path)

        ## we make the dataset in order
        Files_data.sort()


        for f in Files_data:
        ## change the data formate and resize the image to 20 * 20
            img = np.array(Image.open(f).resize((20, 20))) + 0
            Data_set.append(img)
            ## change the each character into vector format
        X_data = np.array(Data_set).reshape(20, 400).astype(gpflow.config.default_float())
        X_alphabet.append(X_data)

    ## We combine all characters in the same alphabet into a same output
    X_output = np.vstack(X_alphabet).astype(gpflow.config.default_float())
    Y_output = np.vstack([np.arange(int(X_output.shape[0]/20))]*20).T.reshape(-1,1).astype(gpflow.config.default_int())

    return X_output, Y_output


def Save_result_more_output(Num_output,num_cv,time_training_MOGP_AR_d,MOGP_AR_whole_time,Prediction_error_MOGP_AR,
                            Precision_weighted_MOGP_AR,Recall_weighted_MOGP_AR,
                            F1_weighted_MOGP_AR,Prediction_error_MOGP_AR_d,Precision_weighted_MOGP_AR_d,Recall_weighted_MOGP_AR_d,
                            F1_weighted_MOGP_AR_d):
    ## Num_outpu: how many outputs
    ## num_cv: how many cross-validation
    ## We get all the results for more output case
    Result = pd.DataFrame({'Model': ['MOGP_AR'],
                           '0': ['&'],
                           'Training Time mean': [np.mean(time_training_MOGP_AR_d)],
                           'Training Time std': [np.std(time_training_MOGP_AR_d)],
                           '1': [np.mean(MOGP_AR_whole_time)],
                           'Mean Prediciton error': [np.mean(Prediction_error_MOGP_AR)],
                           '2': [np.std(MOGP_AR_whole_time)],
                           'Std prediction error': [np.std(Prediction_error_MOGP_AR)],
                           '3': ['&'],
                           'Mean Precision_weighted': [np.mean(Precision_weighted_MOGP_AR)],
                           '4': ['&'],
                           'Std Precision_weighted': [np.std(Precision_weighted_MOGP_AR)],
                           '5': ['&'],
                           'Mean Recall_weighted': [np.mean(Recall_weighted_MOGP_AR)],
                           '6': ['&'],
                           'Std Recall_weighted': [np.std(Recall_weighted_MOGP_AR)],
                           '7': ['&'],
                           'Mean F1_weighted': [np.mean(F1_weighted_MOGP_AR)],
                           '8': ['&'],
                           'Std F1_weighted': [np.std(F1_weighted_MOGP_AR)]
                           })

    ## Add MOGP_AR result in to result
    Prediction_error = []
    Precision_weighted = []
    Recall_weighted = []
    F1_weighted = []

    for m in range(Num_output):
        ## Make the train and test data for each output
        Prediction_error_cv = []
        Precision_weighted_cv = []
        Recall_weighted_cv = []
        F1_weighted_cv = []
        ## we do the c-v
        for q in range(num_cv):
            Prediction_error_cv.append(Prediction_error_MOGP_AR_d[q][m])
            Precision_weighted_cv.append(Precision_weighted_MOGP_AR_d[q][m])
            Recall_weighted_cv.append(Recall_weighted_MOGP_AR_d[q][m])
            F1_weighted_cv.append(F1_weighted_MOGP_AR_d[q][m])

        Prediction_error.append(Prediction_error_cv)
        Precision_weighted.append(Precision_weighted_cv)
        Recall_weighted.append(Recall_weighted_cv)
        F1_weighted.append(F1_weighted_cv)


    for f in range(Num_output):
        Result.loc[f+1] = [ f+1 , '&', '&', '&', '&',
                         np.mean(Prediction_error[f]), '&', np.std(Prediction_error[f]), '&',
                         np.mean(Precision_weighted[f]), '&',np.std(Precision_weighted[f]), '&',
                         np.mean(Recall_weighted[f]), '&',np.std(Recall_weighted[f]), '&',
                         np.mean(F1_weighted[f]), '&', np.std(F1_weighted[f])
                         ]

    return Result


def prepare_data_list(X,Y,Num_output,Num_cv):
    '''
    The X and Y include all dataset for different outputs.
    This function split the input and ouptut dataset into training and test dataset.
    The training and test are list that inclued training and test for each cross-validation.
    We will return the training and test data for x and y for each cross-validation.
    E.g Xtrain_all include all the training dataset for all cross-valdataion dataset for all output.
    E.g Xtrain_all[0] is all the training dataset for all cross-valdataion dataset for first output.
    Xtrain_all[0][0] is all the training dataset for first cross-valdataion dataset for first output

    :param X: a list
    :param Y: a list
    :param Num_output: the number of outputs
    :param Num_cv: the number of cross-validation
    :return: Xtrain_all, Xtest_all, Ytrain_all, Ytest_all dataset
    '''
    sfolder = StratifiedKFold(n_splits=Num_cv, random_state=None, shuffle=False)

    X_all = X  ### all the X for different outputs
    Y_all = Y ### all the corrsponding label for the different outputs

    ############################################################################
    #### The training and test data X and Y with cross-validation as a list #####
    #############################################################################

    ## We split the training and test dataset
    Xtrain_all = []  ### training dataset in X for different outputs; In the list, each element has all cross-validation datasets, e.g. c-v = 2
    Xtest_all = []  ### test dataset in X for different outputs; In the list, each element has all cross-validation datasets, e.g. c-v = 2
    Ytrain_all = []  ### training dataset in Y for different outputs; In the list, each element has all cross-validation datasets, e.g. c-v = 2
    Ytest_all = []  ### test dataset in Y for different outputs; In the list, each element has all cross-validation datasets, e.g. c-v = 2

    for i in range(Num_output):
        ## Make the train and test data for each output
        Xtrain_each = []
        Xtest_each = []
        Ytrain_each = []
        Ytest_each = []
        ## we do the c-v
        for train_index, test_index in sfolder.split(X_all[i], Y_all[i]):
            Xtrain, Xtest = X_all[i][train_index], X_all[i][test_index]
            Ytrain, Ytest = Y_all[i][train_index], Y_all[i][test_index]

            ## all split training and test data into a list in each output
            Xtrain_each.append(Xtrain)
            Xtest_each.append(Xtest)
            Ytrain_each.append(Ytrain)
            Ytest_each.append(Ytest)

        ## We combine all the output dataset together
        Xtrain_all.append(Xtrain_each)
        Xtest_all.append(Xtest_each)
        Ytrain_all.append(Ytrain_each)
        Ytest_all.append(Ytest_each)
    return Xtrain_all, Xtest_all, Ytrain_all, Ytest_all


def Find_optimal_U_MOGP_AR(Num_output,Xtrain_all,Ytrain_all,i,Num_U_CV,ns,
                           whole_latent_f,num_sample,num_latent_f_list,num_output,
                           minibatch_size,Num_class,maxiter,cv_measure):
    '''
    This function is to find the optmizal number of U for MOGP-AR.
    We split the existed training dataset into train and validataion dataset
    The train data is to train our model with different number of U.
    The validation data set used to find the optimizal number of U
    :param Num_output: the number of output
    :param Xtrain_all: training input data
    :param Ytrain_all: training output data
    :param i: the index for cross-validation for
    :param Num_U_CV: a list that include all the number of U
    :param ns: parameter for MOGP-AR
    :param whole_latent_f: parameter for MOGP-AR
    :param num_sample: parameter for MOGP-AR
    :param num_latent_f_list: parameter for MOGP-AR
    :param num_output: parameter for MOGP-AR
    :param minibatch_size: parameter for MOGP-AR
    :param Num_class: parameter for MOGP-AR
    :param maxiter: parameter for MOGP-AR
    :param cv_measure: the performance that we used
    :return: The optimal number of U
    '''
    X_train_cv = []
    X_val = []
    Y_train_cv = []
    Y_val = []

    ## The loop for loop is for all outputs
    for j in range(Num_output):
        X_train_cv_each, X_val_each, Y_train_cv_each, Y_val_each = train_test_split(Xtrain_all[j][i],
                                                                                    Ytrain_all[j][i], test_size=0.2,
                                                                                    random_state=0)
        ## There a list for each outputs for training and validation dataset.
        X_train_cv.append(X_train_cv_each) # X_train_cv : the training dataset for all 50 outputs for one c-v
        X_val.append(X_val_each)  # X_val : the validation dataset for all 50 outputs for one c-v
        Y_train_cv.append(Y_train_cv_each) # Y_train_cv : the training dataset for all 50 outputs for one c-v
        Y_val.append(Y_val_each) # Y_val : the val dataset for all 50 outputs for one c-v

    CV_error_AR = []
    ##############################################
    ####### Selecting the number of U  ###########
    ##############################################
    for N_U in Num_U_CV:
        MOGP_AR_cv = Build_MOGPAR_RBF_outputs(X_train_cv,
                                              Num_U=N_U, Num_sub=ns,
                                              Whole_f=whole_latent_f,
                                              num_subsample=num_sample,
                                              num_latent_f=num_latent_f_list,
                                              num_output=num_output, minibatch=minibatch_size)

        Data_opti_cv = Transform_data_for_training(X_train_cv, Y_train_cv, Num_class, Num_output)

        logf_mogp_ar_cv = run_adam(MOGP_AR_cv, maxiter, Data_opti_cv, 100, minibatch_size, Moreoutput=True)

        ## The X_val is a list that include all the values for all the outputs
        ## X_pre_MOGP_AR_cv is a matrix that include for all the outputs

        ## mu_y_MOGP_AR_cv is a list that include all the prediction for all the outputs
        X_pre_MOGP_AR_cv = Combining_X_with_Index(X_val,Num_output)
        mu_y_MOGP_AR_cv = []
        for task_index in tf.range(Num_output):
            mu_y_MOGP_AR_initial_cv,_ = MOGP_AR_cv.predict_y_one_output(Task=task_index,Xnew=X_pre_MOGP_AR_cv[task_index])
            mu_y_MOGP_AR_cv.append(mu_y_MOGP_AR_initial_cv)


        y_predi = []
        for h in range(Num_output):
            y_predi.append(np.argmax(mu_y_MOGP_AR_cv[h], axis=1))

        mu_y_whole_cv = np.hstack(y_predi)
        Y_val_cv = np.vstack(Y_val)

        if cv_measure == 'Accuracy':
            Test_error_AR_cv = np.mean(mu_y_whole_cv == Y_val_cv.squeeze())
        elif cv_measure == 'Recall':
            Test_error_AR_cv = recall_score(Y_val_cv, mu_y_whole_cv[:, None], average='weighted')
        else:
            print('Please print the correct the performance measure')
            break
        CV_error_AR.append(Test_error_AR_cv)
    # Choose the optimal value for the number of U
    Optimal_Num_u_MOGP_AR = Num_U_CV[CV_error_AR.index(max(CV_error_AR))]

    return Optimal_Num_u_MOGP_AR


def Find_optimal_U_MOGP_AR_Conv(Num_output,Xtrain_all,Ytrain_all,i,Num_U_CV,ns,
                           whole_latent_f,num_sample,num_latent_f_list,num_output,
                           minibatch_size,Num_class,maxiter,cv_measure,patch1, patch2, Maximum_induc_patch):
    '''
    This function is to find the optmizal number of U for MOGP-AR-Conv.
    We split the existed training dataset into train and validataion dataset
    The train data is to train our model with different number of U.
    The validation data set used to find the optimizal number of U
    :param Num_output: the number of output
    :param Xtrain_all: training input data
    :param Ytrain_all: training output data
    :param i: the index for cross-validation for
    :param Num_U_CV: a list that include all the number of U
    :param ns: parameter for MOGP-AR-Conv
    :param whole_latent_f: parameter for MOGP-AR-Conv
    :param num_sample: parameter for MOGP-AR-Conv
    :param num_latent_f_list: parameter for MOGP-AR-Conv
    :param num_output: parameter for MOGP-AR-Conv
    :param minibatch_size: parameter for MOGP-AR-Conv
    :param patch1: parameter for MOGP-AR-Conv
    :param patch2: parameter for MOGP-AR-Conv
    :param Maximum_induc_patch: parameter for MOGP-AR-Conv
    :param Num_class: parameter for MOGP-AR-Conv
    :param maxiter: parameter for MOGP-AR-Conv
    :param cv_measure: the performance that we used
    :return: The optimal number of U
    '''
    X_train_cv = []
    X_val = []
    Y_train_cv = []
    Y_val = []

    ## The loop for loop is for all outputs
    for j in range(Num_output):
        X_train_cv_each, X_val_each, Y_train_cv_each, Y_val_each = train_test_split(Xtrain_all[j][i],
                                                                                    Ytrain_all[j][i], test_size=0.2,
                                                                                    random_state=0)
        ## There a list for each outputs for training and validation dataset.
        X_train_cv.append(X_train_cv_each) # X_train_cv : the training dataset for all 50 outputs for one c-v
        X_val.append(X_val_each)  # X_val : the validation dataset for all 50 outputs for one c-v
        Y_train_cv.append(Y_train_cv_each) # Y_train_cv : the training dataset for all 50 outputs for one c-v
        Y_val.append(Y_val_each) # Y_val : the val dataset for all 50 outputs for one c-v

    CV_error_AR = []
    ##############################################
    ####### Selecting the number of U  ###########
    ##############################################
    for N_U in Num_U_CV:
        MOGP_AR_cv = Build_MOGPAR_Conv_outputs(X_train_cv,
                                              Num_U=N_U, Num_sub=ns,
                                              Whole_f=whole_latent_f,
                                              num_subsample=num_sample,
                                              num_latent_f=num_latent_f_list,
                                              num_output=num_output, minibatch=minibatch_size,
                                              patch1=patch1,patch2=patch2, Maximum_induc_patch=Maximum_induc_patch)

        Data_opti_cv = Transform_data_for_training(X_train_cv, Y_train_cv, Num_class, Num_output)

        logf_mogp_ar_cv = run_adam(MOGP_AR_cv, maxiter, Data_opti_cv, 100, minibatch_size, Moreoutput=True)

        ## The X_val is a list that include all the values for all the outputs
        ## X_pre_MOGP_AR_cv is a matrix that include for all the outputs

        ## mu_y_MOGP_AR_cv is a list that include all the prediction for all the outputs
        X_pre_MOGP_AR_cv = Combining_X_with_Index(X_val,Num_output)
        mu_y_MOGP_AR_cv = []
        for task_index in tf.range(Num_output):
            mu_y_MOGP_AR_initial_cv,_ = MOGP_AR_cv.predict_y_one_output(Task=task_index,Xnew=X_pre_MOGP_AR_cv[task_index])
            mu_y_MOGP_AR_cv.append(mu_y_MOGP_AR_initial_cv)


        y_predi = []
        for h in range(Num_output):
            y_predi.append(np.argmax(mu_y_MOGP_AR_cv[h], axis=1))

        mu_y_whole_cv = np.hstack(y_predi)
        Y_val_cv = np.vstack(Y_val)

        if cv_measure == 'Accuracy':
            Test_error_AR_cv = np.mean(mu_y_whole_cv == Y_val_cv.squeeze())
        elif cv_measure == 'Recall':
            Test_error_AR_cv = recall_score(Y_val_cv, mu_y_whole_cv[:, None], average='weighted')
        else:
            print('Please print the correct the performance measure')
            break
        CV_error_AR.append(Test_error_AR_cv)
    # Choose the optimal value for the number of U
    Optimal_Num_u_MOGP_AR = Num_U_CV[CV_error_AR.index(max(CV_error_AR))]

    return Optimal_Num_u_MOGP_AR