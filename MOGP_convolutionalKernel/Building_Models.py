"""
                    Here we build different models:
## Prediction for one output datasets
GPC_RBF: We build the Gaussian Processes for Classification with RBF kernel
GPC_Conv: We build the Gaussian Processes for Classification with Convolutional kernel
GPAR_RBF: We build the Gaussian Processes for Classification with RBF kernel in the Haitao Paper
GPAR_Conv: We build the Gaussian Processes for Classification with Convolutional kernel in the Haitao Paper
MOGP_RBF: We build the MOGP for Classification with RBF kernel in a single output
MOGPAR_RBF: We build the MOGPAR for Classification with RBF kernel in a single output

## Prediction for multiple output datasets
MOGPAR_RBF_outputs: We build the MOGPAR for Classification with RBF kernel in More outputs
MOGPAR_Conv_outputs: We build the MOGPAR Processes for Classification with Convolutional kernel in More outputs
"""

import numpy as np
import gpflow
import sys
sys.path.append('/home/chunchao/Desktop/First_project_clean_code/MOGP-AR/')
## Inducing variables
from gpflow.inducing_variables import InducingPoints, SeparateIndependentInducingVariables,InducingPatches
## Kernel
from MOGP_convolutionalKernel.kernels import lmc_kernel
from MOGP_convolutionalKernel.kernels import Convolutional_SE as CSE

## Likelihood
from MOGP_convolutionalKernel.likelihoods import SwitchedLikelihood_MOGP, Softmax_mogp, MultiClass_SoftMax_Aug, \
     SwitchedLikelihoodMOGP_AR, SoftmaxMOGP_AR
## Model
from MOGP_convolutionalKernel.models import MOGP_Categorical, MOGPC_AR,MOGPC_AR_More,SVGP_convolution

# reproducibility
import random
random.seed(3)
np.random.seed(1)


                    #####################################
                    ###### GPC with RBF kernel ##########
                    #####################################

def Build_GPC_RBF(Xtrain, Ytrain, Num_classes, Num_inducing_point = 100):
    '''
    We build GPC model with RBF kernel

    Args:
        Xtrain: the training data for input
        Ytrain: the training data for output
        Num_classes: the number of classes for outputs
    Returns:
        GPC: The GPC model
        data_GPC: All the training data for GPC
        N: The number of training data
     '''

    ## The training data
    data_GPC = (Xtrain, Ytrain)
    ## Inducing input
    Z = Xtrain[:Num_inducing_point].copy()
    N = Xtrain.shape[0]
    N_c = Num_classes

    ### Build our model
    ## kernel
    kernel = gpflow.kernels.RBF(lengthscales=[1]*Xtrain.shape[1])
    ## Changing the inducing points format
    feature = gpflow.inducing_variables.InducingPoints(Z)
    ## Building GPC
    GPC = gpflow.models.SVGP(kernel=kernel,
                           likelihood=Softmax_mogp(num_classes=N_c),  # Multiclass likelihood
                           inducing_variable=feature, num_latent_gps=N_c)
    return GPC, data_GPC, N

                    ###############################################
                    ###### GPC with convolutional kernel ##########
                    ###############################################

def Build_GPC_Conv(Xtrain, Ytrain,Num_classes,patch1, patch2,Maximum_induc_patch):
    '''
    We build GPC model with convolutional kernel

    Args:
        Xtrain: the training data for input
        Ytrain: the training data for output
        Num_classes: the number of classes for outputs
    Returns:
        GPC: The GPC model
        data_GPC: All the training data for GPC
        N: The number of training data
     '''
    ## The training data
    data_GPC_conv = (Xtrain, Ytrain)
    ## Inducing input
    N = Xtrain.shape[0]
    N_c = Num_classes

    ### Build our model
    kern_conv = CSE(image_shape=[20, 20], patch_shape=[patch1, patch2])
    ## Inducing points
    Z = np.unique(CSE(image_shape=[20, 20], patch_shape=[patch1, patch2]).get_patches(Xtrain).numpy().reshape(-1, patch1*patch2), axis=0)
    a = Z.shape[0]
    ### We choose maximum 1000 inducing data points
    if a > Maximum_induc_patch:
        a = Maximum_induc_patch
    else:
        a = a

    print('Size of inducing pathsize', a)
    idx = np.random.randint(Z.shape[0], size=a)
    Z_m = Z[idx, :].copy()
    conv_f = InducingPatches(Z_m)

    GPC_conv = SVGP_convolution(kernel=kern_conv,
                         likelihood=Softmax_mogp(num_classes=N_c),  # Multiclass likelihood
                         inducing_variable=conv_f, num_latent_gps=N_c)

    return GPC_conv, data_GPC_conv, N


                    ######################################
                    ###### GP-AR with RBF kernel #########
                    ######################################

def Build_GP_AR_RBF(Xtrain, Ytrain,Num_classes, Num_inducing_point = 100):
    '''
    We build GPAR model with RBF kernel
    Args:
        Xtrain: the training data for input
        Ytrain: the training data for output
        Num_classes: the number of classes for outputs
    Returns:
        GPAR: The GPAR model
        data_GPAR: All the training data for GPAR
        N: The number of training data
     '''
    ## The training data
    data_GPAR = (Xtrain, Ytrain)
    ## Inducing input
    Z = Xtrain[:Num_inducing_point].copy()
    N = Xtrain.shape[0]
    N_c = Num_classes
    ### Build our model
    ## kernel
    kernel = gpflow.kernels.RBF(lengthscales=[1]*Xtrain.shape[1])
    ## Changing the inducing points format
    feature = gpflow.inducing_variables.InducingPoints(Z)
    ## Building GPAR
    GPAR = gpflow.models.SVGP(kernel=kernel,
                           likelihood=MultiClass_SoftMax_Aug(N_c),
                           inducing_variable=feature, num_latent_gps=N_c)

    return GPAR, data_GPAR, N

                    ################################################
                    ###### GP-AR with Convolutional kernel #########
                    ################################################

def Build_GP_AR_Conv(Xtrain, Ytrain,Num_classes,patch1,patch2,Maximum_induc_patch):
    '''
    We build GPAR model with convolutional kernel

    Args:
        Xtrain: the training data for input
        Ytrain: the training data for output
        Num_classes: the number of classes for outputs
    Returns:
        GPAR: The GPAR model
        data_GPAR: All the training data for GPAR
        N: The number of training data
     '''
    ## The training data
    data_GPAR_conv = (Xtrain, Ytrain)
    ## Inducing input
    N = Xtrain.shape[0]
    N_c = Num_classes

    ### Build our model
    kern_conv = CSE(image_shape=[20, 20], patch_shape=[patch1, patch2])
    ## Inducing points
    Z = np.unique(CSE(image_shape=[20, 20], patch_shape=[patch1, patch2]).get_patches(Xtrain).numpy().reshape(-1, patch1*patch2), axis=0)
    a = Z.shape[0]
    ### We choose maximum 1000 inducing data points
    if a > Maximum_induc_patch:
        a = Maximum_induc_patch
    else:
        a = a

    print('Size of inducing pathsize', a)
    idx = np.random.randint(Z.shape[0], size=a)
    Z_m = Z[idx, :].copy()
    conv_f = InducingPatches(Z_m)

    ## Building GPAR
    GPAR_conv = SVGP_convolution(kernel=kern_conv,
                           likelihood=MultiClass_SoftMax_Aug(N_c),
                           inducing_variable=conv_f, num_latent_gps=N_c)

    return GPAR_conv, data_GPAR_conv, N

                    ###########################################################
                    ###### MOGP-AR with RBF kernel for single output ##########
                    ###########################################################

def Build_MOGP_AR_RBF(Xtrain, Ytrain, Num_U, Num_sub, Whole_f, num_subsample,num_latent_f,num_output, Size_minibatch):
    '''
    We build MOGP_AR model with RBF kernel for single output
    Args:
        Xtrain: the training data for input
        Ytrain: the training data for output
    Returns:
        MOGP_AR: The MOGP_AR model
        data_MOGP_AR: All the training data for MOGP_AR
        N: The number of training data
     '''
    ## Inducing input
    Z = Xtrain[:100].copy()
    ## Change training input and output format for MOGP
    index0 = Ytrain
    X_m = np.hstack((Xtrain, index0))
    Y_m = np.vstack((np.hstack((Ytrain, np.zeros_like(Ytrain)))))
    ## The number of training data
    N = X_m.shape[0]

    ## Build our model
    ## The latent functions u
    ks = []
    for j in range(Num_U):
        ks.append(gpflow.kernels.RBF(lengthscales=[1] * Xtrain.shape[1]))

    ## Change the inducing input format
    L = len(ks)
    Zs = [Z.copy() for _ in range(L)]
    iv_list = [InducingPoints(Z) for Z in Zs]
    iv = SeparateIndependentInducingVariables(iv_list)

    ## the number of subsampling class size ($|\mathbf{S}| = 1$)
    ns = Num_sub
    # ns = Num_sub

    ## All the sample we need to use. Since we always have one corrsponding sample, we plus 1.

    ## Kernel for MOGP_AR
    kern = lmc_kernel(Whole_f, ks)
    ## Building Likelihood
    lik = SwitchedLikelihoodMOGP_AR(likelihood_list=[SoftmaxMOGP_AR()],
                               num_subsample_list=num_subsample,
                               num_latent_f_list=num_latent_f, ns = ns)
    ## Building MOAP_AR
    MOGP_AR = MOGPC_AR(kernel=kern, likelihood=lik, inducing_variable=iv, num_latent_gps=L,
                k=num_latent_f, ns=ns, num_data=N, minibatch=Size_minibatch, num_output=num_output)

    ## The training data
    data_MOGP_AR = (X_m, Y_m)
    return MOGP_AR, data_MOGP_AR, N


                    #########################################################
                    ###### MOGPs with RBF kernel for single output ##########
                    #########################################################

def Build_MOGP_RBF(Xtrain, Ytrain,Num_U, Num_classes):
    '''
    We build MOGP model with RBF kernel for single output

    Args:
        Xtrain: the training data for input
        Ytrain: the training data for output
        Num_classes: the number of classes for outputs
    Returns:
        MOGP: The MOGP model
        data_MOGP: All the training data for MOGP
        N: The number of training data
     '''
    ## Inducing input
    Z = Xtrain[:100].copy()
    ## Change training input and output format for MOGP
    X_m = np.vstack((np.hstack((Xtrain, np.zeros_like(Xtrain[:, 0][:, None])))))
    Y_m = np.vstack((np.hstack((Ytrain, np.zeros_like(Ytrain)))))
    ## The number of training data
    N = X_m.shape[0]

    ## Build our model
    ## The latent functions u
    ks = []
    for j in range(Num_U):
        ks.append(gpflow.kernels.RBF(lengthscales=[1] * Xtrain.shape[1]))

    ## Change the inducing input format
    L = len(ks)
    Zs = [Z.copy() for _ in range(L)]
    iv_list = [InducingPoints(Z) for Z in Zs]
    iv = SeparateIndependentInducingVariables(iv_list)
    N_c = Num_classes

    ## The kernel for MOGP
    kern = lmc_kernel(N_c, ks)
    ## Build likelihood
    lik = SwitchedLikelihood_MOGP(likelihood_list = [Softmax_mogp(num_classes=N_c)], num_latent_list =[N_c])
    ## Build MOGP
    MOGP = MOGP_Categorical(kernel=kern, likelihood=lik, inducing_variable=iv, num_latent_gps=L, num_data=N)
    ## The training data
    data_MOGP = (X_m, Y_m)
    return MOGP, data_MOGP, N

                    ###########################################################
                    ###### MOGP-AR with RBF kernel for multi-outputs ##########
                    ###########################################################

def Build_MOGPAR_RBF_outputs(X, Num_U, Num_sub, Whole_f, num_subsample, num_latent_f, num_output,minibatch,Num_inducing_point=100):
    '''
    We build MOGPAR_RBF_outputs model for multi-output cases
    Args:
        Xtrain: the training data for input
        Ytrain: the training data for output
    Returns:
        MOGP_AR: The MOGPAR_RBF_outputs model
     '''

    ## The number of training data
    XX_list = []
    X_ratio = []
    for i in range(num_output):
        XX_list.append(X[i])
        X_ratio.append(X[i].shape[0]/minibatch) ## each datasize / minibatch

    ## Build our model

    ## The latent functions u
    ks = []
    for j in range(Num_U):
        ks.append(gpflow.kernels.RBF(lengthscales=[1] * X[0].shape[1]))

    ## Change the inducing input format
    ZZ = []
    for i in range(num_output):
        Z1 = X[i][:Num_inducing_point].copy()
        ZZ.append(Z1)
    Z = np.vstack(ZZ)
    L = len(ks)
    Zs = [Z.copy() for _ in range(L)]
    iv_list = [InducingPoints(Z) for Z in Zs]
    iv = SeparateIndependentInducingVariables(iv_list)

    ## the number of subsampling class size ($|\mathbf{S}| = 1$)
    ns = Num_sub


    ## Kernel for MOGP_AR
    kern = lmc_kernel(Whole_f, ks)

    ## Building Likelihood
    lik_list = []
    for k in range(num_output):
        lik_list.append(SoftmaxMOGP_AR())
    lik = SwitchedLikelihoodMOGP_AR(likelihood_list=lik_list,
                                    num_subsample_list=num_subsample,
                                    num_latent_f_list=num_latent_f, ns=ns)
    ## Building MOAP_AR
    MOGPAR_RBF_outputs = MOGPC_AR_More(kernel=kern, likelihood=lik, inducing_variable=iv, num_latent_gps=L,
                       k=num_latent_f, ns=ns, num_output=num_output,X_subratio = X_ratio,minibatch_eachoutput=minibatch)

    return MOGPAR_RBF_outputs

                    #####################################################################
                    ###### MOGP-AR with convolutional kernel for multi-outputs ##########
                    #####################################################################

def Build_MOGPAR_Conv_outputs(X, Num_U, Num_sub, Whole_f, num_subsample, num_latent_f, num_output,minibatch,patch1, patch2,Maximum_induc_patch):
    '''
    We build MOGPAR_Conv_outputs model for multi-output cases
    Args:
        Xtrain: the training data for input
        Ytrain: the training data for output
    Returns:
        MOGP_AR: The MOGPAR_Conv_outputs model
     '''

    ## The number of training data
    XX_list = []
    X_ratio = []
    for i in range(num_output):
        XX_list.append(X[i])
        X_ratio.append(X[i].shape[0]/minibatch) ## each datasize / minibatch
    XX = np.vstack(XX_list)

    ## Build our model
    ## The latent functions u
    ks = []
    for j in range(Num_U):
        ks.append(CSE(image_shape=[20, 20], patch_shape=[patch1, patch2]))

    ## Change the inducing input format

    L = len(ks)
    Z = np.unique(CSE(image_shape=[20, 20], patch_shape=[patch1, patch2]).get_patches(XX).numpy().reshape(-1, patch1 * patch2), axis=0)
    a = Z.shape[0]
    ### We choose maximum 1000 inducing data points
    if a > Maximum_induc_patch:
        a = Maximum_induc_patch
    else:
        a = a
    print('Inducing path size',a)
    idx = np.random.randint(Z.shape[0], size=a)
    Z_m = Z[idx, :].copy()
    Zs = [Z_m.copy() for _ in range(L)]
    iv_list = [InducingPatches(Z) for Z in Zs]
    iv = SeparateIndependentInducingVariables(iv_list)

    ## the number of subsampling class size ($|\mathbf{S}| = 1$)
    ns = Num_sub

    ## Kernel for MOGP_AR
    kern = lmc_kernel(Whole_f, ks)

    ## Building Likelihood
    lik_list = []
    for k in range(num_output):
        lik_list.append(SoftmaxMOGP_AR())
    lik = SwitchedLikelihoodMOGP_AR(likelihood_list=lik_list,
                                    num_subsample_list=num_subsample,
                                    num_latent_f_list=num_latent_f, ns=ns)
    ## Building MOAP_AR
    MOGPAR_Conv_outputs = MOGPC_AR_More(kernel=kern, likelihood=lik, inducing_variable=iv, num_latent_gps=L,
                       k=num_latent_f, ns=ns, num_output=num_output,X_subratio = X_ratio,minibatch_eachoutput=minibatch)

    return MOGPAR_Conv_outputs
