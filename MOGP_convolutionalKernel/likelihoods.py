                        #####################################################
                        ### We build our own likelihood based on GPflow.#####
                        #####################################################
##### we import from gpflow and tensorflow #####
import tensorflow as tf
import numpy as np
from gpflow.likelihoods import ScalarLikelihood, SwitchedLikelihood,Softmax
from MOGP_convolutionalKernel.utils import ndiag_mc_updated


                            #####################################
                            #### Building new likelihood ########
                            #####################################

##################################################################################################################
##################################################################################################################
##################################################################################################################

                            #####################################
                            ####            GPC             #####
                            #####################################

class Softmax_mogp(Softmax):
    """
    This is Softmax likelhood that is similar to the Softmax likelihood in the gpflow.
    The only difference is that we change the ndiag_mc_updated in order to make it complile in minibatch format in SVGP.
    Softmax_mogp has the same output as Softmax.
    """
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.num_monte_carlo_points=100


    def mc_quadrature(self, funcs, Fmu, Fvar, logspace: bool = False, epsilon=None, **Ys):
        return ndiag_mc_updated(funcs, self.num_monte_carlo_points, Fmu, Fvar, logspace,epsilon, **Ys)

    def variational_expectations(self, Fmu, Fvar, Y, epsilon=None):

        return tf.reduce_sum(
            self.mc_quadrature(self.log_prob,
                                   Fmu,
                                   Fvar,
                                   Y=Y,
                                   epsilon=epsilon), axis=-1
        )


##################################################################################################################
##################################################################################################################
##################################################################################################################

                        ######################################
                        ####            GP-AR         ########
                        ######################################
# This likelihood is from Liu paper
# Link : https://github.com/LiuHaiTao01/GPCnoise
# "Scalable Gaussian process classification with additive noise for various likelihoods"
class MultiClass_SoftMax_Aug(ScalarLikelihood):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes

    def variational_expectations(self, Fmu, Fvar, Y):
        Y = tf.cast(Y, tf.int32)
        oh_on = tf.cast(tf.one_hot(tf.reshape(Y, (-1,)), self.num_classes, 1., 0.),
                        tf.float64)  # (lht): one-hot encode
        oh_off = tf.cast(tf.one_hot(tf.reshape(Y, (-1,)), self.num_classes, 0., 1.), tf.float64)
        Fmu_selected = tf.reduce_sum(oh_on * Fmu, 1)  # (lht): oh_on * mu: element-wsie dot of on_on and mu
        Fvar_selected = tf.reduce_sum(oh_on * Fvar, 1)
        P = tf.exp(0.5 * Fvar_selected - Fmu_selected) * tf.reduce_sum(tf.exp(0.5 * Fvar + Fmu) * oh_off, 1)  # (N,)
        ve = - tf.math.log(1. + P)
        return ve

    def predict_mean_and_var(self, Fmu, Fvar):
        # MC solution
        np.random.seed(1)
        N_sample = 1000
        u = np.random.randn(N_sample, self.num_classes)  # N_sample x C
        u_3D = tf.tile(tf.expand_dims(u, 1), [1, tf.shape(Fmu)[0], 1])  # N_sample x N* x C
        Fmu_3D = tf.tile(tf.expand_dims(Fmu, 0), [N_sample, 1, 1])  # N_sample x N* x C
        Fvar_3D = tf.tile(tf.expand_dims(Fvar, 0), [N_sample, 1, 1])  # N_sample x N* x C
        exp_term = tf.exp(
            Fmu_3D + tf.sqrt(Fvar) * u_3D)  # mu_3D + tf.sqrt(Fvar) * u_3D are samples from Gaussian distribution
        exp_sum_term = tf.tile(tf.expand_dims(tf.reduce_sum(exp_term, -1), 2), [1, 1, self.num_classes])
        ps = tf.reduce_sum(exp_term / exp_sum_term, 0) / N_sample
        vs = tf.reduce_sum(tf.square(exp_term / exp_sum_term), 0) / N_sample - tf.square(ps)
        return ps, vs

    def _scalar_log_prob(self):
        pass

##################################################################################################################
##################################################################################################################
##################################################################################################################

                        #####################################
                        ####            MOGP         ########
                        #####################################
class SwitchedLikelihood_MOGP(SwitchedLikelihood):
    def __init__(self, likelihood_list, num_latent_list, **kwargs):
        self.likelihoods = likelihood_list
        self.num_latent_list = num_latent_list
        self.num_latent = sum(num_latent_list)
        self.num_task = len(likelihood_list)

    def _partition_and_stitch(self, args, func_name):
        """
        args is a list of tensors, to be passed to self.likelihoods.<func_name>

        args[-1] is the 'Y' argument, which contains the indexes to self.likelihoods.

        This function splits up the args using dynamic_partition, calls the
        relevant function on the likelihoods, and re-combines the result.
        """
        # get the index from Y
        Y = args[-1]
        ind = Y[..., -1]
        ind = tf.cast(ind, tf.int32)
        num_data = tf.math.bincount(ind, minlength=self.num_task)
        num_data = tf.repeat(num_data, self.num_latent_list)
        ind_task = tf.repeat(tf.range(self.num_task), self.num_latent_list)
        ind_task = tf.repeat(ind_task, num_data)
        Y = Y[..., :-1]
        args = args[:-1]


        # split up the arguments into chunks corresponding to the relevant likelihoods
        args = [tf.dynamic_partition(X, ind_task, self.num_task) for X in args]
        args = [
            [
                tf.transpose(tf.reshape(f_t, [n_latent, -1]))
                for f_t, n_latent in zip(arg, self.num_latent_list)
            ]
            for arg in args
        ]
        #
        args = zip(*args)
        arg_Y = tf.dynamic_partition(Y, ind, self.num_task)

        # # apply the likelihood-function to each section of the data
        funcs = [getattr(lik, func_name) for lik in self.likelihoods]
        results = [f(*args_i, Yi) for f, args_i, Yi in zip(funcs, args, arg_Y)]


        # stitch the results back together
        partitions = tf.dynamic_partition(tf.range(0, tf.size(ind)), ind, self.num_task)
        results = tf.dynamic_stitch(partitions, results)
        return results


    def variational_expectations(self, Fmu, Fvar, Y):
        return self._partition_and_stitch([Fmu, Fvar, Y], "variational_expectations")

##################################################################################################################
##################################################################################################################
##################################################################################################################

                        #####################################
                        ####          MOGP-AR        ########
                        #####################################

class SwitchedLikelihoodMOGP_AR(SwitchedLikelihood):
    def __init__(self, likelihood_list, num_subsample_list, num_latent_f_list,ns,**kwargs):
        '''
        :param likelihood_list: It is a list that include different Soft_AR likelhood
        :param num_subsample_list: (a list).The number of sample function (exited latent function (1) + the sampled latent functions (ns))
        :param num_latent_f_list: A list that inculdes the number of latent parameter function in each likelihood.
        '''
        self.likelihoods = likelihood_list
        self.num_subsample_list = num_subsample_list
        self.num_outputs = len(likelihood_list)
        self.num_latent_f_list = num_latent_f_list
        self.num_latent_f_sum = sum(num_latent_f_list)
        self.ns = ns

    def _partition_and_stitch(self, args, func_name):
        """
        args is a list of tensors, to be passed to self.likelihoods.<func_name>

        args[-1] is the 'Y' argument, which contains the indexes to self.likelihoods.

        This function splits up the args using dynamic_partition, calls the
        relevant function on the likelihoods, and re-combines the result.
        """
        # get the index from Y
        Y = args[-1]
        ind = Y[..., -1]
        ind = tf.cast(ind, tf.int32)
        num_data = tf.math.bincount(ind, minlength=self.num_outputs)
        num_data = tf.repeat(num_data, self.num_subsample_list)
        ind_task = tf.repeat(tf.range(self.num_outputs), self.num_subsample_list)
        ind_task = tf.repeat(ind_task, num_data)
        Y = Y[..., :-1]
        args = args[:-1]


        # split up the arguments into chunks corresponding to the relevant likelihoods
        args = [tf.dynamic_partition(X, ind_task, self.num_outputs) for X in args]

        ## The chang one dimension data into ns dimension
        ## E.g f_1 = [f1,f1,f2,f2] = [[f1 f2],[f1 f2]]
        args = [
            [
                tf.transpose(tf.reshape(f_t, [n_latent, -1]))
                for f_t, n_latent in zip(arg, self.num_subsample_list)
            ]
            for arg in args
        ]
        args = zip(*args)

        ### split up the Y
        arg_Y = tf.dynamic_partition(Y, ind, self.num_outputs)

        # # apply the likelihood-function to each section of the data
        funcs = [getattr(lik, func_name) for lik in self.likelihoods]
        results = [f(*args_i, Yi,n_s,total_f) for f, args_i, Yi, n_s, total_f in zip(funcs, args, arg_Y,self.ns ,self.num_latent_f_list)]


        # stitch the results back together
        partitions = tf.dynamic_partition(tf.range(0, tf.size(ind)), ind, self.num_outputs)
        results = tf.dynamic_stitch(partitions, results)

        return results


    def variational_expectations(self, Fmu, Fvar, Y):
        return self._partition_and_stitch([Fmu, Fvar, Y], "variational_expectations")



### Softmax likelihood with AR method for single output
class SoftmaxMOGP_AR(ScalarLikelihood):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def variational_expectations(self, Fmu, Fvar, Y, ns, Num_latent_function):

        ## Building the oh_on and oh_off is used to deleted the un-sampled latent parameter functions
        oh_on = tf.concat([tf.ones_like(Fmu[:, 0][:, None]), tf.zeros_like(Fmu[:, 1:])], 1)
        oh_off = tf.concat([tf.zeros_like(Fmu[:, 0][:, None]), tf.ones_like(Fmu[:, 1:])], 1)
        oh_on = tf.cast(oh_on, tf.float64)
        oh_off = tf.cast(oh_off, tf.float64)

        ## select the relevent Fmu and F_var
        Fmu_selected = tf.reduce_sum(oh_on * Fmu, 1)
        Fvar_selected = tf.reduce_sum(oh_on * Fvar, 1)

        ## This is used for the unbiased the  estimator
        sviFactorClasses = (Num_latent_function-1)/ ns
        ## calculate the variational_expectations
        P = tf.exp(0.5 * Fvar_selected - Fmu_selected) * tf.reduce_sum(tf.exp(0.5 * Fvar  + Fmu ) * oh_off, 1) * sviFactorClasses  # (N,)
        ve = - tf.math.log(1. + P)
        return ve

    def predict_mean_and_var(self, Fmu, Fvar, Num_latent_function):
        # MC solution. We use the idear from Scalable Gaussian Process Classification with Additive Noise for Various Likelihoods by Haotao
        np.random.seed(1)
        N_sample = 1000
        u = np.random.randn(N_sample, Num_latent_function)  # N_sample x C
        u_3D = tf.tile(tf.expand_dims(u, 1), [1, tf.shape(Fmu)[0], 1])  # N_sample x N* x C
        Fmu_3D = tf.tile(tf.expand_dims(Fmu, 0), [N_sample, 1, 1])  # N_sample x N* x C
        Fvar_3D = tf.tile(tf.expand_dims(Fvar, 0), [N_sample, 1, 1])  # N_sample x N* x C
        exp_term = tf.exp(
            Fmu_3D + tf.sqrt(Fvar) * u_3D)  # mu_3D + tf.sqrt(Fvar) * u_3D are samples from Gaussian distribution
        exp_sum_term = tf.tile(tf.expand_dims(tf.reduce_sum(exp_term, -1), 2), [1, 1, Num_latent_function])
        ps = tf.reduce_sum(exp_term / exp_sum_term, 0) / N_sample
        vs = tf.reduce_sum(tf.square(exp_term / exp_sum_term), 0) / N_sample - tf.square(ps)
        return ps, vs

    def _scalar_log_prob(self):
        pass