                                #################################################
                                ### We build our own model based on GPflow. #####
                                #################################################

##### we import from gpflow and tensorflow #####
import tensorflow as tf
from gpflow.models.model import InputData, MeanAndVariance, RegressionData
from gpflow.models.svgp import SVGP
import numpy as np
from MOGP_convolutionalKernel.conditionals import lmc_conditional_mogp, lmc_conditional_singleogp
from typing import Callable, TypeVar, Union
OutputData = Union[tf.Tensor]
Data = TypeVar("Data", RegressionData, InputData, OutputData)
from tensorflow.python.data.ops.iterator_ops import OwnedIterator as DatasetOwnedIterator

                                        ##################################
                                        #### Building our model ##########
                                        ##################################

####################################################################################################################################
####################################################################################################################################
####################################################################################################################################

                                ##############################################################
                                ####### MOGPs-AR for multi-output prediciton model ###########
                                ##############################################################

class MOGPC_AR_More(SVGP):
    """
    The model is used when we build the MOGP_AR model.MOGPs-AR for multi-output prediciton model.
    """

    def __init__(self,k, ns,num_output,X_subratio,minibatch_eachoutput, **kwargs):
        super().__init__(**kwargs)
        self.k = k   ### the number of the latent funciton
        self.ns = ns ### the number of subsampling
        self.num_output = num_output
        self.X_subratio = X_subratio
        self.minibatch_eachoutput = minibatch_eachoutput

    def elbo(self, data: RegressionData) -> tf.Tensor:
        """
        This gives a variational bound (the evidence lower bound or ELBO) on
        the log marginal likelihood of the model.
        """
        X, Y = data
        kl = self.prior_kl()
        f_mean, f_var = self.predict_f(X, Y, full_cov=False, full_output_cov=False) ### We add Y here
        var_exp = self.likelihood.variational_expectations(f_mean, f_var, Y)
        var_exp_weight = tf.cast(tf.repeat(self.X_subratio, self.minibatch_eachoutput), kl.dtype) * var_exp
        return tf.reduce_sum(var_exp_weight) - kl

    def predict_f(self, Xnew: InputData,Ynew: tf.Tensor, full_cov=False, full_output_cov=False) -> MeanAndVariance:
        '''
        In this function, we can only deal with the minibatch case. If does not work for calculating for the ELBO for all the dataset
        '''
        indX = tf.cast(Xnew[..., -1], tf.int32)[:,None] ## We get the index for X. The index coresspond to the latent parameter function f.
        ind_output = tf.cast(Ynew[..., -1], tf.int32)[:,None] ### This is the index for different outputs.

        ## We partition indX based on the num_output
        argsX_ind = tf.dynamic_partition(indX, ind_output, self.num_output)
        ### the is the index for excited latent parameter function exculed the index0
        ind_lpf = [(ind[:, None] + 1 + range(K - 1)) % K for ind, K in zip(argsX_ind, self.k)]

        IndexF = [
            tf.concat([tf.map_fn(fn=lambda x:tf.transpose(tf.random.shuffle(tf.range(tf.shape(ind_lpfX)[1]))), elems=tf.range(tf.shape(ind_lpfX)[0]))
                       ], axis=0)[..., :n_s]
            for ind_lpfX, n_s in zip(ind_lpf, self.ns)
        ]

        ### the index dataset points
        index_data_points = [tf.range(tf.shape(ind_lpff)[0])[:, None] for ind_lpff in ind_lpf]

        ## combing the index for all dataset with the subsampling index for each latent paramerter
        index_sample = [tf.reshape(tf.transpose(Index_f), [-1, 1]) for Index_f in IndexF]
        index_data = [tf.tile(index_data_points_x, [n_s, 1]) for index_data_points_x, n_s in zip(index_data_points, self.ns)]
        ind_all_lpf = [tf.concat([index_data_x, index_sample_x], axis=1) for index_data_x, index_sample_x in
                       zip(index_data, index_sample)]
        ## all the index for each latent parameter function
        K = tf.split(tf.cumsum(self.k, exclusive=True),len(self.k))

        subsample_latent_parameter_function = [
            tf.concat([argsX_ind_x[:, None], tf.gather_nd(ind_lpf_x, ind_all_lpf_x)[:, None] + k_x], axis=0) for
            argsX_ind_x, ind_lpf_x, ind_all_lpf_x, k_x in zip(argsX_ind, ind_lpf, ind_all_lpf, K)]

        Xp = Xnew[..., :-1] ## This is training input

        ## We partition Xp based on the num_output
        Xp_index = tf.dynamic_partition(tf.range(0, tf.size(ind_output))[:,None], ind_output, self.num_output)
        Xp_ind = [tf.gather(Xp, Xp_index_x) for Xp_index_x in Xp_index]
        X_lpf = [tf.tile(Xp_ind_x, [n_s + 1, 1]) for Xp_ind_x, n_s in zip(Xp_ind, self.ns)]
        X_lpf_train = tf.concat([tf.concat(
            [tf.cast(X_lpf_x, tf.float64), tf.cast(subsample_latent_parameter_function_x, tf.float64)], axis=-1)
                                 for X_lpf_x, subsample_latent_parameter_function_x in
                                 zip(X_lpf, subsample_latent_parameter_function)
                                 ], axis=0)

        q_mu = self.q_mu
        q_sqrt = self.q_sqrt
        mu, var = lmc_conditional_mogp(
            X_lpf_train,
            self.inducing_variable,
            self.kernel,
            q_mu,
            q_sqrt=q_sqrt,
            full_cov=full_cov,
            white=self.whiten,
            full_output_cov=full_output_cov,
        )
        return mu + self.mean_function(X_lpf_train), var

    def predict_y_categorical(self,
                  Xnew: InputData,
                  full_cov: bool = False,
                  full_output_cov: bool = False) -> MeanAndVariance:
        """
        Compute the mean and variance of the held-out data at the input points.
        """
        ### We change the X into our format X_lpf
        num_task = len(self.likelihood.likelihoods)
        ind = tf.cast(Xnew[..., -1], tf.int32)
        Xnew = Xnew[..., :-1]
        X_lpf = tf.dynamic_partition(Xnew, ind, num_task) ## ### partition of input for each output
        num_data = [tf.shape(Xi)[0] for Xi in X_lpf] ## number of data for each output
        num_data = tf.repeat(num_data, self.likelihood.num_latent_f_list)
        number_point_task = [tf.shape(tf.tile(Xi, [l, 1]))[0] for Xi, l in zip(X_lpf, self.likelihood.num_latent_f_list)] ## The number of data for each output after we augment for each output
        X_lpf = tf.concat(
            [tf.tile(Xi, [l, 1]) for Xi, l in zip(X_lpf, self.likelihood.num_latent_f_list)], axis=0
        ) ## Each task corresponding to each latent parameter function f
        ind_lpf = tf.repeat(tf.range(self.likelihood.num_latent_f_sum, dtype=Xnew.dtype), num_data)[
                  :, None
                  ] ## This is the index for latent parameter functions
        X_lpf = tf.concat([X_lpf, ind_lpf], axis=-1) ## We consider all the input X with the index for latent parameter function


        q_mu = self.q_mu
        q_sqrt = self.q_sqrt
        mu, var = lmc_conditional_mogp(
            X_lpf,
            self.inducing_variable,
            self.kernel,
            q_mu,
            q_sqrt=q_sqrt,
            full_cov=full_cov,
            white=self.whiten,
            full_output_cov=full_output_cov,
        )
        # return mu + self.mean_function(X_lpf), var
        ### we got mu and var for all the latent functions and parameter functions
        f_mean_total = mu + self.mean_function(X_lpf)
        f_var_total = var

        ## We want to divide the whole mean and variance funtion for each task
        Y_mean = []
        Y_var = []
        ## We get the number of the point index for each task
        number_point_task_index = np.append(0, np.cumsum(number_point_task))
        ## We calculate mean and variance for each task
        for i in range(num_task):
            f_mean = f_mean_total[number_point_task_index[i]:number_point_task_index[i+1]]
            f_var = f_var_total[number_point_task_index[i]:number_point_task_index[i+1]]
            N = f_mean.shape[0]
            m = tf.dtypes.cast(N/self.likelihood.num_latent_f_list[i], tf.int64)
            Fmu = tf.transpose(tf.reshape(f_mean, [self.likelihood.num_latent_f_list[i], m]))
            Fvar = tf.transpose(tf.reshape(f_var, [self.likelihood.num_latent_f_list[i], m]))
            mean_y, var_y = self.likelihood.likelihoods[i].predict_mean_and_var(Fmu=Fmu, Fvar=Fvar,Num_latent_function=self.likelihood.num_latent_f_list[i])
            Y_mean.append(mean_y)
            Y_var.append(var_y)
        return Y_mean, Y_var

    def predict_y_one_output(self,Task,
                  Xnew: InputData,
                  full_cov: bool = False,
                  full_output_cov: bool = False) -> MeanAndVariance:
        """
        Compute the mean and variance of the held-out data at the input points.
        """
        ### We change the X into our format X_lpf

        ind = tf.cast(Xnew[..., -1], tf.int32)
        Xnew = Xnew[..., :-1]
        num_data = tf.shape(Xnew)[0]
        num_data = tf.repeat(num_data, self.likelihood.num_latent_f_list[Task])
        X_lpf = tf.tile(Xnew, [self.likelihood.num_latent_f_list[Task], 1]) ## We copy Xnew for each corresponded latent parametter function.
        Cumcum_index_f = np.append(0,np.cumsum(self.likelihood.num_latent_f_list))
        ## Each task corresponding to each latent parameter function f for the index.
        # ## We new index for the X in the, e.g The first outputs  is 6 outputs and the second is 7 outputs. The second index for f will be 6,7,8,9,10,11,12
        ind_lpf = tf.repeat(tf.range(Cumcum_index_f[Task],Cumcum_index_f[Task+ 1], dtype=Xnew.dtype), num_data)[
                  :, None
                  ] ## This is the index for latent parameter functions
        X_lpf = tf.concat([X_lpf, ind_lpf], axis=-1) ## We consider all the input X with the index for latent parameter function

        q_mu = self.q_mu
        q_sqrt = self.q_sqrt
        mu, var = lmc_conditional_mogp(
            X_lpf,
            self.inducing_variable,
            self.kernel,
            q_mu,
            q_sqrt=q_sqrt,
            full_cov=full_cov,
            white=self.whiten,
            full_output_cov=full_output_cov,
        )
        # tf.debugging.assert_positive(var)  # We really should make the tests pass with this here
        # return mu + self.mean_function(X_lpf), var
        ### we got mu and var for all the latent functions and parameter functions
        f_mean = mu + self.mean_function(X_lpf)
        f_var = var
        N = f_mean.shape[0]
        ## We tranfor the vector shape of the latent parameter function into matrix shape
        m = tf.dtypes.cast(N/self.likelihood.num_latent_f_list[Task], tf.int64)
        Fmu = tf.transpose(tf.reshape(f_mean, [self.likelihood.num_latent_f_list[Task], m]))
        Fvar = tf.transpose(tf.reshape(f_var, [self.likelihood.num_latent_f_list[Task], m]))
        ## We calculate the mean the variance for predictive mean and variance
        mean_y, var_y = self.likelihood.likelihoods[Task].predict_mean_and_var(Fmu=Fmu, Fvar=Fvar,Num_latent_function=self.likelihood.num_latent_f_list[Task])
        return mean_y, var_y

    def training_loss(self, data: Data) -> tf.Tensor:
        """
        Returns the training loss for this model.

        :param data: the data to be used for computing the model objective.
        """
        return self._training_loss(data)

    def training_loss_closure(
            self, data: Union[Data, DatasetOwnedIterator], *, compile=True,
    ) -> Callable[[], tf.Tensor]:

        """
        Returns a closure that computes the training loss, which by default is
        wrapped in tf.function(). This can be disabled by passing `compile=False`.

        :param data: the data to be used by the closure for computing the model
        objective. Can be the full dataset or an iterator, e.g.
        `iter(dataset.batch(batch_size))`, where dataset is an instance of
        tf.data.Dataset.
        :param compile: if True, wrap training loss in tf.function()
            """
        training_loss = self.training_loss
        if isinstance(data, DatasetOwnedIterator):
            if compile:
                # input_signature = [data.element_spec]
                # training_loss = tf.function(training_loss, input_signature=input_signature)
                training_loss = tf.function(training_loss)
            def closure():
                batch = next(data)


                ## The batch_combine is that we combine all the bath in different outputs together
                batch_combine = batch[0]
                for i in range(self.num_output-1):
                    batch_combine = batch_combine + batch[i+1]
                ## We combine the all the batch for different outputs together in this format (X,Y), where
                ## X is a matrix that include all input for all outputs
                ## Y is a matrix that include all label for all outputs
                New_batch = (tf.concat((batch_combine[::2]), 0), tf.concat((batch_combine[1::2]), 0))
                return training_loss(New_batch)
        else:
            def closure():
                return training_loss(data)

            if compile:
                closure = tf.function(closure)
        return closure

####################################################################################################################################
####################################################################################################################################
####################################################################################################################################

                                    ###########################################################
                                    ####### MOGPs for one ouptut data prediction model ########
                                    ###########################################################

class MOGP_Categorical(SVGP):
    """
    The model is used when we used the categorcial distribution;
    We making a SVGP for Softmatx
    We need to make sure that the first likelihood is categorical likelihood
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def elbo(self, data: RegressionData) -> tf.Tensor:
        """
        This gives a variational bound (the evidence lower bound or ELBO) on
        the log marginal likelihood of the model.
        """
        X, Y = data
        kl = self.prior_kl()
        f_mean, f_var = self.predict_f(X, full_cov=False, full_output_cov=False)
        var_exp = self.likelihood.variational_expectations(f_mean, f_var, Y)
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl.dtype)
        return tf.reduce_sum(var_exp) * scale - kl

    def predict_f(self, Xnew: InputData, full_cov=False, full_output_cov=False) -> MeanAndVariance:
        num_task = len(self.likelihood.likelihoods)
        ind = tf.cast(Xnew[..., -1], tf.int32)
        Xnew = Xnew[..., :-1]
        X_lpf = tf.dynamic_partition(Xnew, ind, num_task)
        num_data = [tf.shape(Xi)[0] for Xi in X_lpf]
        num_data = tf.repeat(num_data, self.likelihood.num_latent_list)
        X_lpf = tf.concat(
            [tf.tile(Xi, [l, 1]) for Xi, l in zip(X_lpf, self.likelihood.num_latent_list)], axis=0
        )
        ind_lpf = tf.repeat(tf.range(self.likelihood.num_latent, dtype=Xnew.dtype), num_data)[
                  :, None
                  ]
        X_lpf = tf.concat([X_lpf, ind_lpf], axis=-1)
        q_mu = self.q_mu
        q_sqrt = self.q_sqrt
        mu, var = lmc_conditional_mogp(
            X_lpf,
            self.inducing_variable,
            self.kernel,
            q_mu,
            q_sqrt=q_sqrt,
            full_cov=full_cov,
            white=self.whiten,
            full_output_cov=full_output_cov,
        )
        return mu + self.mean_function(X_lpf), var

    def predict_y_categorical(self,
                  Xnew: InputData,
                  full_cov: bool = False,
                  full_output_cov: bool = False) -> MeanAndVariance:
        """
        Compute the mean and variance of the held-out data at the input points.
        """
        num_task = len(self.likelihood.likelihoods)
        ind = tf.cast(Xnew[..., -1], tf.int32)
        Xnew = Xnew[..., :-1]
        X_lpf = tf.dynamic_partition(Xnew, ind, num_task)
        num_data = [tf.shape(Xi)[0] for Xi in X_lpf]
        num_data = tf.repeat(num_data, self.likelihood.num_latent_list)
        number_point_task = [tf.shape(tf.tile(Xi, [l, 1]))[0] for Xi, l in zip(X_lpf, self.likelihood.num_latent_list)] ## The number of data for each output after we augment for each output
        X_lpf = tf.concat(
            [tf.tile(Xi, [l, 1]) for Xi, l in zip(X_lpf, self.likelihood.num_latent_list)], axis=0
        )
        ind_lpf = tf.repeat(tf.range(self.likelihood.num_latent, dtype=Xnew.dtype), num_data)[
                  :, None
                  ]
        X_lpf = tf.concat([X_lpf, ind_lpf], axis=-1)

        q_mu = self.q_mu
        q_sqrt = self.q_sqrt
        mu, var = lmc_conditional_mogp(
            X_lpf,
            self.inducing_variable,
            self.kernel,
            q_mu,
            q_sqrt=q_sqrt,
            full_cov=full_cov,
            white=self.whiten,
            full_output_cov=full_output_cov,
        )
        # return mu + self.mean_function(X_lpf), var
        ### we got mu and var for all the latent functions and parameter functions
        f_mean_total = mu + self.mean_function(X_lpf)
        f_var_total = var

        ## We want to divide the whole mean and variance funtion for each task
        Y_mean = []
        Y_var = []
        ## We get the number of the point index for each task
        number_point_task_index = np.append(0, np.cumsum(number_point_task))
        ## We calculate mean and variance for each task
        for i in range(num_task):
            f_mean = f_mean_total[number_point_task_index[i]:number_point_task_index[i+1]]
            f_var = f_var_total[number_point_task_index[i]:number_point_task_index[i+1]]
            N = f_mean.shape[0]
            m = tf.dtypes.cast(N/self.likelihood.num_latent_list[i], tf.int64)
            Fmu = tf.transpose(tf.reshape(f_mean, [self.likelihood.num_latent_list[i], m]))
            Fvar = tf.transpose(tf.reshape(f_var, [self.likelihood.num_latent_list[i], m]))
            mean_y, var_y = self.likelihood.likelihoods[i].predict_mean_and_var(Fmu=Fmu, Fvar=Fvar)
            Y_mean.append(mean_y)
            Y_var.append(var_y)
        return Y_mean, Y_var

                                        #########################################################
                                        ####### MOGPs-AR for one output prediction model ########
                                        #########################################################

class MOGPC_AR(SVGP):
    """
    The model is used when we build the MOGP_AR model. MOGPs-AR for one output prediction model
    """
    def __init__(self,k, ns,minibatch,num_output, **kwargs):
        super().__init__(**kwargs)
        self.k = k   ### the number of the latent funciton
        self.ns = ns ### the number of subsampling
        self.minibatch = minibatch
        self.num_output = num_output
    def elbo(self, data: RegressionData) -> tf.Tensor:
        """
        This gives a variational bound (the evidence lower bound or ELBO) on
        the log marginal likelihood of the model.
        """
        X, Y = data
        kl = self.prior_kl()
        f_mean, f_var = self.predict_f(X, Y, full_cov=False, full_output_cov=False) ### We add Y here
        var_exp = self.likelihood.variational_expectations(f_mean, f_var, Y)
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl.dtype)
        return tf.reduce_sum(var_exp) * scale - kl

    def predict_f(self, Xnew: InputData,Ynew: tf.Tensor, full_cov=False, full_output_cov=False) -> MeanAndVariance:
        '''
        In this function, we can only deal with the minibatch case. If does not work for calculating for the ELBO for all the dataset
        '''
        indX = tf.cast(Xnew[..., -1], tf.int32)[:,None] ## We get the index for X. The index coresspond to the latent parameter function f.
        ind_output = tf.cast(Ynew[..., -1], tf.int32)[:,None] ### This is the index for different outputs.

        ## We partition indX based on the num_output
        argsX_ind = tf.dynamic_partition(indX, ind_output, self.num_output)

        ##########################################################################################
        ####### The below we subsample latent parameter functions for single-output GP ###########
        ##########################################################################################

        ### the is the index for excited latent parameter function exculed the index0
        ind_lpf = [(ind[:, None] + 1 + range(K - 1)) % K for ind, K in zip(argsX_ind, self.k)]

        ### subsamling for each latent parameter function
        ## subsample index for each latent parameter
        IndexF = [
            tf.concat([tf.map_fn(fn=lambda x:tf.transpose(tf.random.shuffle(tf.range(tf.shape(ind_lpfX)[1]))), elems=tf.range(tf.shape(ind_lpfX)[0]))
                       ], axis=0)[..., :n_s]
            for ind_lpfX, n_s in zip(ind_lpf, self.ns)
        ]

        ### the index dataset points
        index_data_points = [tf.range(tf.shape(ind_lpff)[0])[:, None] for ind_lpff in ind_lpf]

        ## combing the index for all dataset with the subsampling index for each latent paramerter
        index_sample = [tf.reshape(tf.transpose(Index_f), [-1, 1]) for Index_f in IndexF]
        index_data = [tf.tile(index_data_points_x, [n_s, 1]) for index_data_points_x, n_s in zip(index_data_points, self.ns)]
        ind_all_lpf = [tf.concat([index_data_x, index_sample_x], axis=1) for index_data_x, index_sample_x in
                       zip(index_data, index_sample)]

        ## all the index for each latent parameter function
        K = tf.split(tf.cumsum(self.k, exclusive=True),len(self.k))

        subsample_latent_parameter_function = [
            tf.concat([argsX_ind_x[:, None], tf.gather_nd(ind_lpf_x, ind_all_lpf_x)[:, None] + k_x], axis=0) for
            argsX_ind_x, ind_lpf_x, ind_all_lpf_x, k_x in zip(argsX_ind, ind_lpf, ind_all_lpf, K)]

        Xp = Xnew[..., :-1] ## This is training input

        ## We partition Xp based on the num_output
        Xp_index = tf.dynamic_partition(tf.range(0, tf.size(ind_output))[:,None], ind_output, self.num_output)
        Xp_ind = [tf.gather(Xp, Xp_index_x) for Xp_index_x in Xp_index]
        X_lpf = [tf.tile(Xp_ind_x, [n_s + 1, 1]) for Xp_ind_x, n_s in zip(Xp_ind, self.ns)]

        X_lpf_train = tf.concat([tf.concat(
            [tf.cast(X_lpf_x, tf.float64), tf.cast(subsample_latent_parameter_function_x, tf.float64)], axis=-1)
                                 for X_lpf_x, subsample_latent_parameter_function_x in
                                 zip(X_lpf, subsample_latent_parameter_function)
                                 ], axis=0)

        q_mu = self.q_mu
        q_sqrt = self.q_sqrt
        mu, var = lmc_conditional_mogp(
            X_lpf_train,
            self.inducing_variable,
            self.kernel,
            q_mu,
            q_sqrt=q_sqrt,
            full_cov=full_cov,
            white=self.whiten,
            full_output_cov=full_output_cov,
        )
        # tf.debugging.assert_positive(var)  # We really should make the tests pass with this here
        return mu + self.mean_function(X_lpf_train), var

    def predict_y_categorical(self,
                  Xnew: InputData,
                  full_cov: bool = False,
                  full_output_cov: bool = False) -> MeanAndVariance:
        """
        Compute the mean and variance of the held-out data at the input points.
        """
        ### We change the X into our format X_lpf
        num_task = len(self.likelihood.likelihoods)
        ind = tf.cast(Xnew[..., -1], tf.int32)
        Xnew = Xnew[..., :-1]
        X_lpf = tf.dynamic_partition(Xnew, ind, num_task) ## ### partition of input for each output
        num_data = [tf.shape(Xi)[0] for Xi in X_lpf] ## number of data for each output
        num_data = tf.repeat(num_data, self.likelihood.num_latent_f_list)
        number_point_task = [tf.shape(tf.tile(Xi, [l, 1]))[0] for Xi, l in zip(X_lpf, self.likelihood.num_latent_f_list)] ## The number of data for each output after we augment for each output
        X_lpf = tf.concat(
            [tf.tile(Xi, [l, 1]) for Xi, l in zip(X_lpf, self.likelihood.num_latent_f_list)], axis=0
        ) ## Each task corresponding to each latent parameter function f
        ind_lpf = tf.repeat(tf.range(self.likelihood.num_latent_f_sum, dtype=Xnew.dtype), num_data)[
                  :, None
                  ] ## This is the index for latent parameter functions
        X_lpf = tf.concat([X_lpf, ind_lpf], axis=-1) ## We consider all the input X with the index for latent parameter function

        q_mu = self.q_mu
        q_sqrt = self.q_sqrt
        mu, var = lmc_conditional_mogp(
            X_lpf,
            self.inducing_variable,
            self.kernel,
            q_mu,
            q_sqrt=q_sqrt,
            full_cov=full_cov,
            white=self.whiten,
            full_output_cov=full_output_cov,
        )
        ### we got mu and var for all the latent functions and parameter functions
        f_mean_total = mu + self.mean_function(X_lpf)
        f_var_total = var

        ## We want to divide the whole mean and variance funtion for each task
        Y_mean = []
        Y_var = []
        ## We get the number of the point index for each task
        number_point_task_index = np.append(0, np.cumsum(number_point_task))
        ## We calculate mean and variance for each task
        for i in range(num_task):
            f_mean = f_mean_total[number_point_task_index[i]:number_point_task_index[i+1]]
            f_var = f_var_total[number_point_task_index[i]:number_point_task_index[i+1]]
            N = f_mean.shape[0]
            m = tf.dtypes.cast(N/self.likelihood.num_latent_f_list[i], tf.int64)
            Fmu = tf.transpose(tf.reshape(f_mean, [self.likelihood.num_latent_f_list[i], m]))
            Fvar = tf.transpose(tf.reshape(f_var, [self.likelihood.num_latent_f_list[i], m]))
            mean_y, var_y = self.likelihood.likelihoods[i].predict_mean_and_var(Fmu=Fmu, Fvar=Fvar,Num_latent_function=self.likelihood.num_latent_f_list[i])
            Y_mean.append(mean_y)
            Y_var.append(var_y)
        return Y_mean, Y_var

                                ###########################################################
                                #### Single output GP using convolutional kernel ##########
                                ###########################################################

class SVGP_convolution(SVGP):
    """
    The function of this model is exact same as svgp. We build this model beacause we want to use inducing patch in our own version.
    The different between SVGP_convolution and SVGP is that we use lmc_conditional_singlegp. After that we can use the own version of covariance
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def predict_f(self, Xnew: InputData, full_cov=False, full_output_cov=False) -> MeanAndVariance:
        q_mu = self.q_mu
        q_sqrt = self.q_sqrt
        mu, var = lmc_conditional_singleogp(
            Xnew,
            self.inducing_variable,
            self.kernel,
            q_mu,
            q_sqrt=q_sqrt,
            full_cov=full_cov,
            white=self.whiten,
            full_output_cov=full_output_cov,
        )
        # tf.debugging.assert_positive(var)  # We really should make the tests pass with this here
        return mu + self.mean_function(Xnew), var