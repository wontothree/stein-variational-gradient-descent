import numpy as np
from scipy.spatial.distance import pdist, squareform

class SVGD:
    def __init__(self):
        pass

    def svgd(self, particle_set, lnprob):
        """
        Stein Variational Gradient Descent (SVGD) Update

        Parameters:
        ----------
        particle_set : numpy array
            An array of shape (N, D) where N is the number of samples and D is the number of dimensions.

        lnprob : numpy array
            The log-probability gradients evaluated at particle_set.

        Returns:
        -------
        gradient_particle_set : numpy array
            The computed gradients for the particle set.
        """
        
        # pairwise euclidean distance matrix (1 x N(N-1)/2 matrix)
        pairwise_distance = pdist(particle_set)

        # square distance matrix (N x N matrix)
        pairwise_distance_matrix = squareform(pairwise_distance) ** 2

        # median of pairwise Euclidean distances (scalar)
        median_pairwise_distance = np.median(pairwise_distance_matrix)  

        # *** RBF kernel (Radial Basis Function kernel) ***
        bandwidth = np.sqrt(0.5 * median_pairwise_distance / np.log(particle_set.shape[0] + 1))
        rbf_kernel = np.exp(-pairwise_distance_matrix / bandwidth ** 2 / 2)

        # *** gradient for RBF kernel ***
        gradient_rbf_kernel = -np.matmul(rbf_kernel, particle_set)
        kernel_sums = np.sum(rbf_kernel, axis=1)
        for i in range(particle_set.shape[1]):
            gradient_rbf_kernel[:, i] = gradient_rbf_kernel[:, i] + np.multiply(particle_set[:, i], kernel_sums)
        gradient_rbf_kernel = gradient_rbf_kernel / (bandwidth ** 2)

        # gradient for particle set
        gradient_particle_set = (np.matmul(rbf_kernel, lnprob) + gradient_rbf_kernel) / particle_set.shape[0]

        return gradient_particle_set
    
    def update(self, initial_particle_set, lnprob, iteration=1000, stepsize=1e-3, momentum=0.9, debug=False):
        """
        Update the particle set using SVGD.

        Parameters:
        ----------
        initial_particle_set : numpy array

        lnprob : callable
            Function that calculates the log-probability gradient.

        iteration : int, optional (default=1000)
            The number of iterations to run the SVGD algorithm.

        stepsize : float, optional (default=1e-3)
            The step size for each iteration of SVGD.

        momentum : float, optional (default=0.9)
            Momentum parameter for the Adagrad optimization.

        debug : bool, optional (default=False)
            If True, prints the progress of the algorithm every 1000 iterations for debugging purposes.

        Returns:
        -------
        approximated_particle_set : numpy array
        """

        # Check input
        if initial_particle_set is None or lnprob is None:
            raise ValueError('Initial particle set or lnprob cannot be None!')
        
        particle_set = np.copy(initial_particle_set)

        # Adagrad with momentum
        fudge_factor = 1e-6
        historical_grad = np.zeros_like(particle_set)

        for iter in range(iteration):
            if debug and (iter + 1) % 1000 == 0:
                print('iteration ' + str(iter + 1))

            lnpgrad = lnprob(particle_set)  # Calculate log-probability gradient
            gradient_particle_set = self.svgd(particle_set, lnpgrad)  # Use the computed gradient
            
            # Adagrad
            historical_grad = momentum * historical_grad + (1 - momentum) * (gradient_particle_set ** 2)
            adj_grad = np.divide(gradient_particle_set, fudge_factor + np.sqrt(historical_grad))

            particle_set += stepsize * adj_grad 
            
        return particle_set
