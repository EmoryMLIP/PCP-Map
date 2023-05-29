import numpy as np
from scipy.special import logsumexp


def calc_dist(stats_1, stats_2, ref_mean=None, ref_std=None):
    """ Calculates the Euclidean distance between two vectors of summary statistics."""
    # normalize stats
    if ref_mean is not None and ref_std is not None:
        stats_1 = (stats_1 - ref_mean) / ref_std
        stats_2 = (stats_2 - ref_mean) / ref_std
    return np.sqrt(np.sum((stats_1 - stats_2) ** 2))


def Rejection_ABC(simulator, obs_star, n_dim, n_particles=100, eps=1.0, ref_mean=None, ref_std=None):

    # sample initial population of parameters
    params    = np.empty([n_particles, n_dim])
    distances = np.zeros(n_particles)
    n_evals   = 0

    # find samples that are epsilon close
    for i in range(n_particles):
        dist = float('inf')
        while dist > eps:
            # simulate (x,y) from joint distribution and count total simulation calls
            params[i,:] = simulator.sample_prior(1,)
            try:
                obs_simulated = simulator.sample_data(params[i,:])
                n_evals += 1
            except:
                continue
            # compute distances
            dist = calc_dist(obs_simulated, obs_star, ref_mean, ref_std)
        distances[i] = dist

    return params, distances, n_evals


def SMC_ABC(simulator, obs_star, n_dim, n_particles=100,
            eps_init=10.0, eps_last=0.1, eps_decay=0.9, ess_min=0.5,
            logprior_bound_max=None, logprior_bound_min=None, ref_mean=None, ref_std=None):
    """Sequential Monte Carlo ABC given model simulator.
       The simulator must have sample_prior and sample_data methods"""

    # declare lists to store results
    all_params = []
    all_logweights = []
    all_eps = []
    all_nsims = []

    # initialize epsilon parameter
    eps = eps_init

    # sample parameters
    (params, _, n_evals) = Rejection_ABC(simulator, obs_star, n_dim,
        n_particles=n_particles, eps=eps, ref_mean=ref_mean, ref_std=ref_std)

    # set weights as uniform
    weights    = np.ones(n_particles, dtype=float) / n_particles
    logweights = np.log(weights)

    # save particles and weights
    all_params.append(params)
    all_logweights.append(logweights)
    all_eps.append(eps)
    all_nsims.append(n_evals)

    # increment epsilon
    round = 0
    while eps > eps_last:
        round += 1

        # decrease epsilon
        eps *= eps_decay

        # calculate population covariance for perturbations
        logparams = np.log(params)
        mean = np.mean(logparams, axis=0)
        cov = 2.0 * (np.dot(logparams.T, logparams) / n_particles - np.outer(mean, mean))
        std = np.linalg.cholesky(cov)

        # define arrays to store samples
        new_params = np.empty_like(params)
        new_logweights = np.empty_like(logweights)

        # find samples that are epsilon close
        for i in range(n_particles):
            dist = float('inf')
            while dist > eps:
                # perturb random parameter
                idx = np.random.choice(len(weights), 1, p=weights)
                new_params[i,:] = params[idx,:] * np.exp(np.dot(std, np.random.randn(n_dim)))
                # simulate observation
                try:
                    obs_simulated = simulator.sample_data(new_params[i,:])
                    n_evals += 1
                except:
                    continue
                # compute distance
                dist = calc_dist(obs_simulated, obs_star, ref_mean, ref_std)
            # evaluate weight
            new_logparams_i = np.log(new_params[i])
            logkernel = -0.5 * np.sum(np.linalg.solve(std, (new_logparams_i - logparams).T) ** 2, axis=0)
            if (logprior_bound_min is not None and np.any(new_logparams_i < logprior_bound_min)) or \
                (logprior_bound_max is not None and np.any(new_logparams_i > logprior_bound_max)):
                new_logweights[i] = -float('inf')
            else:
                new_logweights[i] = -logsumexp(logweights + logkernel)

        # normalize weights
        params = new_params
        logweights = new_logweights - logsumexp(new_logweights)
        weights = np.exp(logweights)

        # calculate effective sample size
        ess = 1.0 / (np.sum(weights ** 2) * n_particles)
        # resample particles if the ESS of the weighted ensemble is too low
        if ess < ess_min:
            new_params = np.empty_like(params)
            for i in range(n_particles):
                idx = np.random.choice(len(weights), 1, p=weights)
                new_params[i] = params[idx,:]
            # set uniform weights
            params  = new_params
            weights = np.ones(n_particles, dtype=float) / n_particles
            logweights = np.log(weights)

        print('Round %d: eps = %f, ESS before re-sampling %f' % (round, eps, ess))

        all_params.append(params)
        all_logweights.append(logweights)
        all_eps.append(eps)
        all_nsims.append(n_evals)

    # save results
    return (all_params, all_logweights, all_eps, all_nsims)
