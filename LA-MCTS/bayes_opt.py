import scipy.optimize
from scipy.stats import norm as normal
from diversipy import polytope
import sklearn.gaussian_process as gp
from scripts.optimization import vacc
import numpy as np
import pandas as pd
import multiprocess
import hopsy

#### Acquisition functions for Bayesian optimization #####
def neg_EI(x, GP, f_n_star, xi=0.1):
    # not completely working??
    pred_mu, pred_sd = GP.predict(x.reshape(1,-1), return_std=True)
    #print(pred_mu,pred_sd)
    delta_x = pred_mu - f_n_star - xi
    Z = delta_x / pred_sd
    #print("delta",delta_x,"mu",pred_mu,"sd",pred_sd)
    #EI_val = max(0,delta_x) + pred_sd*normal.pdf(delta_x/pred_sd) - np.abs(delta_x)*normal.cdf(delta_x/pred_sd)
    EI_val = delta_x*normal.cdf(Z) + pred_sd*normal.pdf(Z)
    print(EI_val)
    return -EI_val

def UCB(x, GP, beta):
    pred_x = GP.predict(x.reshape(1, -1), return_std=True)
    pred_mu = pred_x[0][0]
    pred_sd = pred_x[1][0]
    return (pred_mu + beta*pred_sd)

#### Constrained sampling from polytope #####
def constraint_sample(n,dim,V_0,P,c):
    samples = polytope.sample(
            n_points=n,
            lower=np.zeros(dim),
            upper=V_0,
            A2=np.array([P]),
            b2=np.array([c*np.sum(P)])
    )
    return samples

class Uniform_Model:
    """
    Defines a uniform distribution on the hyper-rectangle
    lb_vec <= x <= ub_vec.
    Needed for Hopsy based polytope sampling.
    """
    def __init__(self, dims,lb_vec,ub_vec):
        assert len(lb_vec) == len(ub_vec) == dims
        self.ub_vec = np.array(ub_vec)
        self.lb_vec = np.array(lb_vec)
        self.volume =  np.prod(self.ub_vec - self.lb_vec)
        self.nll = -np.log(1/self.volume)
    def compute_negative_log_likelihood(self, x):
        if np.all(self.lb_vec <= x) and np.all(x <= self.ub_vec):
            return self.nll
        else:
            return np.inf

def constraint_sample_hopsy(n,dim,V_0,P,c, thinning=2, n_threads=2):
    model = Uniform_Model(dims=dim,
                          lb_vec=np.zeros(dim),
                          ub_vec=V_0)
    problem = hopsy.Problem(
            np.array([P]),
            np.array([c*np.sum(P)]),
            model
    )
    problem = hopsy.add_box_constraints(
            problem=problem,
            lower_bound = np.zeros(dim),
            upper_bound = V_0
    )
    mc = hopsy.MarkovChain(problem, 
                           proposal=hopsy.UniformCoordinateHitAndRunProposal, 
                           starting_point=c*np.ones(dim))
    rng = hopsy.RandomNumberGenerator()
    acceptance_rate, states = hopsy.sample(mc, rng, 
                                           n_samples=n, 
                                           thinning=thinning,
                                           n_threads=n_threads)
    return states


def vacc_bayes_opt_w_constr(
        n_initial_pts,
        vacc_engine, 
        n_sim, pool,
        noise_prior=0.1,
        acq="UCB",
        tolerance=1e-6, max_iters=100):
    """
    Do Bayesian optimization for the vaccination problem.
    n_initial_pts:
        how many initial pts to sample
    vacc_engine: VaccRateOptEngine
    pool: multiprocess.Pool
    """
    V_0 = vacc_engine.V_0
    P = np.array(vacc_engine.pop_df['pop'])
    P_sum = np.sum(P)
    c = vacc_engine.opt_config['constraint_bnd']
    dim = len(V_0)
    assert len(V_0) == len(P)  # dimension check
    # sample initial points
    query_pts = constraint_sample(n_initial_pts,dim,V_0, P, c)
    # evaluate objective function at each sample
    query_fvals = np.array([np.mean(vacc_engine.query(V_delta=s, pool=pool, n_sim=n_sim)) for s in query_pts])
    #print(f_values)
    # build gaussian process regressor
    model = gp.GaussianProcessRegressor(alpha=noise_prior)
    model.fit(query_pts,query_fvals)
    # PROPOSING THE NEXT POINT #
    # setup constraint functions
    ineq_constr = scipy.optimize.LinearConstraint(A=np.eye(dim), lb=np.zeros(dim), ub=V_0)
    budget_constr = scipy.optimize.LinearConstraint(A=P, lb=c*P_sum, ub=c*P_sum)

    n_iters = 0
    best_argmax_sofar = np.argmax(query_fvals)
    best_max_sofar = query_fvals[best_argmax_sofar]

    def acq(x):
        return -UCB(x, model, 1.4)

    opt_restarts = 5 #  
    dist_btwn_pts = np.inf
    last_query_x = best_argmax_sofar
    while n_iters <= max_iters and dist_btwn_pts > tolerance:
        next_query_x = np.inf
        next_query_acqval = np.inf
        for opt_number in range(opt_restarts):
            starting_point = constraint_sample(1, dim, V_0, P, c)[0]
            print(starting_point,flush=True)
            acq_opt = scipy.optimize.minimize(
                    fun=acq,
                    x0=starting_point,
                    constraints=(ineq_constr, budget_constr)
                )
            if acq_opt['fun'] < next_query_acqval:
                next_query_x = acq_opt['x']
                next_query_acqval = acq_opt['fun']
        next_fval = np.mean(vacc_engine.query(V_delta=next_query_x, pool=pool, n_sim=n_sim))
        print("query mat")
        print(query_pts)
        query_pts = np.append(query_pts,[next_query_x],axis=0)  # add it to the rows
        query_fvals = np.append(query_fvals,next_fval)

        print(next_query_x,next_fval)

        model = gp.GaussianProcessRegressor(alpha=noise_prior)
        model.fit(query_pts,query_fvals)

        dist_btwn_pts = np.linalg.norm(last_query_x-next_query_x,ord=np.inf)
        last_query_x = next_query_x
        n_iters += 1
    last_query_x = np.array([round(x) if np.isclose(x,0) else x for x in last_query_x])
    print(last_query_x)
    print(vacc_engine.check_constraint(last_query_x))

        #print(max_ucb)
        ##max_ucb['x']
        #print(np.mean(vacc_engine.query(V_delta=max_ucb['x'],pool=pool,n_sim=n_sim)))



if __name__ == "__main__":
    # NOTE: we assume the labeling is consistent
    # between vacc_df and dist_mat

    # set up population dataframe
    # with vaccination rates
    vacc_df = {
            'pop': [1200, 1200, 4000, 2000, 5000],
            'vacc': [0.9,0.9,0.9,0.9, 0.75]
    }
    vacc_df = pd.DataFrame(vacc_df)

    # set up distance matrix
    # all pairwise distances
    dist_mat = [
            [0,1,5,5,1],
            [1,0,5,5,1],
            [5,5,0,10,1],
            [5,5,10,0,1],
            [5,5,10,1,0]
    ]

    # setup the configuration
    # for the disease simulation
    tsir_config = {
        "iters": 75,    # number of iterations to run sim for
        "tau1": 0.7,    # gravity model parameters
        "tau2": 1.2,
        "rho": 0.97,
        "theta": 0.05,
        "alpha": 0.97, # mixing rate
        "beta": 7      # disease infectiousness
    }
    # arguments for optimizer oracle
    sim_params = {
            'config':tsir_config,  # contains all disease parameters
            'pop':vacc_df,
            'distances':np.array(dist_mat)
    }

    # optimizer oracle configuration
    opt_config = {
        'obj':"attacksize",   # objective function
        'V_repr':"ratio",     # represent vacc rates as a ratio: [0,1]
        'constraint_bnd':0.05, # set c=0.05 (percentage can go down by 5%)
        'constraint_type':'ineq'
    }

    I = np.array([0,0,1,0,0])   # seeding: set outbreak to begin in district 4

    v = vacc.VaccProblemLAMCTSWrapper(
            opt_config = opt_config, 
            V_0=vacc_df['vacc'], 
            seed = I,
            sim_config = tsir_config, 
            pop = vacc_df, 
            distances = np.array(dist_mat),
            cores=5, n_sim=10,
            output_dir = "/home/nicholasw/Documents/sync/UVA files/Semesters/Semester 7/CS 4501/project/CS_4501_Fall22_RL_Project/output/"
        )

    # plug all arguments into oracle
    engine = vacc.VaccRateOptEngine(
            opt_config=opt_config,
            V_0=vacc_df['vacc'], seed=I,
            sim_config=tsir_config,
            pop=vacc_df,
            distances=np.array(dist_mat))

    # setup for multithreading using 5 threads
    with multiprocess.Pool(10) as p:
        # query the vector where we uniformly distribute the vaccination decrease over all districts
        result, sim_pool = engine.query(V_delta=0.05*np.ones(5),pool=p,n_sim=150, return_sim_pool=True)
        vacc_bayes_opt_w_constr(
                n_initial_pts=20,
                vacc_engine=engine,
                n_sim=150,
                pool=p,
                max_iters=10
                )
        print(np.mean(result))

        result, sim_pool = engine.query(V_delta=np.array([0,0,1,0,0])*\
                                                    (opt_config['constraint_bnd']*sum(vacc_df['pop']))/vacc_df['pop'][2],
                                        pool=p,n_sim=150, return_sim_pool=True)
        print(np.mean(result))



