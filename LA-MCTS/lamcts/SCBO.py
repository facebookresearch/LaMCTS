# https://botorch.org/tutorials/scalable_constrained_bo
import os
import math
from dataclasses import dataclass

import torch
from torch import Tensor
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize
from torch.quasirandom import SobolEngine

import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.exceptions.errors import ModelFittingError
# Constrained Max Posterior Sampling
# is a new sampling class, similar to MaxPosteriorSampling,
# which implements the constrained version of Thompson Sampling described in [1]
from botorch.generation.sampling import ConstrainedMaxPosteriorSampling

import warnings
warnings.filterwarnings("ignore")
torch.set_num_threads(12)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
SMOKE_TEST = os.environ.get("SMOKE_TEST")

import numpy as np

### class and methods to store and act on state of SCBO optimizer ###
@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10  # Note: The original paper uses 3
    best_value: float = -float("inf")
    best_constraint_values: Tensor = torch.ones(2,)*torch.inf
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )

def update_tr_length(state):
    # Update the length of the trust region according to
    # success and failure counters
    # (Just as in original TuRBO paper)
    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    if state.length < state.length_min:  # Restart when trust region becomes too small
        state.restart_triggered = True

    return state


def update_state(state, Y_next, C_next):
    ''' Method used to update the TuRBO state after each
        step of optimization.

        Success and failure counters are updated accoding to
        the objective values (Y_next) and constraint values (C_next)
        of the batch of candidate points evaluated on the optimization step.

        As in the original TuRBO paper, a success is counted whenver
        any one of the new candidate points imporves upon the incumbent
        best point. The key difference for SCBO is that we only compare points
        by their objective values when both points are valid (meet all constraints).
        If exactly one of the two points beinc compared voliates a constraint, the
        other valid point is automatically considered to be better. If both points
        violate some constraints, we compare them inated by their constraint values.
        The better point in this case is the one with minimum total constraint violation
        (the minimum sum over constraint values)'''

    # Determine which candidates meet the constraints (are valid)
    bool_tensor = C_next <= 0
    bool_tensor = torch.all(bool_tensor, dim=-1)
    Valid_Y_next = Y_next[bool_tensor]
    Valid_C_next = C_next[bool_tensor]
    if Valid_Y_next.numel() == 0:  # if none of the candidates are valid
        # pick the point with minimum violation
        sum_violation = C_next.sum(dim=-1)
        min_violation = sum_violation.min()
        # if the minimum voilation candidate is smaller than the violation of the incumbent
        if min_violation < state.best_constraint_values.sum():
            # count a success and update the current best point and constraint values
            state.success_counter += 1
            state.failure_counter = 0
            # new best is min violator
            state.best_value = Y_next[sum_violation.argmin()].item()
            state.best_constraint_values = C_next[sum_violation.argmin()]
        else:
            # otherwise, count a failure
            state.success_counter = 0
            state.failure_counter += 1
    else:  # if at least one valid candidate was suggested,
          # throw out all invalid candidates
          # (a valid candidate is always better than an invalid one)

        # Case 1: if best valid candidate found has a higher obj value that incumbent best
            # count a success, the obj valuse has been improved
        imporved_obj = max(Valid_Y_next) > state.best_value + \
                           1e-3 * math.fabs(state.best_value)
        # Case 2: if incumbent best violates constraints
            # count a success, we now have suggested a point which is valid and therfore better
        obtained_validity = torch.all(state.best_constraint_values > 0)
        if imporved_obj or obtained_validity:  # If Case 1 or Case 2
            # count a success and update the best value and constraint values
            state.success_counter += 1
            state.failure_counter = 0
            state.best_value = max(Valid_Y_next).item()
            state.best_constraint_values = Valid_C_next[Valid_Y_next.argmax()]
        else:
            # otherwise, count a fialure
            state.success_counter = 0
            state.failure_counter += 1

    # Finally, update the length of the trust region according to the
    # updated success and failure counts
    state = update_tr_length(state)

    return state


class SCBO:
    def __init__(
        self,
        dim, batch_size
    ):
        self.dim = dim
        self.batch_size = batch_size

    def get_initial_points(self, n_pts, seed=0):
        sobol = SobolEngine(dimension=self.dim, scramble=True, seed=seed)
        X_init = sobol.draw(n=n_pts).to(dtype=dtype, device=device)
        return X_init

    def generate_batch(
        state,
        model,  # GP model
        X,  # Evaluated points on the domain [0, 1]^d
        Y,  # Function values
        batch_size,
        n_candidates=None,  # Number of candidates for Thompson sampling
        constraint_model=None
    ):
        assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
        if n_candidates is None:
            n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

        # Scale the TR to be proportional to the lengthscales
        x_center = X[Y.argmax(), :].clone()
        weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
        weights = weights / weights.mean()
        weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

        # Thompson Sampling w/ Constraints (SCBO)
        dim = X.shape[-1]
        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = (
            torch.rand(n_candidates, dim, dtype=dtype, device=device)
            <= prob_perturb
        )
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

        # Create candidate points from the perturbations and the mask
        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand[mask] = pert[mask]

        # Sample on the candidate points using Constrained Max Posterior Sampling
        constrained_thompson_sampling = ConstrainedMaxPosteriorSampling(
            model=model, constraint_model=constraint_model, replacement=False)
        with torch.no_grad():
            X_next = constrained_thompson_sampling(X_cand, num_samples=batch_size)

        return X_next

    def get_fitted_model(X, Y, dim, max_cholesky_size=2000):
        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
        MaternKernel(nu=2.5, ard_num_dims=dim,
                     lengthscale_constraint=Interval(0.005, 4.0))
        )
        model = SingleTaskGP(X, Y, covar_module=covar_module,
                            likelihood=likelihood, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        with gpytorch.settings.max_cholesky_size(max_cholesky_size):
            fit_gpytorch_mll(mll)

        return model

    def optimize(self,
        objective, constraints,
        n_init, num_samples=np.inf,
        init_X = None, init_Y = None,
        max_cholesky_size = 2000
    ):
        """
            objective: callable fn
            constraints: list of functions
            n_init
            num_samples: how many valid samples do we want?
        """
        if init_X is None or init_Y is None:
            train_X = self.get_initial_points(n_init)
            train_Y = torch.tensor(
                [objective(np.array(x)) for x in train_X], dtype=dtype, device=device
            ).unsqueeze(-1)
        else:
            train_X = init_X
            train_Y = init_Y
        print(train_X)
        print(train_Y)
        train_constr = []
        valid_samples = 0

        # TODO: counting valid initial points doesn't work rn
        for constr in constraints:
            # okay yeah I should probably be using tensors but let's do this dumb
            # way first
            C = torch.tensor(
                [constr(np.array(x)) for x in train_X], dtype=dtype, device=device
            ).unsqueeze(-1)
            train_constr.append(C)
        print(train_constr)

        state = TurboState(dim=self.dim, batch_size=self.batch_size)
        N_CANDIDATES = min(5000, max(2000, 200 * self.dim)) if not SMOKE_TEST else 4
        #print(train_constr)
        # Run until TuRBO converges, or we get enough valid samples
        while not state.restart_triggered and valid_samples < num_samples:  
        #for i in range(5):
            # Fit GP models for objective and constraints
            # TODO: idk if this is good practice
            try:
                model = SCBO.get_fitted_model(train_X, train_Y, self.dim)
                c_models = [SCBO.get_fitted_model(train_X, train_C, self.dim) for train_C in train_constr]
            except ModelFittingError:
                break

            # Generate a batch of candidates
            with gpytorch.settings.max_cholesky_size(max_cholesky_size):
                X_next = SCBO.generate_batch(
                    state=state,
                    model=model,
                    X=train_X,
                    Y=train_Y,
                    batch_size=self.batch_size,
                    n_candidates=N_CANDIDATES,
                    constraint_model=ModelListGP(*c_models)
                    )

            # Evaluate both the objective and constraints for the selected candidaates
            Y_next = torch.tensor(
                [objective(np.array(x)) for x in X_next], dtype=dtype, device=device
            ).unsqueeze(-1)

            C_next_list = []
            for constr in constraints:
                C_next_list.append(
                    torch.tensor(
                        [constr(np.array(x)) for x in X_next], dtype=dtype, device=device
                    ).unsqueeze(-1)
                )

            # TODO: works as long as batch_size = 1
            C_next = torch.cat(C_next_list, dim=-1)
            is_valid = C_next <= 0
            is_valid = torch.all(is_valid).item()
            valid_samples += np.int64(is_valid)

            # Update TuRBO state
            state=update_state(state, Y_next, C_next)

            # Append data
            #   Notice we append all data, even points that violate
            #   the constriants, this is so our constraint models
            #   can learn more about the constranit functions and
            #   gain confidence about where violation occurs
            train_X=torch.cat((train_X, X_next), dim=0)
            train_Y=torch.cat((train_Y, Y_next), dim=0)
            for i, C in enumerate(C_next_list):
                train_constr[i]=torch.cat((train_constr[i], C), dim=0)
            # C1 = torch.cat((C1, C1_next), dim=0)
            # C2 = torch.cat((C2, C2_next), dim=0)

            # Print current status
            #   Note: state.best_value is always the best objective value
            #   found so far which meets the constraints, or in the case
            #   that no points have been found yet which meet the constraints,
            #   it is the objective value of the point with the
            #   minimum constraint violation
            print(
                f"{len(train_X)}) Best value: {state.best_value:.2e}, TR length: {state.length:.2e}"
            )
            print(valid_samples)
        #  Valid samples must have BOTH c1 <= 0 and c2 <= 0
        constraint_vals=torch.cat(train_constr, dim=-1)
        bool_tensor=constraint_vals <= 0
        bool_tensor=torch.all(bool_tensor, dim=-1).unsqueeze(-1)
        Valid_Y=train_Y[bool_tensor]
        #Valid_X=train_X[torch.broadcast_to(bool_tensor,train_X.size())]
        Valid_X = train_X[bool_tensor.squeeze()]

        #Y_argmax = torch.argmax(Valid_Y)
        #best_x = Valid_X[Y_argmax]
        #print(
        #    f"With constraints, the best value we found is: {Valid_Y.max().item():.4f} with X=",best_x)

        #return (best_x,Valid_Y.max().item())
        return np.array(Valid_X), np.array(Valid_Y)



# run the test with the Ackley function
if __name__ == "__main__":
    # Here we define the example 20D Ackley function 
    fun = Ackley(dim=20, negate=True).to(dtype=dtype, device=device)
    fun.bounds[0, :].fill_(-5)
    fun.bounds[1, :].fill_(10)
    dim = fun.dim
    lb, ub = fun.bounds

    batch_size = 4
    n_init = 2 * dim
    max_cholesky_size = float("inf")  # Always use Cholesky

    # When evaluating the function, we must first unnormalize the inputs since 
    # we will use normalized inputs x in the main optimizaiton loop
    def eval_objective(x):
        """This is a helper function we use to unnormalize and evalaute a point"""
        return fun(unnormalize(x, fun.bounds))

    def c1(x): # Equivalent to enforcing that x[0] >= 0
        return -x[0] 

    def c2(x): # Equivalent to enforcing that x[1] >= 0
        return -x[1] 

    # We assume c1, c2 have same bounds as the Ackley function above 
    def eval_c1(x):
        """This is a helper function we use to unnormalize and evalaute a point"""
        return c1(unnormalize(x, fun.bounds))

    def eval_c2(x):
        """This is a helper function we use to unnormalize and evalaute a point"""
        return c2(unnormalize(x, fun.bounds))

    opt = SCBO(dim=20, batch_size=4)
    X, fX = opt.optimize(
        objective=eval_objective,
        constraints = [eval_c1,eval_c2],
        n_init = 2*20, num_samples=10
        )
    print(X)
    print(fX)
