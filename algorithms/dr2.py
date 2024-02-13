from dataclasses import dataclass

import numpy as np
import ioh
import pandas as pd

from .algorithm import Algorithm, SolutionType, DEFAULT_MAX_BUDGET, SIGMA_MAX
from .utils import Weights, init_lambda


@dataclass
class DR2(Algorithm):
    budget: int = DEFAULT_MAX_BUDGET
    mu: int = 1
    lambda_: int = 2
    sigma0: float = 1
    verbose: bool = True
    mirrored: bool = True
    use_old_data: bool = False
    old_data_file: pd.DataFrame = pd.DataFrame()

    def __call__(self, problem: ioh.ProblemType) -> SolutionType:

        n = problem.meta_data.n_variables
        self.lambda_ = self.lambda_ or init_lambda(n, "default")
        self.mu = self.mu or self.lambda_ // 2

        beta_scale = 1 / n
        beta = np.sqrt(beta_scale)
        c = beta

        zeta = np.zeros((n, 1))
        sigma_local = np.ones((n, 1)) * self.sigma0
        sigma = self.sigma0

        c1 = np.sqrt(c / (2 - c))
        c2 = np.sqrt(n) * c1
        c3 = 1 / (5 * n)

        weights = Weights(self.mu, self.lambda_, n)
        x_prime = np.zeros((n, 1))
        n_samples = self.lambda_ if not self.mirrored else self.lambda_ // 2

        
        if self.use_old_data == True:
            num_columns = self.old_data_file.shape[1]
            nr_of_previous_gen = self.old_data_file['generation'][:].max()
            for gen in range(int(nr_of_previous_gen)):
                z_prime = np.zeros((n, 1))
                Y = np.zeros((n, 1))
                sel = gen*self.lambda_ + self.old_data_file[self.old_data_file['generation'] == 1]['fitness'].idxmin()
                #change to idxmax to maximize fitness instead
                X = self.old_data_file.iloc[sel, num_columns-n:]
                X = X.to_numpy().reshape((3, 1))
                Y = X - x_prime
                Z = Y / (sigma * sigma_local)
                x_prime = X
                z_prime = Z*weights.w
                z_prime *= np.sqrt(weights.mueff)

                zeta = ((1 - c) * zeta) + (c * z_prime)
                sigma = min(
                    sigma
                    * np.power(np.exp((np.linalg.norm(zeta) / c2) - 1 + c3), beta),
                    SIGMA_MAX,
                )
                sigma_factor = np.power((np.abs(zeta) / c1) + (7 / 20), beta_scale)
                sigma_local = sigma_local * sigma_factor
                sigma_local = sigma_local.clip(0, SIGMA_MAX)

        try:
            while not self.should_terminate(problem, self.lambda_):
                Z = np.random.normal(size=(n, n_samples))
                if self.mirrored:
                    Z = np.hstack([Z, -Z])
                Y = sigma * (sigma_local * Z)
                X = x_prime + Y
                f = problem(X.T)
                idx = np.argsort(f)
                mu_best = idx[: self.mu]

                z_prime = np.sum(
                    Z[:, mu_best] * weights.w, axis=1, keepdims=True
                ) * np.sqrt(weights.mueff)
                y_prime = np.sum(Y[:, mu_best] * weights.w, axis=1, keepdims=True)
                x_prime = x_prime + y_prime
                zeta = ((1 - c) * zeta) + (c * z_prime)
                sigma = min(
                    sigma
                    * np.power(np.exp((np.linalg.norm(zeta) / c2) - 1 + c3), beta),
                    SIGMA_MAX,
                )
                sigma_local *= np.power((np.abs(zeta) / c1) + (7 / 20), beta_scale)
                sigma_local = sigma_local.clip(0, SIGMA_MAX)

                if self.verbose:
                    print(
                        f"e: {problem.state.evaluations}/{self.budget}",
                        f"fopt: {problem.state.current_best.y:.3f};",
                        f"f: {np.median(f):.3f} +- {np.std(f):.3f} ",
                        f"[{np.min(f):.3f}, {np.max(f):.3f}];",
                        f"sigma: {sigma:.3e}",
                        f"sigma_local: {np.median(sigma_local):.3e} +- {np.std(sigma_local):.3f};",
                    )
        except KeyboardInterrupt:
            pass
        return x_prime
