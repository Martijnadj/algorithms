from dataclasses import dataclass

import numpy as np
import ioh
import pandas as pd

from .algorithm import Algorithm, SolutionType, DEFAULT_MAX_BUDGET


@dataclass
class CMAES(Algorithm):
    budget: int = DEFAULT_MAX_BUDGET
    lambda_: int = None
    mu: float = None
    sigma0: float = 1.0
    verbose: bool = True
    sep: bool = False
    use_old_data: bool = False
    old_data_file: pd.DataFrame = pd.DataFrame()

    def inverse_scale(self, x_scaled):
        #Convert simulation parameters to algorithm parameters for easier 
        #Evolution strategy simulation
        x_min = np.array([1e-9, 1e-9, 1e-9, 0, 0, 0, 0])
        x_max = np.array([1e-5, 1e-5, 1e-5, 5000, 500, 500, 500])
        x = x_scaled.copy()
        
        for i in range(len(x_scaled)):
            for j in range(len(x_scaled[i])):
                x[i,j] = 10 * (x_scaled[i,j] - x_min[j]) / (x_max[j] - x_min[j]) - 5
        return x

    def __call__(self, problem: ioh.ProblemType) -> SolutionType:
        n = problem.meta_data.n_variables
        self.lambda_ = self.lambda_ or (4 + np.floor(3 * np.log(n))).astype(int)
        self.mu = self.lambda_ // 2
        sigma = self.sigma0
        # w
        w = np.log((self.lambda_ + 1) / 2) - np.log(np.arange(1, self.lambda_ + 1))
        w = w[: self.mu]
        mueff = w.sum() ** 2 / (w**2).sum()
        w = w / w.sum()

        # Learning rates
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + 2 * mueff / 2)
        cc = (4 + (mueff / n)) / (n + 4 + (2 * mueff / n))
        cs = (mueff + 2) / (n + mueff + 5)
        damps = 1.0 + (2.0 * max(0.0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs)
        chiN = n**0.5 * (1 - 1 / (4 * n) + 1 / (21 * n**2))

        # dynamic parameters
        m = np.random.rand(n, 1)
        dm = np.zeros(n)
        pc = np.zeros((n, 1))
        ps = np.zeros((n, 1))
        B = np.eye(n)
        C = np.eye(n)
        D = np.ones((n, 1))
        invC = np.eye(n)

        

        if self.use_old_data == True:
            num_columns = self.old_data_file.shape[1]
            nr_of_previous_gen = self.old_data_file['generation'][:].max()
            print(nr_of_previous_gen)
            for gen in range(int(nr_of_previous_gen)):
                # select data for current generation
                f = self.old_data_file[self.old_data_file['generation'] == gen+1]

                
                # select best fitness parameters
                f = f.sort_values(by='fitness', ascending=False)
                f = f.reset_index()

                # recombine
                m_old = m.copy()
                X = self.inverse_scale(f.iloc[:, -7:].values)
                Y = (X-m.T)/sigma

                m = m_old + (1 * ((X[:self.mu, :].T - m_old) @ w).reshape(-1, 1))

                # adapt
                dm = (m - m_old) / sigma
                ps = (1 - cs) * ps + (np.sqrt(cs * (2 - cs) * mueff) * invC @ dm)
                sigma *= np.exp((cs / damps) * ((np.linalg.norm(ps) / chiN) - 1))
                hs = (
                    np.linalg.norm(ps)
                    / np.sqrt(1 - np.power(1 - cs, 2 * (gen+1)))
                ) < (1.4 + (2 / (n + 1))) * chiN

                dhs = (1 - hs) * cc * (2 - cc)
                pc = (1 - cc) * pc + (hs * np.sqrt(cc * (2 - cc) * mueff)) * dm


                rank_one = c1 * pc * pc.T
                old_C = (1 - (c1 * dhs) - c1 - (cmu * w.sum())) * C
                rank_mu = cmu * (w * Y[:self.mu,:].T @ Y[:self.mu:,:])
                C = old_C + rank_one + rank_mu

                if np.isinf(C).any() or np.isnan(C).any() or (not 1e-16 < sigma < 1e6):
                    sigma = self.sigma0
                    pc = np.zeros((n, 1))
                    ps = np.zeros((n, 1))
                    C = np.eye(n)
                    B = np.eye(n)
                    D = np.ones((n, 1))
                    invC = np.eye(n)

                else:
                    C = np.triu(C) + np.triu(C, 1).T
                    if not self.sep:
                        D, B = np.linalg.eigh(C)
                    else:
                        D = np.diag(C)


                D = np.sqrt(D).reshape(-1, 1)
                invC = np.dot(B, D ** -1 * B.T)
                

        while not self.should_terminate(problem, self.lambda_):
            Z = np.random.normal(0, 1, (n, self.lambda_))
            Y = np.dot(B, D * Z)
            X = m + (sigma * Y)
            f = np.array(problem(X.T))

            # select
            fidx = np.argsort(f)[::-1]
            mu_best = fidx[: self.mu]

            # recombine

            m_old = m.copy()
            m = m_old + (1 * ((X[:, mu_best] - m_old) @ w).reshape(-1, 1))

            # adapt
            dm = (m - m_old) / sigma
            ps = (1 - cs) * ps + (np.sqrt(cs * (2 - cs) * mueff) * invC @ dm)
            sigma *= np.exp((cs / damps) * ((np.linalg.norm(ps) / chiN) - 1))
            hs = (
                np.linalg.norm(ps)
                / np.sqrt(1 - np.power(1 - cs, 2 * (problem.state.evaluations / self.lambda_)))
            ) < (1.4 + (2 / (n + 1))) * chiN
            dhs = (1 - hs) * cc * (2 - cc)
            pc = (1 - cc) * pc + (hs * np.sqrt(cc * (2 - cc) * mueff)) * dm


            rank_one = c1 * pc * pc.T
            old_C = (1 - (c1 * dhs) - c1 - (cmu * w.sum())) * C
            rank_mu = cmu * (w * Y[:, mu_best] @ Y[:, mu_best].T)
            C = old_C + rank_one + rank_mu

            if np.isinf(C).any() or np.isnan(C).any() or (not 0 <= sigma < 1e6):
                sigma = self.sigma0
                pc = np.zeros((n, 1))
                ps = np.zeros((n, 1))
                C = np.eye(n)
                B = np.eye(n)
                D = np.ones((n, 1))
                invC = np.eye(n)
            else:
                C = np.triu(C) + np.triu(C, 1).T
                if not self.sep:
                    D, B = np.linalg.eigh(C)
                else:
                    D = np.diag(C)


            D = np.sqrt(D).reshape(-1, 1)
            invC = np.dot(B, D ** -1 * B.T)
