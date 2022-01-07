import time
from datetime import timedelta
import numpy as np
import pandas as pd
from numpy.random import default_rng
from scipy.optimize._differentialevolution import DifferentialEvolutionSolver
from dataset import Dataset
from model import Model
from model_runner import ModelRunner
from utils import normalize, mae


def run_model_multi(params, length, trials=16):
    i = []
    r = []
    d = []
    for _ in range(trials):
        model = Model(params=params, max_iter=length)  # Popsize
        runner = ModelRunner(model)
        runner.run()
        i.append(np.array(runner.infected))
        r.append(np.array(runner.recovered))
        d.append(np.array(runner.deaths))
    i = sum(i) / len(i)
    r = sum(r) / len(r)
    d = sum(d) / len(d)
    return i, r, d


def objective_function(x, yi, yr, yd):
    i, r, d = run_model_multi((x[0], 1/x[1], 1/x[2]), len(yi))
    i = pd.Series(i).rolling(14, 1, center=True).mean().to_numpy()
    r = pd.Series(r).rolling(14, 1, center=True).mean().to_numpy()
    d = pd.Series(d).rolling(14, 1, center=True).mean().to_numpy()
    i = normalize(i)
    r = normalize(r)
    d = normalize(d)
    return mae(i, yi) + mae(r, yr) + 0.3 * mae(d, yd)


class Optimizer:
    def __init__(self):
        #self.dataset = Dataset()
        #self.dataset = Dataset('Poland', '2020-03-10', '2020-04-20')
        #self.dataset = Dataset('Czechia', '2020-03-01', '2020-04-10')
        self.dataset = Dataset('Austria', '2020-03-01', '2020-04-01')
        #self.dataset = Dataset('South Korea', '2020-02-20', '2020-03-17')
        i = self.dataset.data['new_cases'].rolling(14, 1, center=True).mean().to_numpy()
        r = self.dataset.data['recovered'].rolling(14, 1, center=True).mean().to_numpy()
        d = self.dataset.data['new_deaths'].rolling(14, 1, center=True).mean().to_numpy()
        #self.data = normalize(i)
        self.i = normalize(i)
        self.r = normalize(r)
        self.d = normalize(d)
        self.function = objective_function
        self.initial = (0.1, 10.0, 10.0)
        self.bounds = ((0.0, 1.0), (1.0, 40.0), (1.0, 40.0))
        self.bounds_pso = (np.array([0.0, 1.0, 1.0]), np.array([1.0, 40.0, 40.0]))
        self.initials = self.create_initials()

    def create_initials(self):
        b = self.bounds
        s0 = np.linspace(b[0][0], b[0][1], 4)
        s1 = np.linspace(b[1][0], b[1][1], 5)
        s2 = np.linspace(b[2][0], b[2][1], 5)
        return np.vstack(np.meshgrid(s0, s1, s2)).reshape(3, -1).T

    def function_iterable(self, x, y):
        return self.function(x, y),

    def function_with_args(self, x):
        return self.function(x, self.i, self.r, self.d)

    def differential_evolution(self, filename):
        rng = default_rng()
        #end_time = time.time() + (3600 * 12.0)
        solver = None
        progress = []

        def callback(xk, convergence):
            print(f'{solver.population_energies[0]}  |  {xk}')
            progress.append((solver.population_energies[0], xk[0], xk[1], xk[2]))
            #t = time.time()
            #if t > end_time:
            #    return True
            #else:
            #    return False

        # population = popsize * len(params) || ignored
        # iter - 120 = 8h
        with DifferentialEvolutionSolver(self.function_with_args, bounds=self.bounds, args=None, popsize=100,
            maxiter=140, seed=rng, callback=callback, polish=False, init=self.initials
        ) as solver:
            result = solver.solve()
        ###
        with open(f'results/{filename}.csv', 'w') as file:
            file.write('f;x0;x1;x2\n')
            for p in progress:
                file.write(f'{p[0]};{p[1]};{p[2]};{p[3]}\n')
            file.write(f'{result.fun};{result.x[0]};{result.x[1]};{result.x[2]}')
        return result.x


def main():
    opt = Optimizer()
    t0 = time.time()
    opt.differential_evolution('austria_1')
    opt.differential_evolution('austria_2')
    t1 = time.time()
    print(t1 - t0)
    print(timedelta(seconds=t1-t0))


if __name__ == "__main__":
    main()
