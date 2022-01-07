import time
from datetime import timedelta
import numpy as np
import pandas as pd
from numpy.random import default_rng
import scipy.optimize as sp
from scipy.optimize._differentialevolution import DifferentialEvolutionSolver
from deap import algorithms, tools, base, creator, cma
import pyswarms
import skopt
from dataset import Dataset
from model import Model
from model_runner import ModelRunner
from utils import normalize, mse, mae


def run_model_ei(e, i, params, length, trials=10):
    ''' Customize trials count '''
    data = []
    for _ in range(trials):
        model = Model(exposed=e, infected=i, params=params, max_iter=length)  # Popsize
        runner = ModelRunner(model)
        runner.run()
        x = np.array(runner.infected)
        data.append(x)
    return sum(data) / len(data)


def run_model_all_data(params, length, trials=10):
    ''' Customize trials count '''
    exposed = []
    infected = []
    recovered = []
    deaths = []
    for _ in range(trials):
        model = Model(params=params, max_iter=length)  # Popsize
        runner = ModelRunner(model)
        runner.run()
        exposed.append(np.array(runner.exposed))
        infected.append(np.array(runner.infected))
        recovered.append(np.array(runner.recovered))
        deaths.append(np.array(runner.deaths))
    return (
        sum(exposed) / len(exposed),
        sum(infected) / len(infected),
        sum(recovered) / len(recovered),
        sum(deaths) / len(deaths)
    )


def run_model(params, length, trials=10):
    ''' Customize trials count '''
    data = []
    for _ in range(trials):
        model = Model(params=params, max_iter=length)  # Popsize
        runner = ModelRunner(model)
        runner.run()
        x = np.array(runner.infected)
        data.append(x)
    return sum(data) / len(data)


def objective_function(x, y):
    data = run_model((x[0], 1/x[1], 1/x[2]), len(y))
    #data = pd.Series(data).rolling(14, 1, center=True).mean().to_numpy()
    data = normalize(data)  # Customize normalization
    return mae(data, y)  # Customize Error measure


class Optimizer:
    def __init__(self):
        #self.dataset = Dataset()
        #self.dataset = Dataset('Poland', '2020-03-10', '2020-04-20')
        #self.dataset = Dataset('Czechia', '2020-03-01', '2020-04-10')
        self.dataset = Dataset('Austria', '2020-03-01', '2020-04-01')
        #self.dataset = Dataset('South Korea', '2020-02-20', '2020-03-17')

        d = self.dataset.data['new_cases']
        #d = self.dataset.data['new_cases'].rolling(14, 1, center=True).mean().to_numpy()
        self.data = normalize(d)
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
        return self.function(x, self.data)

    def brute(self):
        progress = []
        total = 15 * 34 * 34
        iteration = 0
        for i in np.linspace(0.0, 1.0, 15):
            for j in np.linspace(1.0, 20.0, 34):
                for k in np.linspace(1.0, 20.0, 34):
                    print(f'Working on {iteration}/{total}')
                    iteration += 1
                    r = self.function((i, j, k), self.data)
                    progress.append((r, i, j, k))
        ###
        with open('results/brute.csv', 'w') as file:
            file.write('f;x0;x1;x2\n')
            for p in progress:
                file.write(f'{p[0]};{p[1]};{p[2]};{p[3]}\n')

    def powell(self):
        progress = []
        total = len(self.initials)
        for i, init in enumerate(self.initials):
            print(f'Working on {i}/{total}')
            result = sp.minimize(self.function, init, bounds=self.bounds, args=(self.data,), method='Powell')
            progress.append((result.fun, result.x[0], result.x[1], result.x[2]))
        ###
        with open('results/powell.csv', 'w') as file:
            file.write('f;x0;x1;x2\n')
            for p in progress:
                file.write(f'{p[0]};{p[1]};{p[2]};{p[3]}\n')

    def bfgs(self):
        progress = []
        total = len(self.initials)
        for i, init in enumerate(self.initials):
            print(f'Working on {i}/{total}')
            result = sp.minimize(self.function, init, bounds=self.bounds, args=(self.data,), method='L-BFGS-B')
            progress.append((result.fun, result.x[0], result.x[1], result.x[2]))
        ###
        with open('results/bfgs.csv', 'w') as file:
            file.write('f;x0;x1;x2\n')
            for p in progress:
                file.write(f'{p[0]};{p[1]};{p[2]};{p[3]}\n')

    def dual_annealing(self):
        rng = default_rng()
        progress = []
        end_time = time.time() + (3600 * 9.5)
        previous_time = time.time()

        def callback(x, f, c):
            nonlocal previous_time
            progress.append((f, x[0], x[1], x[2]))
            t = time.time()
            print(f'{t - previous_time:.2f} || {f}  |  {x}')
            previous_time = t
            if t > end_time:
                return True
            else:
                return False

        result = sp.dual_annealing(self.function, bounds=self.bounds, args=(self.data,), maxiter=1500, seed=rng,
                                   callback=callback, maxfun=1e8, x0=self.initial
        )
        ###
        with open('results/dual_annealing.csv', 'w') as file:
            file.write('f;x0;x1;x2\n')
            for p in progress:
                file.write(f'{p[0]};{p[1]};{p[2]};{p[3]}\n')
            file.write(f'{result.fun};{result.x[0]};{result.x[1]};{result.x[2]}')
        return result.x

    def differential_evolution(self):
        rng = default_rng()
        end_time = time.time() + (3600 * 12.0)
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
        with DifferentialEvolutionSolver(self.function, bounds=self.bounds, args=(self.data,), popsize=100,
            maxiter=120, seed=rng, callback=callback, polish=False, init=self.initials
        ) as solver:
            result = solver.solve()
        ###
        with open('results/austria_2.csv', 'w') as file:
            file.write('f;x0;x1;x2\n')
            for p in progress:
                file.write(f'{p[0]};{p[1]};{p[2]};{p[3]}\n')
            file.write(f'{result.fun};{result.x[0]};{result.x[1]};{result.x[2]}')
        return result.x

    def cma_es(self):
        def generate(x):
            x = np.clip(x, -10.0, 10.0)
            return creator.Individual(x)
        def feasible(ind):
            if ind[0] < 0.0 or ind[0] > 1.0 or ind[1] < 1.0 or ind[1] > 40.0 or ind[2] < 1.0 or ind[2] > 40.0:
                return False
            return True

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        strategy = cma.Strategy(self.initial, sigma=5, lambda_=100)
        toolbox = base.Toolbox()
        toolbox.register("evaluate", self.function_iterable, y=self.data)
        toolbox.register("update", strategy.update)
        toolbox.register("generate", strategy.generate, creator.Individual)
        toolbox.decorate("evaluate", tools.DeltaPenalty(feasible, 1.0))
        #toolbox.register("generate", strategy.generate, generate)

        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        stats.register("med", np.median)
        hof = tools.HallOfFame(5)
        pop, logbook = algorithms.eaGenerateUpdate(toolbox, ngen=120, stats=stats, halloffame=hof, verbose=True)
        ###
        with open('results/cma_es_results.csv', 'w') as file:
            file.write('min;max;med;avg;std\n')
            for p in logbook:
                file.write(f"{p['min']};{p['max']};{p['med']};{p['avg']};{p['std']}\n")
        with open('results/cma_es_hof.csv', 'w') as file:
            file.write('x0;x1;x2\n')
            for p in hof:
                file.write(f'{p[0]};{p[1]};{p[1]}\n')
        return hof[0]

    def pso(self):
        options = {'c1': 0.5, 'c2': 0.25, 'w': 0.9}
        optimizer = pyswarms.single.GlobalBestPSO(n_particles=len(self.initials),
            dimensions=3, options=options, init_pos=self.initials, bounds=self.bounds_pso
        )
        def vectorized(x):
            v = np.vectorize(self.function_with_args, signature='(i)->()')
            return v(x)
        best_cost, best_pos = optimizer.optimize(vectorized, iters=200)
        print(f'{best_cost} | {best_pos}')
        ###
        with open('results/pso.csv', 'w') as file:
            file.write('f;mean_cost;x0;x1;x2\n')
            for f, m in zip(optimizer.cost_history, optimizer.mean_pbest_history):
                file.write(f'{f};{m};;;\n')
            file.write(f'{best_cost};;{best_pos[0]};{best_pos[1]};{best_pos[2]}')
        return best_pos

    def bayes(self):
        end_time = time.time() + (3600 * 10.0)
        progress = []
        initials = self.initials.tolist()

        def callback(x):
            print(f'{x.fun}  |  {x.x}')
            progress.append((x.fun, x.x[0], x.x[1], x.x[2]))
            t = time.time()
            if t > end_time:
                return True
            else:
                return False

        i = len(initials) // 2
        n_calls = 10000 + i + len(initials)
        print(i, n_calls)
        result = skopt.gp_minimize(self.function_with_args, self.bounds, n_calls=n_calls,
            callback=callback, n_jobs=1, n_initial_points=i, initial_point_generator='hammersly',
            acq_func='LCB', kappa=314.0, x0=initials, verbose=True
        )
        ###
        with open('results/bayes.csv', 'w') as file:
            file.write('f;x0;x1;x2\n')
            for p in progress:
                file.write(f'{p[0]};{p[1]};{p[2]};{p[3]}\n')
            file.write(f'{result.fun};{result.x[0]};{result.x[1]};{result.x[2]}')
        return result.x


def main():
    opt = Optimizer()
    t0 = time.time()
    #opt.function_with_args(opt.initial)
    #print(opt.function_with_args(opt.initial))
    #opt.brute()
    #opt.powell()
    #opt.bfgs()
    #opt.dual_annealing()
    opt.differential_evolution()
    #opt.cma_es()
    #opt.pso()
    #opt.bayes()
    t1 = time.time()
    print(t1 - t0)
    print(timedelta(seconds=t1-t0))
    # PSO, DE -> 19 iter / h


if __name__ == "__main__":
    main()
