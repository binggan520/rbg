import random
import time
import numpy as np
from tqdm import tqdm
from algorithm import AlgorithmBase

class GA(AlgorithmBase):
    """
    遗传算法 GA，实现：
    - 精英保留 + 锦标赛选择 + 交叉 + 变异
    - 鲁棒影响力评估（多次取平均）
    """

    def __init__(self, diffusion_model, pop_size=100, generations=200, elites=2,
                 crossover_rate=0.6, mutation_rate=0.4, tournament_size=3,
                 num_robust_evaluations=5,   # 👈 新增参数
                 seed=None, verbose=True):
        super().__init__()
        self._diffusion_model = diffusion_model
        self._diffusion_model.set_verbose(False)
        self._verbose = verbose

        self.pop_size = pop_size
        self.generations = generations
        self.elites = elites
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.num_robust_evaluations = num_robust_evaluations  # 👈 保存参数
        self.seed = seed

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # -------------------- 历史记录 --------------------
        self.history = {
            "best_fitness": [],
            "avg_fitness": [],
            "delta_fitness": [],
            "seed_retention_ratio": [],
            "best_solution": None
        }

    # -------------------- 种子保留比例 --------------------
    def _update_seed_retention_ratio(self, initial_best, current_best):
        ratio = len(set(current_best) & set(initial_best)) / len(initial_best) if initial_best else 0.0
        self.history["seed_retention_ratio"].append(ratio)

    # -------------------- 核心操作 --------------------
    def _random_individual(self, network, k):
        return random.sample(list(network.nodes()), k)

    def _fitness(self, network, individual):
        # 👇 唯一修改：多次评估取平均（鲁棒性）
        total = 0.0
        for _ in range(self.num_robust_evaluations):
            total += self._diffusion_model.approx_func(network, individual)
        return total / self.num_robust_evaluations

    def _crossover(self, network, a, b, k):
        if random.random() >= self.crossover_rate:
            return a.copy() if random.random() < 0.5 else b.copy()
        common = set(a) & set(b)
        child = list(common)
        diff = (set(a) | set(b)) - common
        diff_list = list(diff)
        random.shuffle(diff_list)
        needed = k - len(child)
        if needed > 0 and diff_list:
            child.extend(diff_list[:min(needed, len(diff_list))])
        if len(child) < k:
            available = set(network.nodes()) - set(child)
            if available:
                child.extend(random.sample(list(available), min(k - len(child), len(available))))
        return child[:k]

    def _mutate(self, network, individual, k):
        indiv = individual.copy()
        for i in range(len(indiv)):
            if random.random() < self.mutation_rate:
                pool = list(set(network.nodes()) - set(indiv))
                if pool:
                    indiv[i] = random.choice(pool)
        uniq = list(dict.fromkeys(indiv))
        while len(uniq) < k:
            available = set(network.nodes()) - set(uniq)
            if available:
                uniq.append(random.choice(list(available)))
            else:
                break
        return uniq[:k]

    def _tournament_select(self, population, fitnesses):
        contenders = random.sample(range(len(population)), self.tournament_size)
        best = max(contenders, key=lambda i: fitnesses[i])
        return population[best]

    # -------------------- 主运行 --------------------
    def run(self, network, k):
        pop_size = min(self.pop_size, network.number_of_nodes())
        population = [self._random_individual(network, k) for _ in range(pop_size)]
        fitnesses = [self._fitness(network, ind) for ind in population]

        best_idx = np.argmax(fitnesses)
        best_individual_overall = population[best_idx]
        best_fitness_overall = fitnesses[best_idx]

        self.history["best_solution"] = best_individual_overall
        self.history["best_fitness"].append(best_fitness_overall)
        self.history["avg_fitness"].append(np.mean(fitnesses))
        self.history["delta_fitness"].append(0.0)

        initial_best = best_individual_overall.copy()
        self._update_seed_retention_ratio(initial_best, best_individual_overall)

        loop = tqdm(range(self.generations), disable=not self._verbose)
        prev_best = best_fitness_overall

        for gen in loop:
            ranked = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)
            newpop = [population[i] for i in ranked[:self.elites]]

            while len(newpop) < pop_size:
                p1 = self._tournament_select(population, fitnesses)
                p2 = self._tournament_select(population, fitnesses)
                child = self._crossover(network, p1, p2, k)
                child = self._mutate(network, child, k)
                newpop.append(child)

            population = newpop
            fitnesses = [self._fitness(network, ind) for ind in population]

            gen_best_idx = np.argmax(fitnesses)
            gen_best_solution = population[gen_best_idx]
            gen_best_fitness = fitnesses[gen_best_idx]

            self.history["best_fitness"].append(gen_best_fitness)
            self.history["avg_fitness"].append(np.mean(fitnesses))
            self.history["delta_fitness"].append(gen_best_fitness - prev_best)
            self._update_seed_retention_ratio(initial_best, gen_best_solution)

            if gen_best_fitness > best_fitness_overall:
                best_fitness_overall = gen_best_fitness
                best_individual_overall = gen_best_solution

            prev_best = best_fitness_overall

            if self._verbose:
                loop.set_postfix({
                    "best": f"{best_fitness_overall:.4f}",
                    "avg": f"{np.mean(fitnesses):.4f}",
                    "delta": f"{gen_best_fitness - prev_best:.4f}",
                    "seed_retention": f"{self.history['seed_retention_ratio'][-1]:.3f}"
                })

        self.history["best_solution"] = best_individual_overall
        return best_individual_overall

    # -------------------- 接口 --------------------
    def get_name(self):
        return "GA"

    def get_convergence_data(self):
        return {
            "generations": list(range(len(self.history["best_fitness"]))),
            "best_fitness": self.history["best_fitness"],
            "avg_fitness": self.history["avg_fitness"],
            "delta_fitness": self.history["delta_fitness"],
            "seed_retention_ratio": self.history.get("seed_retention_ratio", []),
            "best_solution": self.history["best_solution"]
        }