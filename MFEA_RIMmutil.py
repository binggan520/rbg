import numpy as np
import random
from tqdm import tqdm
from joblib import Parallel, delayed
from algorithm import AlgorithmBase


class MFEA_RIMmutil(AlgorithmBase):
    """
    MFEA-RIMm
    多层网络鲁棒影响力最大化
    """

    def __init__(self,
                 diffusion_model,
                 pop0=30,
                 pop_size=50,
                 max_generations=150,
                 pc=0.6,
                 pm=0.4,
                 pl=0.6,
                 Nei=3,
                 verbose=True,
                 seed=None,
                 n_jobs=-1,
                 num_robust_evaluations=5):

        super().__init__()

        self.diffusion_model = diffusion_model
        self.pop0 = pop0
        self.pop_size = pop_size
        self.max_generations = max_generations

        self.pc = pc
        self.pm = pm
        self.pl = pl
        self.Nei = Nei

        self.verbose = verbose
        self.n_jobs = n_jobs
        self.num_robust_evaluations = num_robust_evaluations

        self.fitness_cache = {}

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.history = {
            "best_fitness": [],
            "avg_fitness": [],
            "best_solution": None,
            "seed_retention_ratio": [],
            "cache_hit": 0,
            "cache_miss": 0
        }

    # --------------------------------------------------
    # 设置任务
    # --------------------------------------------------

    def set_setting(self, network, k):

        if not hasattr(network, "layers"):
            raise ValueError("network 必须包含 layers")

        self.network = network
        self.layers = network.layers

        self.nodes = list(set().union(*[g.nodes() for g in self.layers]))

        self.k = k
        self.T = len(self.layers)

    # --------------------------------------------------
    # 综合度
    # --------------------------------------------------

    def _synthetic_degree(self, node):

        deg = 0
        for g in self.layers:
            if node in g:
                deg += g.degree(node)

        return deg

    # --------------------------------------------------
    # seed distance
    # --------------------------------------------------

    def _seed_distance(self, s1, s2):

        n1 = set()
        n2 = set()

        for g in self.layers:

            if s1 in g:
                n1 |= set(g.neighbors(s1))

            if s2 in g:
                n2 |= set(g.neighbors(s2))

        return len(n1 ^ n2)

    # --------------------------------------------------
    # repair
    # --------------------------------------------------

    def _repair(self, ind):

        ind = list(dict.fromkeys(ind))

        while len(ind) < self.k:
            ind.append(random.choice(self.nodes))

        return ind[:self.k]

    # --------------------------------------------------
    # evaluate（多次鲁棒评估 + cache）
    # --------------------------------------------------

    def _evaluate(self, ind):

        key = (tuple(sorted(ind)), self.num_robust_evaluations)

        if key in self.fitness_cache:
            self.history["cache_hit"] += 1
            return self.fitness_cache[key]

        self.history["cache_miss"] += 1

        vals = []

        for _ in range(self.num_robust_evaluations):

            if hasattr(self.diffusion_model, "approx_func_fast"):
                v = self.diffusion_model.approx_func_fast(self.network, ind)
            else:
                v = self.diffusion_model.approx_func(self.network, ind)

            vals.append(v)

        val = np.mean(vals)

        cost = -val

        self.fitness_cache[key] = cost

        return cost

    # --------------------------------------------------
    # 初始化
    # --------------------------------------------------

    def _initialize_population(self):

        population = []

        part = self.pop0 // 3

        # 随机初始化
        for _ in range(part):
            population.append(random.sample(self.nodes, self.k))

        # degree初始化
        deg = np.array([self._synthetic_degree(n) for n in self.nodes])
        prob = deg / deg.sum()

        for _ in range(part):

            seeds = list(np.random.choice(self.nodes,
                                          self.k,
                                          replace=False,
                                          p=prob))

            population.append(seeds)

        # distance初始化
        for _ in range(self.pop0 - 2 * part):

            seeds = [random.choice(self.nodes)]

            while len(seeds) < self.k:

                candidates = random.sample(self.nodes, 10)

                best_node = None
                best_dist = -1

                for c in candidates:

                    dist = min(self._seed_distance(c, s) for s in seeds)

                    if dist > best_dist:
                        best_dist = dist
                        best_node = c

                seeds.append(best_node)

            population.append(seeds)

        while len(population) < self.pop_size:
            population.append(random.sample(self.nodes, self.k))

        return population

    # --------------------------------------------------
    # factorial rank
    # --------------------------------------------------

    def _factorial_rank(self, cost):

        pop_size = len(cost)

        ranks = np.zeros((pop_size, self.T))

        for t in range(self.T):

            order = np.argsort([cost[i][t] for i in range(pop_size)])

            for r, idx in enumerate(order):
                ranks[idx][t] = r + 1

        return ranks

    # --------------------------------------------------
    # scalar fitness
    # --------------------------------------------------

    def _scalar_fitness(self, ranks):

        return np.array([1 / min(r) for r in ranks])

    # --------------------------------------------------
    # skill factor
    # --------------------------------------------------

    def _skill_factor(self, ranks):

        return np.array([np.argmin(r) for r in ranks])

    # --------------------------------------------------
    # neighbor niching
    # --------------------------------------------------

    def _find_neighbors(self, population):

        neighbors = []

        for i, p in enumerate(population):

            dist_list = []

            for j, q in enumerate(population):

                if i == j:
                    continue

                dist = sum(self._seed_distance(a, b)
                           for a, b in zip(p, q))

                dist_list.append((dist, j))

            dist_list.sort()

            neighbors.append([j for _, j in dist_list[:self.Nei]])

        return neighbors

    # --------------------------------------------------
    # crossover
    # --------------------------------------------------

    def _crossover(self, p1, p2, sf1, sf2):

        child = p1.copy()

        if sf1 == sf2:

            for i in range(self.k):
                if random.random() < 0.5:
                    child[i] = p2[i]

        else:

            pos = random.randint(0, self.k - 1)
            child[pos] = random.choice(p2)

        return self._repair(child)

    # --------------------------------------------------
    # mutation
    # --------------------------------------------------

    def _mutation(self, ind):

        if random.random() > self.pm:
            return ind

        ind = ind.copy()

        i = random.randint(0, self.k - 1)

        ind[i] = random.choice(self.nodes)

        return self._repair(ind)

    # --------------------------------------------------
    # 主算法
    # --------------------------------------------------

    def run(self, network, k):

        self.set_setting(network, k)

        population = self._initialize_population()

        initial_best = population[0]

        # 并行评估初始化种群
        cost_values = Parallel(n_jobs=self.n_jobs)(
            delayed(self._evaluate)(ind) for ind in population
        )

        cost = [[v] * self.T for v in cost_values]

        loop = tqdm(range(self.max_generations),
                    disable=not self.verbose)

        for gen in loop:

            neighbors = self._find_neighbors(population)

            ranks = self._factorial_rank(cost)

            scalar_fitness = self._scalar_fitness(ranks)

            skill_factor = self._skill_factor(ranks)

            offspring = []

            while len(offspring) < self.pop_size:

                i = random.randint(0, len(population) - 1)

                if random.random() < self.pc:

                    if random.random() < 0.5:
                        j = random.choice(neighbors[i])
                    else:
                        j = random.randint(0, len(population) - 1)

                    child = self._crossover(population[i],
                                            population[j],
                                            skill_factor[i],
                                            skill_factor[j])

                else:
                    child = population[i].copy()

                child = self._mutation(child)

                offspring.append(child)

            # 并行评估 offspring
            offspring_values = Parallel(n_jobs=self.n_jobs)(
                delayed(self._evaluate)(ind) for ind in offspring
            )

            offspring_cost = [[v] * self.T for v in offspring_values]

            population += offspring
            cost += offspring_cost

            ranks = self._factorial_rank(cost)

            scalar_fitness = self._scalar_fitness(ranks)

            idx = np.argsort(scalar_fitness)[::-1][:self.pop_size]

            population = [population[i] for i in idx]
            cost = [cost[i] for i in idx]

            real_fit = [-c[0] for c in cost]

            best = np.max(real_fit)
            avg = np.mean(real_fit)

            self.history["best_fitness"].append(best)
            self.history["avg_fitness"].append(avg)

            current_best = population[0]

            overlap = len(set(initial_best) & set(current_best))

            ratio = overlap / self.k

            self.history["seed_retention_ratio"].append(ratio)

        self.history["best_solution"] = population[0]

        return population[0]

    def get_name(self):
        return "MFEA_RIMmutil"

    def get_convergence_data(self):

        hit = self.history["cache_hit"]
        miss = self.history["cache_miss"]

        if hit + miss > 0:
            rate = hit / (hit + miss) * 100
        else:
            rate = 0

        print(f"Cache hit rate: {rate:.2f}%")

        return self.history