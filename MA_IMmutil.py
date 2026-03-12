import numpy as np
import random
from tqdm import tqdm
from algorithm import AlgorithmBase
from joblib import Parallel, delayed
from typing import List
import networkx as nx


class MAIMmutil(AlgorithmBase):
    def __init__(self, diffusion_model, pop_size=50, max_generations=150,
                 p_crossover=0.6, p_mutation=0.4, p_local_search=0.5,
                 verbose=True, seed=None, n_jobs=-1,
                 use_fitness_cache=True,
                 max_local_search_tries=10,
                 tournament_size=3,
                 num_robust_evaluations=5):
        super().__init__()
        self.diffusion_model = diffusion_model
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.p_local_search = p_local_search
        self.verbose = verbose
        self.seed = seed
        self.n_jobs = n_jobs
        self.use_fitness_cache = use_fitness_cache
        self.max_local_search_tries = max_local_search_tries
        self.tournament_size = tournament_size
        self.num_robust_evaluations = num_robust_evaluations

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # ===== history（与 MA-RIMmulti 对齐）=====
        self.history = {
            "best_fitness": [],
            "avg_fitness": [],
            "delta_fitness": [],
            "seed_retention_ratio": [],
            "best_solution": None
        }

        # 状态变量
        self.fitness_cache = {}
        self.network = None
        self.k = 0
        self.nodes = []
        self.num_nodes = 0
        self.neighbors = {}
        self.synthetic_degrees = {}
        self.nodes_sorted_by_degree = []
        self.top_k_nodes = []

        # 缓存
        self.two_hop_cache = {}
        self.node_cache = {}

    # ---------------- 设置 ----------------
    def set_setting(self, network, k):
        self.reset()
        self.network = network
        self.k = k

        nodes = set()
        if hasattr(network, 'layers'):
            for layer in network.layers:
                nodes.update(layer.nodes())
        else:
            nodes.update(network.nodes())

        self.nodes = list(nodes)
        self.num_nodes = len(self.nodes)

        self._precompute_neighbors_and_degrees()
        self._precompute_node_ranks()

    def reset(self):
        self.network = None
        self.k = 0
        self.nodes = []
        self.num_nodes = 0
        self.neighbors = {}
        self.synthetic_degrees = {}
        self.nodes_sorted_by_degree = []
        self.top_k_nodes = []
        self.fitness_cache.clear()
        self.two_hop_cache.clear()
        self.node_cache.clear()

    # ---------------- 预计算 ----------------
    def _precompute_neighbors_and_degrees(self):
        self.neighbors = {}
        self.synthetic_degrees = {}

        if hasattr(self.network, 'layers'):
            for node in self.nodes:
                self.neighbors[node] = set()
                self.synthetic_degrees[node] = 0
            for layer in self.network.layers:
                for node in layer.nodes():
                    nbs = set(layer.neighbors(node))
                    self.neighbors[node].update(nbs)
                    self.synthetic_degrees[node] += len(nbs)
            for node in self.neighbors:
                self.neighbors[node] = list(self.neighbors[node])
        else:
            for node in self.nodes:
                nbs = list(self.network.neighbors(node))
                self.neighbors[node] = nbs
                self.synthetic_degrees[node] = len(nbs)

    def _precompute_node_ranks(self):
        self.nodes_sorted_by_degree = sorted(
            self.nodes,
            key=lambda x: self.synthetic_degrees[x],
            reverse=True
        )
        self.top_k_nodes = self.nodes_sorted_by_degree[:min(100, self.num_nodes)]

    # ---------------- 距离 ----------------
    def _get_2hop_neighbors(self, node):
        if node in self.two_hop_cache:
            return self.two_hop_cache[node]
        two_hop = set(self.neighbors.get(node, []))
        for n in list(two_hop):
            two_hop.update(self.neighbors.get(n, []))
        two_hop.discard(node)
        self.two_hop_cache[node] = two_hop
        return two_hop

    def _distance_to_set(self, v, S):
        if not S:
            return 1.0
        key = tuple(sorted(S))
        if key not in self.node_cache:
            r = set()
            for s in S:
                r.update(self._get_2hop_neighbors(s))
            self.node_cache[key] = r
        v_range = self._get_2hop_neighbors(v)
        return len(v_range ^ self.node_cache[key]) / self.num_nodes

    # ---------------- 适应度 ----------------
    def _evaluate_fitness(self, individual):
        key = tuple(sorted(individual))
        if self.use_fitness_cache and key in self.fitness_cache:
            return self.fitness_cache[key]

        vals = []
        for _ in range(self.num_robust_evaluations):
            try:
                vals.append(self.diffusion_model.approx_func(self.network, list(individual)))
            except Exception:
                vals.append(0.0)

        fit = np.mean(vals)
        if self.use_fitness_cache:
            self.fitness_cache[key] = fit
        return fit

    # ---------------- 初始化 ----------------
    def _seed_initialization(self):
        pop = []
        for _ in range(self.pop_size // 3):
            pop.append(random.sample(self.nodes, self.k))
        for _ in range(self.pop_size - len(pop)):
            ind = [random.choice(self.top_k_nodes)]
            while len(ind) < self.k:
                cand = max(
                    [n for n in self.nodes if n not in ind],
                    key=lambda x: self._distance_to_set(x, ind)
                )
                ind.append(cand)
            pop.append(ind)
        return pop

    # ---------------- 遗传操作 ----------------
    def _tournament_selection(self, population, fitness):
        idxs = random.sample(range(len(population)), self.tournament_size)
        best = max(idxs, key=lambda i: fitness[i])
        return population[best]

    def _crossover(self, p1, p2):
        if random.random() > self.p_crossover:
            return p1.copy(), p2.copy()
        a, b = sorted(random.sample(range(self.k), 2))
        c1 = list(dict.fromkeys(p1[:a] + p2[a:b] + p1[b:]))
        c2 = list(dict.fromkeys(p2[:a] + p1[a:b] + p2[b:]))
        return self._repair(c1), self._repair(c2)

    def _mutation(self, ind):
        if random.random() > self.p_mutation:
            return ind
        i = random.randrange(self.k)
        cand = random.choice([n for n in self.nodes if n not in ind])
        ind[i] = cand
        return self._repair(ind)

    def _repair(self, ind):
        ind = list(dict.fromkeys(ind))
        while len(ind) < self.k:
            ind.append(random.choice(self.nodes))
        return ind

    def _local_search(self, individual, fitness, generation=None):
        """
        - 每一轮替换仅与“当前原始种子集 p”比较
        """
        # 当前个体 p（论文中的 seed set）
        p = individual.copy()
        # 当前个体的影响力值 σ̂multi(p)
        p_fit = fitness

        # 多层网络的所有层（单层时退化为一个 layer）
        layers = self.network.layers if hasattr(self.network, 'layers') else [self.network]

        # Stage 1
        # 对种子集中的每一个 seed 逐个进行邻居替换尝试
        for idx in range(self.k):
            # 当前处理的第 idx 个种子 s_i
            s_i = p[idx]
            # 存储各层产生的候选种子集及其影响力
            candidate_sets = []

            # 在每一层网络中分别尝试替换
            for layer in layers:
                # 获取 s_i 在该层的所有邻居节点
                neighbors = list(layer.neighbors(s_i))
                # 去除已经在种子集中的节点，保证有效性
                neighbors = [n for n in neighbors if n not in p]
                if not neighbors:
                    continue

                # 在该层中选择度最大的邻居节点 n_max
                n_max = max(neighbors, key=lambda n: layer.degree(n))

                # 构造候选种子集 S_l
                S_l = p.copy()
                # 用 n_max 替换第 idx 个种子
                S_l[idx] = n_max
                # 修复种子集（去重并补齐 k 个节点）
                S_l = self._repair(S_l)

                # 计算候选种子集的鲁棒影响力 σ̂multi(S_l)
                f_l = self._evaluate_fitness(S_l)
                candidate_sets.append((f_l, S_l))

            # 从所有层产生的候选种子集中选取影响力最大的一个
            # 对应论文 Algorithm 2 第 5 行
            if candidate_sets:
                f_best, S_best = max(candidate_sets, key=lambda x: x[0])
                # 仅当候选解优于“当前原始种子集 p”时才接受
                # 对应论文第 6–8 行
                if f_best > p_fit:
                    p, p_fit = S_best, f_best

        #  Stage 2

        # 找到种子集中 synthetic degree 最小的种子 s
        smin_idx = np.argmin([self.synthetic_degrees[n] for n in p])

        # 随机选择一层网络
        layer = random.choice(layers)

        # 在该层中选取度最大的节点 n_l
        n_l = max(layer.nodes(), key=lambda n: layer.degree(n))

        # 若该节点尚未在种子集中，则尝试替换
        if n_l not in p:
            # 构造候选种子集 S_t
            S_t = p.copy()
            S_t[smin_idx] = n_l
            # 修复有效性
            S_t = self._repair(S_t)

            # 计算新种子集的影响力
            f_t = self._evaluate_fitness(S_t)
            # 仅当影响力提升时才接受替换
            if f_t > p_fit:
                p, p_fit = S_t, f_t

        # 返回局部搜索后的种子集 p′ 及其影响力
        return p, p_fit

    # ---------------- 主循环 ----------------
    def run(self, network, k):
        self.set_setting(network, k)

        population = self._seed_initialization()
        fitness = [self._evaluate_fitness(ind) for ind in population]

        best_idx = np.argmax(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        initial_best = best_solution.copy()

        # 初代 history
        self.history["best_fitness"].append(best_fitness)
        self.history["avg_fitness"].append(np.mean(fitness))
        self.history["delta_fitness"].append(0.0)
        self.history["seed_retention_ratio"].append(1.0)

        loop = tqdm(range(self.max_generations), disable=not self.verbose)

        for _ in loop:
            new_pop = []
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_selection(population, fitness)
                p2 = self._tournament_selection(population, fitness)
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutation(c1)
                c2 = self._mutation(c2)
                f1 = self._evaluate_fitness(c1)
                f2 = self._evaluate_fitness(c2)
                c1, f1 = self._local_search(c1, f1)
                c2, f2 = self._local_search(c2, f2)
                new_pop.extend([(c1, f1), (c2, f2)])

            combined = list(zip(population, fitness)) + new_pop
            combined.sort(key=lambda x: x[1], reverse=True)
            population = [c[0] for c in combined[:self.pop_size]]
            fitness = [c[1] for c in combined[:self.pop_size]]

            prev_best = self.history["best_fitness"][-1]
            if fitness[0] > best_fitness:
                best_fitness = fitness[0]
                best_solution = population[0].copy()

            self.history["best_fitness"].append(best_fitness)
            self.history["avg_fitness"].append(np.mean(fitness))
            self.history["delta_fitness"].append(best_fitness - prev_best)
            self.history["seed_retention_ratio"].append(
                len(set(best_solution) & set(initial_best)) / len(initial_best)
            )

            loop.set_postfix({
                "best": f"{best_fitness:.2f}",
                "avg": f"{np.mean(fitness):.2f}"
            })

        self.history["best_solution"] = best_solution
        return best_solution

    def get_name(self):
        return "MA-IMmulti"
