import numpy as np
import random
from tqdm import tqdm
from algorithm import AlgorithmBase
import time
import networkx as nx
from concurrent.futures import ThreadPoolExecutor
import threading


class MARIM(AlgorithmBase):
    """
    MA-RIM 单层网络版本（只在第一层或单层网络上运行）
    高速运行，保持论文逻辑：
    - 初始种群 + 度数轮盘赌 + 距离最大化初始化
    - 精英保留 + 交叉 + 变异 + 严格论文三阶段局部搜索
    - 支持多次模拟取均值的鲁棒评估
    - 并行评估 + 线程安全缓存
    """

    class SingleLayerWrapper:
        """包装单层 nx.Graph，使其兼容多层网络接口"""

        def __init__(self, G):
            self.layers = [G]

        def number_of_layers(self):
            return 1

    def __init__(self, diffusion_model, pop_size=80, pop0_size=60, max_generations=150,
                 p_crossover=0.6, p_mutation=0.4, p_local_search=0.5, verbose=True, seed=None,
                 use_fast_evaluation=True,
                 num_robust_evaluations=5, n_jobs=None):
        super().__init__()
        self.diffusion_model = diffusion_model
        self.pop_size = pop_size
        self.pop0_size = pop0_size
        self.max_generations = max_generations
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.p_local_search = p_local_search
        self.verbose = verbose
        self.seed = seed
        self.use_fast_evaluation = use_fast_evaluation
        self.num_robust_evaluations = num_robust_evaluations
        self.n_jobs = n_jobs or min(32, (self.pop_size + 1) // 2)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # 历史记录
        self.history = {
            "best_fitness": [],
            "avg_fitness": [],
            "delta_fitness": [],
            "seed_retention_ratio": [],
            "best_solution": None
        }

        # 缓存
        self.fitness_cache = {}
        self.degree_cache = {}
        self.neighbor_cache = {}
        self.nodes_list = []

        # 性能统计
        self.eval_count = 0
        self.cache_hits = 0

        # 线程锁保护缓存
        self.cache_lock = threading.Lock()

    # ------------------- 基础设置 -------------------
    def set_setting(self, network, k):
        self.reset()
        self.network = network
        self.k = k

        # 只取第一层网络作为搜索层
        if hasattr(network, 'layers') and network.layers:
            self.layer0 = network.layers[0]
        else:
            self.layer0 = network

        self.nodes_list = list(self.layer0.nodes())
        self.num_nodes = len(self.nodes_list)
        self.node_to_idx = {n: i for i, n in enumerate(self.nodes_list)}

        self._precompute_degrees()
        self._precompute_neighbors()
        self.sorted_by_degree = sorted(
            self.nodes_list, key=lambda x: self.degree_cache[x], reverse=True
        )

        self.fitness_cache.clear()

    def reset(self):
        self.network = None
        self.layer0 = None
        self.k = 0
        self.nodes_list = []
        self.num_nodes = 0
        self.node_to_idx = {}
        self.degree_cache = {}
        self.neighbor_cache = {}
        self.sorted_by_degree = []
        self.fitness_cache.clear()
        self.eval_count = 0
        self.cache_hits = 0

    def _precompute_degrees(self):
        self.degree_cache = {n: self.layer0.degree(n) for n in self.nodes_list}

    def _precompute_neighbors(self):
        self.neighbor_cache = {n: list(self.layer0.neighbors(n)) for n in self.nodes_list}

    # ------------------- 初始化 -------------------
    def _initialization_operator(self):
        population = []

        #  随机初始化（1/3）
        random_count = self.pop0_size // 3
        for _ in range(random_count):
            population.append(random.sample(self.nodes_list, self.k))

        #  度数轮盘赌初始化（1/3）
        roulette_count = self.pop0_size // 3
        if roulette_count > 0:
            weights = np.array([self.degree_cache[n] for n in self.nodes_list])
            total = weights.sum()
            probs = weights / total if total > 0 else np.ones_like(weights) / len(weights)

            for _ in range(roulette_count):
                selected = []
                available = self.nodes_list.copy()
                for _ in range(self.k):
                    if not available:
                        break
                    idxs = [self.node_to_idx[n] for n in available]
                    p = probs[idxs]
                    p = p / p.sum() if p.sum() > 0 else np.ones_like(p) / len(p)
                    idx = np.random.choice(len(available), p=p)
                    selected.append(available.pop(idx))

                if len(selected) < self.k:
                    remaining = [n for n in self.nodes_list if n not in selected]
                    selected.extend(random.sample(remaining, self.k - len(selected)))
                population.append(selected)

        #  距离最大化初始化
        remaining = self.pop0_size - len(population)
        for _ in range(remaining):
            ind = []
            top_candidates = self.sorted_by_degree[:min(self.k * 3, len(self.sorted_by_degree))]
            if top_candidates:
                ind.append(random.choice(top_candidates[:self.k]))

            while len(ind) < self.k:
                rest = [n for n in self.nodes_list if n not in ind]
                if rest:
                    ind.append(random.choice(rest))
                else:
                    break
            population.append(ind)

        return population

    # ------------------- 适应度评估 -------------------
    def _evaluate_fitness(self, individual):
        key = tuple(sorted(individual))
        with self.cache_lock:
            if key in self.fitness_cache:
                self.cache_hits += 1
                return self.fitness_cache[key]

        vals = []
        for _ in range(self.num_robust_evaluations):
            if hasattr(self.diffusion_model, 'approx_func_single_layer_robust'):
                v = self.diffusion_model.approx_func_single_layer_robust(
                    self.layer0,
                    individual
                )
            else:
                v = sum(self.degree_cache.get(n, 0) for n in individual)
            vals.append(v)

        fitness = np.mean(vals)

        with self.cache_lock:
            self.fitness_cache[key] = fitness
            self.eval_count += 1
        return fitness

    def _evaluate_fitness_batch(self, population):
        # 使用线程池并行
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            results = list(executor.map(self._evaluate_fitness, population))
        return results

    # ------------------- 交叉 & 修复 -------------------
    def _crossover_operator(self, parent1, parent2):
        if random.random() > self.p_crossover or self.k < 2:
            return parent1.copy(), parent2.copy()
        a, b = sorted(random.sample(range(self.k), 2))
        child1 = parent1[:a] + parent2[a:b] + parent1[b:]
        child2 = parent2[:a] + parent1[a:b] + parent2[b:]
        return self._repair(child1), self._repair(child2)

    def _repair(self, individual):
        unique = list(dict.fromkeys(individual))
        while len(unique) < self.k:
            available = [n for n in self.nodes_list if n not in unique]
            if available:
                unique.append(random.choice(available))
            else:
                unique.append(random.choice(self.nodes_list))
        return unique

    # ------------------- 变异 -------------------
    def _mutation_operator(self, individual):
        if random.random() > self.p_mutation:
            return individual.copy()
        idx = random.randint(0, self.k - 1)
        available = [n for n in self.nodes_list if n not in individual]
        if not available:
            return individual.copy()
        individual[idx] = random.choice(available)
        return self._repair(individual)

    # ------------------- 局部搜索 -------------------
    def _local_search_fast(self, individual, fitness, generation):
        best_sol = individual.copy()
        best_fit = fitness
        MaxGen = self.max_generations
        t = generation
        layer = self.layer0

        # ========= Stage 1：邻居搜索 =========
        for idx in range(self.k):
            seed = best_sol[idx]
            neighbors = self.neighbor_cache.get(seed, [])
            valid = [n for n in neighbors if n not in best_sol]
            if not valid:
                continue

            cand_scores = []
            for n in valid:
                deg = self.degree_cache.get(n, 0)
                try:
                    dis = nx.shortest_path_length(layer, source=n, target=seed)
                except nx.NetworkXNoPath:
                    dis = 0
                score = deg * (MaxGen - t) + dis * t
                cand_scores.append((score, n))

            # 单层等价于 L=1
            selected = [n for _, n in sorted(cand_scores, reverse=True)[:1]]

            for n in selected:
                cand = best_sol.copy()
                cand[idx] = n
                cand = self._repair(cand)
                f = self._evaluate_fitness(cand)
                if f > best_fit:
                    best_sol, best_fit = cand, f
                    break

        # ========= Stage 2：全局搜索（你原来缺的） =========
        smin_idx = np.argmin([self.degree_cache[n] for n in best_sol])

        max_deg_nodes = sorted(layer.nodes(), key=lambda n: layer.degree(n), reverse=True)
        if max_deg_nodes:
            n = random.choice(max_deg_nodes[:min(5, len(max_deg_nodes))])
            if n not in best_sol:
                cand = best_sol.copy()
                cand[smin_idx] = n
                cand = self._repair(cand)
                f = self._evaluate_fitness(cand)

                pl_global = self.p_local_search * (MaxGen - t) / MaxGen
                if f > best_fit and random.random() < pl_global:
                    best_sol, best_fit = cand, f

        # ========= Stage 3：随机搜索 =========
        pl_random = self.p_local_search * t / MaxGen
        if random.random() < pl_random:
            for idx in random.sample(range(self.k), min(2, self.k)):
                n = random.choice(self.nodes_list)
                cand = best_sol.copy()
                cand[idx] = n
                cand = self._repair(cand)
                f = self._evaluate_fitness(cand)
                if f > best_fit:
                    best_sol, best_fit = cand, f

        return best_sol, best_fit

    # ------------------- 锦标赛选择 -------------------
    def _tournament_selection(self, population, fitness_values):
        idxs = random.sample(range(len(population)), 2)
        return population[idxs[0]] if fitness_values[idxs[0]] >= fitness_values[idxs[1]] else population[idxs[1]]

    def _update_seed_retention_ratio(self, initial_best, current_best):
        ratio = len(set(initial_best) & set(current_best)) / len(initial_best) if initial_best else 0.0
        self.history["seed_retention_ratio"].append(ratio)

    # ------------------- 主运行 -------------------
    def run(self, network, k=None):
        start_time = time.time()
        if k is not None:
            self.k = k
        self.set_setting(network, self.k)

        if self.verbose:
            print(f"MA-RIM 单层网络版本开始运行")
            print(f"节点数: {self.num_nodes}, 种子数: {self.k}")
            print(f"种群大小: {self.pop_size}, 最大代数: {self.max_generations}")

        population = self._initialization_operator()
        fitness_values = self._evaluate_fitness_batch(population)

        best_idx = np.argmax(fitness_values)
        best_solution = population[best_idx].copy()
        best_fitness = fitness_values[best_idx]
        initial_best = best_solution.copy()

        self._update_seed_retention_ratio(initial_best, best_solution)
        self.history["best_fitness"].append(best_fitness)
        self.history["avg_fitness"].append(np.mean(fitness_values))
        self.history["delta_fitness"].append(0.0)
        self.history["best_solution"] = best_solution

        loop = tqdm(range(self.max_generations), desc="MA-RIM 迭代", disable=not self.verbose)
        prev_best_fitness = best_fitness

        for gen in loop:
            elite_idx = np.argsort(fitness_values)[-max(1, int(self.pop_size * 0.05)):]
            new_population = [population[i] for i in elite_idx]

            while len(new_population) < self.pop_size:
                parent1 = self._tournament_selection(population, fitness_values)
                parent2 = self._tournament_selection(population, fitness_values)
                child1, child2 = self._crossover_operator(parent1, parent2)
                child1 = self._mutation_operator(child1)
                child2 = self._mutation_operator(child2)

                # 并行局部搜索
                with ThreadPoolExecutor(max_workers=2) as executor:
                    results = list(executor.map(
                        lambda args: self._local_search_fast(*args),
                        [(child1, self._evaluate_fitness(child1), gen),
                         (child2, self._evaluate_fitness(child2), gen)]
                    ))
                child1, child2 = results[0][0], results[1][0]
                new_population.extend([child1, child2])

            new_fitness = self._evaluate_fitness_batch(new_population)
            combined_pop = population + new_population
            combined_fit = fitness_values + new_fitness
            sorted_idx = np.argsort(combined_fit)[::-1][:self.pop_size]
            population = [combined_pop[i] for i in sorted_idx]
            fitness_values = [combined_fit[i] for i in sorted_idx]

            gen_best_idx = np.argmax(fitness_values)
            gen_best_solution = population[gen_best_idx].copy()
            gen_best_fitness = fitness_values[gen_best_idx]

            self._update_seed_retention_ratio(initial_best, gen_best_solution)
            delta = gen_best_fitness - prev_best_fitness
            self.history["delta_fitness"].append(delta)
            self.history["best_fitness"].append(gen_best_fitness)
            self.history["avg_fitness"].append(np.mean(fitness_values))

            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_solution = gen_best_solution.copy()

            prev_best_fitness = best_fitness

            if self.verbose:
                loop.set_postfix({
                    "最佳": f"{best_fitness:.2f}",
                    "平均": f"{np.mean(fitness_values):.2f}",
                    "delta": f"{delta:.2f}",
                    "保留比例": f"{self.history['seed_retention_ratio'][-1] * 100:.1f}%"
                })

        self.history["best_solution"] = best_solution
        total_time = time.time() - start_time
        if self.verbose:
            print(
                f"\nMA-RIM 完成! 用时: {total_time:.2f}s, 评估次数: {self.eval_count}, 缓存命中率: {self.cache_hits / (self.eval_count + self.cache_hits) * 100:.1f}%")
            print(f"最佳影响力: {best_fitness:.4f}")
            print(f"最佳种子集: {sorted(best_solution)}")
        return best_solution

    def get_name(self):
        return "MA-RIM"

    def get_convergence_data(self):
        return {
            "generations": list(range(len(self.history["best_fitness"]))),
            "best_fitness": self.history["best_fitness"],
            "avg_fitness": self.history["avg_fitness"],
            "delta_fitness": self.history["delta_fitness"],
            "seed_retention_ratio": self.history["seed_retention_ratio"],
            "best_solution": self.history["best_solution"]
        }
