import argparse

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import copy
import datetime
import random
from collections import Counter
import time


# 计算网络鲁棒性值
def calculate_fitness(individual):
    copy_g = copy.deepcopy(G)
    for node in individual:
        copy_g.remove_node(node)
    sub_graph_list = list(nx.connected_components(copy_g))
    fitness = 0
    for sub_graph_i in range(len(sub_graph_list)):
        fitness += len(sub_graph_list[sub_graph_i]) * (len(sub_graph_list[sub_graph_i]) - 1) / 2
    return fitness


def pop_select(pops):
    fitness_list = []
    for pop_i in range(len(pops)):
        a = calculate_fitness(pops[pop_i])
        fitness_list.append(a)
    fitness_sort = np.array(fitness_list).argsort()
    # print(fitness_list[fitness_sort[0]], "\n", fitness_list, "\n", pops[fitness_sort[0]], "\n", pops)
    if fitness_list[fitness_sort[0]] < history[1]:
        history[0] = pops[fitness_sort[0]]
        history[1] = fitness_list[fitness_sort[0]]
    return pops[fitness_sort[0]]


def genes_select(pops):
    genes = []
    for pop in pops:
        genes = genes + list(pop)
    data = dict(Counter(genes))
    data = sorted(data.items(), key=lambda x: x[1])[::-1][:genes_num]
    genes = [data[i][0] for i in range(len(data))]
    genes = list(set(genes + list(history[0])))
    return genes


def crossover(cross_pop1, cross_pop2):
    for gene_i in range(k):
        if random.random() < node_crossover_rate:
            t = cross_pop1[gene_i]
            cross_pop1[gene_i] = cross_pop2[gene_i]
            cross_pop2[gene_i] = t
    # 个体内节点去重(未包含个体间去重)
    cross_pop1 = np.array(list(set(cross_pop1)))
    cross_pop2 = np.array(list(set(cross_pop2)))
    while len(cross_pop1) < k:
        while True:
            temp_node = random.sample(genes, 1)[0]
            if temp_node not in cross_pop1:
                cross_pop1 = np.append(cross_pop1, temp_node)
                break
    while len(cross_pop2) < k:
        while True:
            temp_node = random.sample(genes, 1)[0]
            if temp_node not in cross_pop2:
                cross_pop2 = np.append(cross_pop2, temp_node)
                break
    return cross_pop1, cross_pop2


def mutate(mutate_pop):
    for mutate_i in range(k):
        if random.random() < node_mutate_rate:
            mutate_pop[mutate_i] = random.sample(genes, 1)[0]
    mutate_pop = np.array(list(set(mutate_pop)))
    while len(mutate_pop) < k:
        while True:
            temp_node = random.sample(genes, 1)[0]
            if temp_node not in mutate_pop:
                mutate_pop = np.append(mutate_pop, temp_node)
                break
    return mutate_pop


def roulette(rotary_table):
    value = random.random()
    for roulette_i in range(len(rotary_table)):
        if rotary_table[roulette_i] >= value:
            return roulette_i


def calc_rotary_table(fit):
    if (max(fit) - min(fit)) != 0:
        fit = (fit - min(fit)) / (max(fit) - min(fit))
        fit = 1 - fit
        fit_interval = fit / fit.sum()
    else:
        fit_interval = np.array([1 / pops_size for i in range(pops_size)])
    rotary_table = []
    add = 0
    for i in range(len(fit_interval)):
        add = add + fit_interval[i]
        rotary_table.append(add)
    return rotary_table


def no_cutoff(genes):
    return genes


def random_cutoff(genes):
    genes = random.sample(genes, selected_genes_num)
    return genes


def greedy_cutoff(genes):
    copy_G = copy.deepcopy(G)
    while len(genes) > selected_genes_num:
        degree = nx.degree_centrality(copy_G)
        genes.remove(min(degree, key=degree.get))
        copy_G.remove_node(min(degree, key=degree.get))
    return genes


def pop_greedy_cutoff(genes, pop_num, pop_size, genes_num):
    greedy_indi = []
    for i in range(pop_num):
        temp_population = np.array([random.sample(genes, genes_num) for j in range(pop_size)])
        temp_fitness = np.array([calculate_fitness(individual) for individual in temp_population])
        top_index = temp_fitness.argsort()[0:1]
        greedy_indi.append(temp_population[top_index])

    genes = []
    for pop in greedy_indi:
        genes = genes + list(pop[0])
    data = dict(Counter(genes))
    data = sorted(data.items(), key=lambda x: x[1])[::-1][:int(selected_genes_num / 2)]
    genes = [data[i][0] for i in range(len(data))]
    # genes = list(set(genes + list(history[0])))

    # 在基因池中添加度值最高的节点
    degree = nx.degree_centrality(G)
    degree = sorted(degree.items(), key=lambda x: x[1])[::-1]
    degree = [degree[i][0] for i in range(len(degree))]
    count_i = 0
    while len(genes) < selected_genes_num:
        if degree[count_i] not in genes:
            genes.append(degree[count_i])
        count_i = count_i + 1
    return genes


def local_optimal_cutoff(genes, pop_num, pop_size, genes_num):
    greedy_indi = []
    for i in range(pop_num):
        temp_population = np.array([random.sample(genes, genes_num) for j in range(pop_size)])
        temp_fitness = np.array([calculate_fitness(individual) for individual in temp_population])
        # top_index = temp_fitness.argsort()[0:1]
        # greedy_indi.append(temp_population[top_index])
        for generation in range(100):
            print(i, "  ", generation)
            rotary_table = calc_rotary_table(temp_fitness)
            # 精英保留
            top_index = temp_fitness.argsort()[0:int(0.1 * pop_size)]
            new_population = list(temp_population[top_index])
            new_fitness = list(temp_fitness[top_index])
            # 交叉变异
            for cross_i in range(int(pops_size * 0.9 / 2)):
                cross_pop1 = copy.deepcopy(temp_population[roulette(rotary_table)])
                cross_pop2 = copy.deepcopy(temp_population[roulette(rotary_table)])
                cross_pop1, cross_pop2 = crossover(cross_pop1, cross_pop2)
                mutate_pop1 = mutate(cross_pop1)
                mutate_pop2 = mutate(cross_pop2)
                new_population.append(mutate_pop1)
                new_population.append(mutate_pop2)
                new_fitness.append(calculate_fitness(mutate_pop1))
                new_fitness.append(calculate_fitness(mutate_pop2))
            # 种群更新
            temp_population = np.array(copy.deepcopy(new_population))
            temp_fitness = np.array(copy.deepcopy(new_fitness))
        top_index = temp_fitness.argsort()[0:1]
        greedy_indi.append(temp_population[top_index])

    # 统计较优个体中的基因并选择出现频次最高的部分基因
    genes = []
    for pop in greedy_indi:
        genes = genes + list(pop[0])
    data = dict(Counter(genes))
    data = sorted(data.items(), key=lambda x: x[1])[::-1][:int(selected_genes_num / 2)]
    genes = [data[i][0] for i in range(len(data))]
    # genes = list(set(genes + list(history[0])))

    # 在基因池中添加度值最高的节点
    degree = nx.degree_centrality(G)
    degree = sorted(degree.items(), key=lambda x: x[1])[::-1]
    degree = [degree[i][0] for i in range(len(degree))]
    count_i = 0
    while len(genes) < selected_genes_num:
        if degree[count_i] not in genes:
            genes.append(degree[count_i])
        count_i = count_i + 1
    return genes


def genes_cutoff(genes, tag):
    if tag == "no_cutoff_":
        genes = no_cutoff(genes)
    if tag == "random_cutoff_":
        genes = random_cutoff(genes)
    if tag == "greedy_cutoff_":
        genes = greedy_cutoff(genes)
    if tag == "popGreedy_cutoff_":
        genes = pop_greedy_cutoff(genes, 10, 100, k)
    if tag == "popGA_cutoff_":
        genes = local_optimal_cutoff(genes, 10, 100, k)
    return genes

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--method', required=True, type=str)
    args = parser.parse_args()

    # 参数设置
    # dataset = 'ForestFire_n500.txt'
    # cutoff_list = ["no_cutoff_", "random_cutoff_", "greedy_cutoff_", "popGreedy_cutoff_", "popGA_cutoff_"]
    # cutoff_tag = cutoff_list[0]
    dataset = args.dataset
    cutoff_tag = args.method

    G = nx.read_adjlist(dataset, nodetype=int)
    A = np.array(nx.to_numpy_matrix(G, nodelist=sorted(list(G.nodes()))))
    G = nx.from_numpy_matrix(A)
    nodes_num = len(G)
    print(len(G))
    k = int(0.1 * nodes_num)
    # k = 15
    if dataset == 'WattsStrogatz_n500.txt':
        k = int(0.2 * nodes_num)


# *********** 度值最大的k个节点作为关键节点 ***********
    copy_G = copy.deepcopy(G)
    greedy_pop = []
    for i in range(k):
        degree = nx.degree_centrality(copy_G)
        greedy_pop.append(max(degree, key=degree.get))
        copy_G.remove_node(max(degree, key=degree.get))
    print("Greedy: ", calculate_fitness(greedy_pop))

    # ******************** 遗传算法 *********************
    pops_num = 10
    pops_size = 100
    node_mutate_rate = 0.2
    node_crossover_rate = 0.6
    max_generation = 5000
    genes = list(G.nodes())
    genes_num = nodes_num
    selected_genes_num = int(0.4 * nodes_num)
    history = [10000000, 10000000]

    print(len(genes))
    genes = genes_cutoff(genes, cutoff_tag)
    print(len(genes))

    history_list = []
    for global_i in range(5):
        # 种群初始化
        population = np.array([random.sample(genes, k) for j in range(pops_size)])
        fitness = np.array([calculate_fitness(individual) for individual in population])
        top_fitness = [min(fitness)]
        start_time = str(datetime.datetime.now()).replace('-', '').replace(' ', '').replace(':', '')[4:14]
        for generation in range(max_generation):
            print(dataset, cutoff_tag, str(datetime.datetime.now()), ' generation: ', generation)
            rotary_table = calc_rotary_table(fitness)
            # 精英保留
            top_index = fitness.argsort()[0:int(0.1 * pops_size)]
            new_population = list(population[top_index])
            new_fitness = list(fitness[top_index])
            # 交叉变异
            for cross_i in range(int(pops_size * 0.9 / 2)):
                cross_pop1 = copy.deepcopy(population[roulette(rotary_table)])
                cross_pop2 = copy.deepcopy(population[roulette(rotary_table)])
                cross_pop1, cross_pop2 = crossover(cross_pop1, cross_pop2)
                mutate_pop1 = mutate(cross_pop1)
                mutate_pop2 = mutate(cross_pop2)
                new_population.append(mutate_pop1)
                new_population.append(mutate_pop2)
                new_fitness.append(calculate_fitness(mutate_pop1))
                new_fitness.append(calculate_fitness(mutate_pop2))
            # 种群更新
            population = np.array(copy.deepcopy(new_population))
            fitness = np.array(copy.deepcopy(new_fitness))
            top_fitness.append(min(fitness))
            print("min fitness: ", min(fitness))
        # 记录每次global_i的信息
        history_list.append(min(fitness))
        with open(dataset[:-4] + "_" + cutoff_tag + "history_bc.txt", "a+") as f:
            f.write(dataset + ": " + str(history_list) + "\n")
        with open(dataset[:-4] + "_" + cutoff_tag + "history_fitness_bc.txt", "a+") as f:
            f.write(str(top_fitness) + "\n")
    # 记录10次global_i信息统计
    print(np.mean(history_list))
    with open(dataset[:-4] + "_" + cutoff_tag + "history_bc.txt", "a+") as f:
        f.write(dataset + ": mean: " + str(np.mean(history_list)) + "  min: " + str(np.min(history_list)) + "\n")
    with open(dataset[:-4] + "_" + cutoff_tag + "history_fitness_bc.txt", "a+") as f:
        f.write(dataset + "\n" + "\n")

    # ***************************************************
    # from itertools import combinations
    # populations = list(combinations(genes, k))
    # print("last populations number: ", len(populations))
    # best_pop = pop_select(populations)
    # print(calculate_fitness(best_pop))
    # print(best_pop)

