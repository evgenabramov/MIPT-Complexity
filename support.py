import numpy as np
import pandas as pd
import scipy.stats as sps
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import permutations
from matplotlib.font_manager import FontProperties
from tqdm.notebook import tqdm
import time


def get_adjacency_matrix(points, metric='euclidean'):
    '''Для заданного набора точек строит матрицу смежности в соответствии с 
    указанной метрикой
    
    :params:
        - points - набор точек
        - metric - используемая метрика
    '''
    
    points = points[np.newaxis, :]
    diff = points - points.reshape((-1, 1, 2))
    if metric == 'euclidean':
        matrix = np.linalg.norm(diff, axis=-1)
    elif metric == 'manhattan':
        matrix = np.sum(np.abs(diff), axis=-1)
    elif metric == 'max':
        matrix = np.max(np.abs(diff), axis=-1)
    else:
        raise Exception('metric not found')
    
    return matrix


def get_odd_degree_vertices(graph):
    '''Находит в графе вершины с нечетными степенями

    :params:
        - graph - граф, наследник класса nx.Graph
    :return value:
        - vertex_indices - индексы вершин
    '''
    vertices, degrees = np.array(graph.degree).T
    return vertices[degrees % 2 == 1]


def find_min_weight_perfect_matching(graph):
    '''Находит в графе максимальное паросочетание минимального веса

    :params:
        - graph - граф, наследник класса nx.Graph
    :return value:
        - matching - ребра паросочетания
    '''
    
    second_graph = nx.Graph()
    for first, second, weight in graph.edges.data('weight'):
        second_graph.add_edge(first, second, weight=-weight)

    return nx.max_weight_matching(second_graph, maxcardinality=True)


def find_hamiltonian_path(points, metric='euclidean'):
    '''Решает 3/2-приближенную метрическую задачу коммивояжера с евклидовой
    метрикой с помощью алгоритма Кристофидеса

    :params:
        - points - координаты вершин в графе
        - metric - используемая метрика
    :return value:
        (vertices, length) - список индексов вершин в гамильтоновом пути
        наименьшего веса и длина этого пути
    '''
    
    adjacency_matrix = get_adjacency_matrix(points, metric)
    graph = nx.from_numpy_array(adjacency_matrix)
    
    minimum_spanning_tree = nx.minimum_spanning_tree(graph)
    
    odd_degree_vertices = get_odd_degree_vertices(minimum_spanning_tree)
    odd_degree_graph = nx.from_numpy_array(
        adjacency_matrix[odd_degree_vertices][:, odd_degree_vertices])
    
    min_weight_perfect_matching = find_min_weight_perfect_matching(
        odd_degree_graph)
    
    eulerian_graph = nx.MultiGraph()
    for first, second, weight in minimum_spanning_tree.edges(data='weight'):
        eulerian_graph.add_edge(first, second, weight=weight)
    
    for first, second in min_weight_perfect_matching:
        first = odd_degree_vertices[first]
        second = odd_degree_vertices[second]
        eulerian_graph.add_edge(first, second, 
                                weight=adjacency_matrix[first][second])
    
    eulerian_cycle = nx.eulerian_circuit(eulerian_graph)
    
    used = np.zeros(shape=(points.shape[0]))
    length = 0
    result = []
    for v_from, v_to in eulerian_cycle:
        if len(result) == 0:
            result.append(v_from)
            used[v_from] = 1
        if used[v_to] == 0:
            length += adjacency_matrix[result[-1]][v_to]
            result.append(v_to)
            used[v_to] = 1
    length += adjacency_matrix[result[-1]][result[0]]
    
    return result, length


def find_min_weight_hamiltonian_path(points, metric='euclidean'):
    '''Находит точное решение задачи коммивояжёра алгоритмом полного перебора
    
    :params:
        - points - точки на плоскости
        - metric - используемая метрика
        
    :return value:
        - (vertices, length) - список индексов вершин в гамильтоновом пути
        наименьшего веса и длина этого пути
    '''
    
    adjacency_matrix = get_adjacency_matrix(points, metric)
    
    min_length = None
    min_cycle = None
    
    for permutation in permutations(range(len(points))):
        length = 0
        
        for i, first in enumerate(permutation):
            second = permutation[(i + 1) % len(permutation)]
            length += adjacency_matrix[first][second]
        
        if min_length is None or length < min_length:
            min_length = length
            min_cycle = permutation
    
    return min_cycle, min_length


def plot_hamiltonian_path(points, path, length, ax, title, font=None,
                          node_size=500, font_size=14, width=6, linewidths=3,
                          measured_time=None):
    '''Строит на графике гамильтонов цикл минимального веса для фиксированного
    набора точек

    :params:
        - points - набор точек на плоскости
        - path - список индексов вершин в гамильтоновом цикле
        - length - длина гамильтонова пути
        - ax - ось
        - title - заголовок
        - font - настройки шрифта, наследник класса FontProperties
        - node_size - размер вершины на графике
        - font_size - размер числа-метки вершины
        - width - ширина ребра
        - linewidths - толщина границы вершины
        - measured_time - время работы алгоритма
    '''

    adjacency_matrix = get_adjacency_matrix(points)

    cycle = nx.Graph()
    for i, first in enumerate(path):
        second = path[(i + 1) % len(path)]
        cycle.add_edge(first, second, weight=adjacency_matrix[first][second])

    with sns.plotting_context('notebook'), sns.axes_style('darkgrid'):
        if measured_time is not None:
            title += ' за время {} сек.\n'.format(round(measured_time, 3))
        ax.set_title(title + ' с ответом {}'.format(length.round(3)),
                     fontsize=28, fontproperties=font)

        nx.draw_networkx(cycle, pos=points, node_color='white', ax=ax,
                         edgecolors='#5E5E5E', linewidths=linewidths,
                         edge_color='#7AB648', font_family='Cambria',
                         node_size=node_size, font_size=font_size, width=width)

        ax.set_xlabel('x', fontsize=18)
        ax.set_ylabel('y', fontsize=18)
        ax.grid(ls=':', b=True)


def compare_solutions(points, metric='euclidean', filename=None):
    '''Рисует на графике 1.5-приближение решения задачи коммивояжера и точное 
    решение

    :params:
        - points - набор точек
        - metric - используемая метрика
        - filename - имя файла для сохранения графика
    '''

    correct_path, correct_length = find_min_weight_hamiltonian_path(
        points, metric)
    path, length = find_hamiltonian_path(points, metric)

    with sns.plotting_context('notebook'), sns.axes_style('whitegrid'):
        fig = plt.figure(figsize=(20.7, 8.9))

        font = FontProperties(family='serif', size=28)

        fig.suptitle('Решения задачи коммивояжера для набора точек\n'
                     '{}'.format(points.tolist()), y=0.97,
                     fontproperties=font)

        ax_left = plt.subplot(1, 2, 1, sharex=plt.gca(), sharey=plt.gca())
        ax_left.axis('square')

        plot_hamiltonian_path(points, path, length, ax_left, '1.5-приближение',
                              font)

        ax_right = plt.subplot(1, 2, 2, sharex=plt.gca(), sharey=plt.gca())
        ax_right.axis('square')

        plot_hamiltonian_path(points, correct_path, correct_length, ax_right,
                              'Точное решение', font)

        fig.tight_layout(rect=[0, 0.01, 1, 1])

        if filename is not None:
            fig.savefig(filename, transparent=True)


'''Генерация и обработка первого тестового набора'''
mean = np.array([3, 3])
cov = np.array([[10, 0],
                [0, 10]])

for i, sample_size in enumerate(tqdm(np.arange(4, 11))):
    sample = sps.multivariate_normal(mean, cov).rvs(size=sample_size).round(1)
    compare_solutions(sample, 'small{}.png'.format(i + 1))

for i, sample_size in enumerate(tqdm(np.arange(9, 11))):
    sample = sps.multivariate_normal(mean, cov).rvs(size=sample_size).round(1)
    compare_solutions(sample, metric='euclidean',
                      filename='eu_metric{}.png'.format(i + 1))
    compare_solutions(sample, metric='manhattan',
                      filename='ma_metric{}.png'.format(i + 1))
    compare_solutions(sample, metric='max',
                      filename='max_metric{}.png'.format(i + 1))


'''Генерация и обработка второго тестового набора'''
mean = np.array([3, 3])
cov = np.array([[100, 0],
                [0, 100]])

num_points = np.arange(4, 12)
time_correct = np.array([])
time_approx = np.array([])

for sample_size in tqdm(num_points):
    sample = sps.multivariate_normal(mean, cov).rvs(size=sample_size).round(1)

    start = time.time()
    path, length = find_hamiltonian_path(sample)
    time_approx = np.append(time_approx, time.time() - start)

    start = time.time()
    path_correct, path_length = find_min_weight_hamiltonian_path(sample)
    time_correct = np.append(time_correct, time.time() - start)

num_points = np.ones_like(time_approx).cumsum() + 3

with sns.plotting_context('notebook'), sns.axes_style('darkgrid'):
    plt.figure(figsize=(13, 8))

    font = FontProperties(family='serif', size=28)

    plt.title('Сравнение времени работы 1.5-приближающего алгоритма \n'
              'с наивной точной реализацией', fontsize=22, fontproperties=font)

    plt.plot(num_points, time_approx, label='алгоритм Кристофидеса')

    plt.plot(num_points, time_correct, label='алгоритм полного перебора')

    plt.ylim((0, 0.01))
    plt.xlabel('Число вершин в графе', fontsize=18, fontproperties=font)
    plt.ylabel('Время работы алгоритма, с.', fontsize=18, fontproperties=font)
    plt.legend(fontsize=16)
    plt.grid(ls=':', b=True)
    plt.savefig('speed2.png')


'''Пример обработки графа из третьего тестового набора'''
cities = pd.read_csv('./data/wi29.tsp', sep=' ', skiprows=8,
                     names=['id', 'x', 'y'], skipfooter=1, engine='python',
                     index_col='id')

points = cities.to_numpy()

start = time.time()
path, length = find_hamiltonian_path(points)
measured_time = time.time() - start

with sns.plotting_context('notebook'), sns.axes_style('whitegrid'):
    fig = plt.figure(figsize=(10, 10))

    font = FontProperties(family='serif', size=28)

    plot_hamiltonian_path(points, path, length, ax=plt.gca(),
                          title='29 городов Западной Сахары', font=font,
                          node_size=300, font_size=12, width=3, linewidths=1,
                          measured_time=measured_time)
    plt.axis('square')

    fig.savefig('29cities.png', transparent=True)