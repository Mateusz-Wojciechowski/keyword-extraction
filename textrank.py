from functionalities import preprocess_text
import networkx as nx


def build_graph(text, window_size):
    phrases = preprocess_text(text)
    graph = nx.Graph()

    for i in range(len(phrases)):
        phrase = phrases[i]
        if phrase not in graph:
            graph.add_node(phrase)

        for j in range(i + 1, i + window_size + 1):
            if j < len(phrases):
                potential_neighbour = phrases[j]
                if graph.has_edge(phrase, potential_neighbour):
                    graph[phrase][potential_neighbour]['weight'] += 1

                else:
                    graph.add_edge(phrase, potential_neighbour, weight=1)

    return graph


def perform_page_rank(graph, d, num_iterations):
    N = len(graph.nodes())

    ranks = {node: 1/N for node in graph.nodes()}
    new_ranks = {}

    for n in range(num_iterations):
        for node in graph.nodes():
            rank_sum = 0
            for neighbour in graph.neighbors(node):
                if graph[node][neighbour]:
                    rank_sum += (ranks[neighbour] * graph[node][neighbour]['weight'])/sum(
                        graph[neighbour][neighbour_]['weight'] for neighbour_ in graph.neighbors(neighbour) if graph[neighbour][neighbour_]['weight'])

            new_ranks[node] = (1 - d) / N + d * rank_sum

    return new_ranks


def get_top_n_keywords(sorted_ranks, num_keywords):
    keyword_list = []
    for i in range(num_keywords):
        keyword, _ = sorted_ranks[i]
        keyword_list.append(keyword)

    return keyword_list


def textrank_extraction(text, num_keywords, window_size, d, num_iterations):
    graph = build_graph(text, window_size)
    ranks = perform_page_rank(graph, d, num_iterations)
    sorted_ranks = sorted(ranks.items(), key=lambda x: x[1], reverse=True)
    print(sorted_ranks)
    keyword_list = get_top_n_keywords(sorted_ranks, num_keywords)
    return keyword_list


if __name__ == '__main__':
    print(textrank_extraction(text, 10, 3, 0.85, 50))