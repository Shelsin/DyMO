import spacy
from itertools import combinations

def process_p_n_groups(data):
    positive_pairs = []
    negative_pairs = []

    for group in data:
        first_list, second_list = group
        for a in first_list:
            for b in second_list:
                positive_pairs.append((a, b))
    first_elements = [group[0] for group in data]
    for list1, list2 in combinations(first_elements, 2):
        for a in list1:
            for b in list2:
                negative_pairs.append((a, b))

    return positive_pairs, negative_pairs
