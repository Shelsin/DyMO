from typing import Any, Callable, Dict, Optional, Union, List, Tuple
import spacy
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
import torch
import tqdm
import numpy as np
from PIL import Image
from torch import nn
import math
import os
import random
from torchvision import transforms
import torch.distributions as dist
from torch.nn import functional as F

start_token = "<|startoftext|>"
end_token = "<|endoftext|>"


def cos_dist(attention_map1, attention_map2):
    if len(attention_map1.shape) > 1:
        attention_map1 = attention_map1.reshape(-1)
    if len(attention_map2.shape) > 1:
        attention_map2 = attention_map2.reshape(-1)
    attention_map1 = attention_map1.to(torch.float32)
    attention_map2 = attention_map2.to(torch.float32)

    epsilon = 1e-8
    attention_map1 = attention_map1 + epsilon
    attention_map2 = attention_map2 + epsilon

    attention_map1 = attention_map1 / attention_map1.sum()
    attention_map2 = attention_map2 / attention_map2.sum()

    p = dist.Categorical(probs=attention_map1)
    q = dist.Categorical(probs=attention_map2)
    cos_dist = 1 - (p.probs * q.probs).sum() / (p.probs.norm() * q.probs.norm())

    return cos_dist



def extract_nouns_and_noun_phrases(nlp, text):
    doc = nlp(text)
    result = []
    current_phrase = []

    for i, token in enumerate(doc):
        # 保留限定词、数词、修饰词和名词
        if token.pos_ in ['DET', 'NUM', 'ADJ', 'NOUN', 'PROPN']:
            current_phrase.append((token.text, token.i + 1))

            # 如果当前 token 是名词，且下一个 token 是限定词、数词或形容词，说明是新短语的开始，结束当前短语
            if token.pos_ == 'NOUN' and i + 1 < len(doc) and doc[i + 1].pos_ in ['DET', 'NUM', 'ADJ']:
                result.append(current_phrase)
                current_phrase = []

        # 忽略连词 'and' 和逗号，不打断短语
        elif token.text == 'and' or token.text == ',':
            continue

    # 确保最后的短语不会遗漏
    if current_phrase:
        result.append(current_phrase)

    return result



def extract_adjectives_and_numbers(nlp, text):
    doc = nlp(text)
    result = []

    # 提取名词短语中的形容词和数字
    for chunk in doc.noun_chunks:
        phrase_with_indices = [(token.text, token.i+1) for token in chunk if token.pos_ in ['ADJ', 'NUM']]
        if phrase_with_indices:
            result.append(phrase_with_indices)

    return result


def extract_nouns(nlp, text):
    doc = nlp(text)
    # 提取名词，包括普通名词和专有名词
    nouns = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']]
    return nouns


# deal with the prompt for positive and negative pairs
# from paper "Object-Conditioned Energy-Based Attention Map Alignment in Text-to-Image Diffusion Models"
# https://arxiv.org/pdf/2404.07389
def extract_attribution_indices(doc):
    # doc = parser(prompt)
    subtrees = []
    modifiers = ["amod", "nmod", "compound", "npadvmod", "advmod", "acomp"]

    for w in doc:
        if w.pos_ not in ["NOUN", "PROPN"] or w.dep_ in modifiers:
            continue
        subtree = []
        stack = []
        for child in w.children:
            if child.dep_ in modifiers:
                subtree.append(child)
                stack.extend(child.children)

        while stack:
            node = stack.pop()
            if node.dep_ in modifiers or node.dep_ == "conj":
                subtree.append(node)
                stack.extend(node.children)
        if subtree:
            subtree.append(w)
            subtrees.append(subtree)
    return subtrees


def extract_attribution_indices_with_verbs(doc):
    '''This function specifically addresses cases where a verb is between
       a noun and its modifier. For instance: "a dog that is red"
       here, the aux is between 'dog' and 'red'. '''

    subtrees = []
    modifiers = ["amod", "nmod", "compound", "npadvmod", "advmod", "acomp",
                 'relcl']
    for w in doc:
        if w.pos_ not in ["NOUN", "PROPN"] or w.dep_ in modifiers:
            continue
        subtree = []
        stack = []
        for child in w.children:
            if child.dep_ in modifiers:
                if child.pos_ not in ['AUX', 'VERB']:
                    subtree.append(child)
                stack.extend(child.children)

        while stack:
            node = stack.pop()
            if node.dep_ in modifiers or node.dep_ == "conj":
                # we don't want to add 'is' or other verbs to the loss, we want their children
                if node.pos_ not in ['AUX', 'VERB']:
                    subtree.append(node)
                stack.extend(node.children)
        if subtree:
            subtree.append(w)
            subtrees.append(subtree)
        return subtrees


def extract_attribution_indices_with_verb_root(doc):
    '''This function specifically addresses cases where a verb is between
       a noun and its modifier. For instance: "a dog that is red"
       here, the aux is between 'dog' and 'red'. '''

    subtrees = []
    modifiers = ["amod", "nmod", "compound", "npadvmod", "advmod", "acomp"]
    for w in doc:
        subtree = []
        stack = []

        # if w is a verb/aux and has a noun child and a modifier child, add them to the stack
        if w.pos_ != 'AUX' or w.dep_ in modifiers:
            continue

        for child in w.children:
            if child.dep_ in modifiers or child.pos_ in ['NOUN', 'PROPN']:
                if child.pos_ not in ['AUX', 'VERB']:
                    subtree.append(child)
                stack.extend(child.children)
        # did not find a pair of noun and modifier
        if len(subtree) < 2:
            continue

        while stack:
            node = stack.pop()
            if node.dep_ in modifiers or node.dep_ == "conj":
                # we don't want to add 'is' or other verbs to the loss, we want their children
                if node.pos_ not in ['AUX']:
                    subtree.append(node)
                stack.extend(node.children)

        if subtree:
            if w.pos_ not in ['AUX']:
                subtree.append(w)
            subtrees.append(subtree)
    return subtrees


def extract_noun_indices(doc):
    noun_indices = []
    # nouns_indices = []
    for k, token in enumerate(doc):
        if token.pos_ == "NOUN" or token.pos_ == "PROPN":
            noun_indices.append(token)
            # nouns_indices.append([k])
    return noun_indices


def get_indices(tokenizer, prompt: str) -> Dict[str, int]:
    """Utility function to list the indices of the tokens you wish to alter"""
    ids = tokenizer(prompt).input_ids
    indices = {
        i: tok
        for tok, i in zip(
            tokenizer.convert_ids_to_tokens(ids), range(len(ids))
        )
    }
    return indices


def get_aligned_indices(tokenizer, prompt: str) -> Dict[str, int]:
    """Utility function to list the indices of the tokens you wish to alter"""
    ids = tokenizer(prompt).input_ids
    indices = {
        i: tok
        for tok, i in zip(
            tokenizer.convert_ids_to_tokens(ids), range(len(ids))
        )
    }
    for key, value in indices.items():
        value = value.replace("</w>", "")
        indices[key] = value
    return indices


def align_wordpieces_indices(
        wordpieces2indices, start_idx, target_word
):
    """
    Aligns a `target_word` that contains more than one wordpiece (the first wordpiece is `start_idx`)
    """

    wp_indices = [start_idx]
    wp = wordpieces2indices[start_idx].replace("</w>", "")

    # Run over the next wordpieces in the sequence (which is why we use +1)
    for wp_idx in range(start_idx + 1, len(wordpieces2indices)):
        if wp == target_word:
            break

        wp2 = wordpieces2indices[wp_idx].replace("</w>", "")
        if target_word.startswith(wp + wp2) and wp2 != target_word:
            wp += wordpieces2indices[wp_idx].replace("</w>", "")
            wp_indices.append(wp_idx)
        else:
            wp_indices = (
                []
            )  # if there's no match, you want to clear the list and finish
            break

    return wp_indices

def align_indices(pipeline, prompt, spacy_pairs):
    wordpieces2indices = get_indices(pipeline.tokenizer, prompt)
    paired_indices = []
    collected_spacy_indices = (
        set()
    )  # helps track recurring nouns across different relations (i.e., cases where there is more than one instance of the same word)
    for pair in spacy_pairs:
        curr_collected_wp_indices = (
            []
        )  # helps track which nouns and amods were added to the current pair (this is useful in sentences with repeating amod on the same relation (e.g., "a red red red bear"))
        for member in pair:
            for idx, wp in wordpieces2indices.items():
                if wp in [start_token, end_token]:
                    continue
                wp = wp.replace("</w>", "")
                if member.text == wp:
                    if idx not in curr_collected_wp_indices and idx not in collected_spacy_indices:
                        curr_collected_wp_indices.append(idx)
                        break
                # take care of wordpieces that are split up
                elif member.text.startswith(wp) and wp != member.text:  # can maybe be while loop
                    wp_indices = align_wordpieces_indices(
                        wordpieces2indices, idx, member.text
                    )
                    # check if all wp_indices are not already in collected_spacy_indices
                    if wp_indices and (wp_indices not in curr_collected_wp_indices) and all(
                            [wp_idx not in collected_spacy_indices for wp_idx in wp_indices]):
                        curr_collected_wp_indices.append(wp_indices)
                        break
        for collected_idx in curr_collected_wp_indices:
            if isinstance(collected_idx, list):
                for idx in collected_idx:
                    collected_spacy_indices.add(idx)
            else:
                collected_spacy_indices.add(collected_idx)
        paired_indices.append(curr_collected_wp_indices)
    return paired_indices



def flatten_one_level(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(item)
        else:
            result.append(item)
    return result


def flatten_lists(pairs):
    new_sublist = []
    for item in pairs:
        new_sublist.append(flatten_one_level(item))
    return new_sublist


def find_tokens_from_text(json_data, doc):
    tokens_output = {}
    matched_words = set()  # To track already matched words

    for item in json_data['knowledge']:
        for key, values in item.items():
            key = key.lower()
            temp_key = None
            # Find the key token
            for token in doc:
                if key == token.text and token.i not in matched_words:
                    temp_key = token
                    matched_words.add((token.text, token.i))
                    break

            temp_values = []
            for word in values:
                word = word.lower()
                found = False
                for i, token in enumerate(doc):
                    # Ensure the word matches and hasn't been matched yet
                    if word in token.text and (word, token.i) not in matched_words:
                        temp_values.append(token)
                        matched_words.add((token.text, token.i))  # Mark the word as matched
                        found = True
                        break
                if not found:
                    temp_values = []

            if temp_key is not None:
                tokens_output[temp_key] = temp_values  # Store the key and its values
            else:
                tokens_output[key] = temp_values  # Default to the original key if not found

    return tokens_output