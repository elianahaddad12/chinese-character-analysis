import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
ASSETS_DIR = PROJECT_ROOT / "assets"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import copy
import os
import csv
import random
import subprocess
import math
from collections import deque
import re

import matplotlib.patheffects as path_effects
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import networkx as nx  # page ranker algorithm
import ast  # convert string to tuple
import matplotlib.pyplot as plt  # visualize page ranker
from networkx.drawing.nx_pydot import to_pydot  # graphviz visualization
from pyvis.network import Network
from scipy.stats import kendalltau
import matplotlib

matplotlib.rcParams['font.family'] = 'Noto Sans SC'  # font used in graphs

# how many nodes we want to see in the top k nodes bar graph
num_nodes = 10

# True if you want to evaluate the page rank graph, can be skipped for faster processing
do_evaluate = False

# what radicals the nodes in the graph should contain (filters those that don't)
main_radicals_for_ranker = {
    '小', '龍', '聿', '耒', '弋', '辵', '角', '赤', '田', '毛', '勹', '鹿',
    '斗', '風', '鳥', '青', '生', '土', '麥', '木', '女', '衣', '鹵', '龜',
    '黃', '黽', '示', '犬', '高', '己', '糸', '長', '耳', '而', '虍', '日',
    '鬼', '里', '至', '方', '鬲', '舌', '色', '冫', '匕', '黑', '凵', '虫',
    '辛', '米', '夕', '鼻', '辰', '瓜', '弓', '彳', '入', '甘', '見', '戶',
    '隹', '鼎', '廾', '又', '言', '雨', '金', '目', '尢', '非', '几', '禾',
    '艮', '貝', '冖', '首', '广', '車', '酉', '爻', '矛', '舟', '自', '皮',
    '囗', '无', '牙', '士', '髟', '韋', '屮', '工', '气', '彡', '亠', '走',
    '鬥', '山', '瓦', '齒', '厶', '鼠', '身', '川', '齊', '香', '手', '攴',
    '月', '豸', '肉', '用', '二', '臼', '足', '鬯', '母', '龠', '阜', '十',
    '疒', '頁', '行', '食', '丨', '癶', '匸', '殳', '匚', '文', '止', '火',
    '臣', '冂', '厂', '卩', '八', '氏', '豆', '片', '廴', '力', '欠', '竹',
    '丿', '門', '幺', '牛', '支', '干', '艸', '疋', '缶', '亅', '羊', '寸',
    '舛', '馬', '魚', '刀', '禸', '邑', '丶', '夂', '爪', '立', '黍', '飛',
    '卜', '宀', '心', '父', '石', '革', '韭', '黹', '网', '戈', '口', '儿',
    '羽', '矢', '曰', '巾', '皿', '歹', '爿', '比', '水', '隶', '尸', '白',
    '老', '谷', '一', '襾', '玄', '面', '鼓', '玉', '骨', '子', '麻', '音',
    '斤', '釆', '彐', '穴', '乙', '人', '豕', '血', '夊', '大'}

# traditional to simplified radical forms dictionary
t_t_s = {'糸': '纟', '見': '见', '言': '讠', '貝': '贝', '車': '车', '長': '长',
         '門': '门', '韋': '韦', '頁': '页', '風': '风', '飛': '飞', '食': '饣', '馬': '马',
         '魚': '鱼', '鳥': '鸟', '麥': '麦', '黽': '黾', '鼠': '鼡', '齊': '齐', '齒': '齿',
         '龍': '龙', '龜': '龟', '金': '钅', '鹵': '卤', "戶": "户", "艸": "⺾", "黃": "黄",
         "邑": "阝", '足': '⻊'}

# simplified to traditional radical forms dictionary
s_t_t = {'纟': '糸', '见': '見', '讠': '言', '贝': '貝', '车': '車', '长': '長',
         '门': '門', '韦': '韋', '页': '頁', '风': '風', '飞': '飛', '饣': '食', '马': '馬',
         '鱼': '魚', '鸟': '鳥', '麦': '麥', '黾': '黽', '鼡': '鼠', '齐': '齊', '齿': '齒',
         '龙': '龍', '龟': '龜', '钅': '金', '卤': '鹵', "户": "戶", "⺾": "艸", "黄": "黃",
         "阝": "邑", '⻊': '足', "攵": "攴"}

# traditional Chinese radicals
traditionals = {
    '小', '龍', '聿', '耒', '弋', '辵', '角', '赤', '田', '毛', '勹', '鹿',
    '斗', '風', '鳥', '青', '生', '土', '麥', '木', '女', '衣', '鹵', '龜',
    '黃', '黽', '示', '犬', '高', '己', '糸', '長', '耳', '而', '虍', '日',
    '鬼', '里', '至', '方', '鬲', '舌', '色', '冫', '匕', '黑', '凵', '虫',
    '辛', '米', '夕', '鼻', '辰', '瓜', '弓', '彳', '入', '甘', '見', '戶',
    '隹', '鼎', '廾', '又', '言', '雨', '金', '目', '尢', '非', '几', '禾',
    '艮', '貝', '冖', '首', '广', '車', '酉', '爻', '矛', '舟', '自', '皮',
    '囗', '无', '牙', '士', '髟', '韋', '屮', '工', '气', '彡', '亠', '走',
    '鬥', '山', '瓦', '齒', '厶', '鼠', '身', '川', '齊', '香', '手', '攴',
    '月', '豸', '肉', '用', '二', '臼', '足', '鬯', '母', '龠', '阜', '十',
    '疒', '頁', '行', '食', '丨', '癶', '匸', '殳', '匚', '文', '止', '火',
    '臣', '冂', '厂', '卩', '八', '氏', '豆', '片', '廴', '力', '欠', '竹',
    '丿', '門', '幺', '牛', '支', '干', '艸', '疋', '缶', '亅', '羊', '寸',
    '舛', '馬', '魚', '刀', '禸', '邑', '丶', '夂', '爪', '立', '黍', '飛',
    '卜', '宀', '心', '父', '石', '革', '韭', '黹', '网', '戈', '口', '儿',
    '羽', '矢', '曰', '巾', '皿', '歹', '爿', '比', '水', '隶', '尸', '白',
    '老', '谷', '一', '襾', '玄', '面', '鼓', '玉', '骨', '子', '麻', '音',
    '斤', '釆', '彐', '穴', '乙', '人', '豕', '血', '夊', '大'}

# simplified Chinese radicals
simplifieds = {'韦', '麦', '马', '鼡', '齐', '龙', '风', '见', '页', '黾', '讠',
               '⻊', '饣', '飞', '鱼', '长', '车', '鸟', '龟', '齿', '纟', '门',
               '贝', '钅', '户', '⺾', '黄', '阝','攵'}

# chars that were manually tempered with
mappodoufu = {
    "誩": "言", "畕": "田", "刕": "刀", "兟": ["牛", "儿"], "吕": "口", "䖵": "虫", "聶": "耳",
    "赑": "贝", "幷": "干", "出": "山", "仌": "人", "乂": "丿", "騳": "馬",
    "芔": ["艸", "屮"], "鑫": "金", "豩": "豕", "𣏟": "木", "戔": "戈",
    "歰": ["止", "刀", "丶"], "众": "人", "孨": "子", "畾": "田", "𠈌": "人", "叒": "又",
    "从": "人", "多": "夕", "蟲": "虫", "垚": "土", "惢": "心", "串": "中", "燚": "火",
    "圭": "土", "喆": ["士", "口"], "皕": ["一", "白"], "矗": ["十", "目", "一"], "斦": "斤",
    "朋": "月", "驫": "馬", "轟": "車", "卯": "卩", "廿": "廾", "丱": ["丨"],
    "戋": ["戈", "一"], "兓": ["无", "丿"], "回": "囗", "吅": "口", "昌": "日", "鱻": "魚",
    "䀠": "目", "賏": "貝", "歮": "止", "囍": ["士", "口", "八", "一", "口"], "㸚": "爻",
    "赫": "赤", "㼌": "瓜", "㚘": ["人", "二"], "淼": "水", "㗊": "口", "叕": "又",
    "𢆶": "幺", "砳": "石", "秝": "禾", "𠓜": "入", "森": "木", "卝": "卜", "𠱠": "口",
    "⺀": "丶", "棘": ["木", "冂"], "哥": ["一", "亅", "口"], "双": "又", "骉": "马",
    "曳": "曰", "厸": "厶", "乑": "丿", "𢛳": ["十", "网", "一", "心"], "雔": "隹",
    "棗": ["木", "冂"], "炏": "火", "炎": "火", "弱": ["弓", "冫"], "厽": "厶",
    "芻": ["艸", "勹"], "凹": "凵", "𠂊": "丿", "贔": "貝", "磊": "石", "槑": ["口", "木"],
    "辡": "辛", "劦": "力", "毳": "毛", "林": "木", "掱": "手", "皛": "白", "晶": "日",
    "祘": "示", "北": "匕", "弜": "弓", "岀": "山", "册": ["冂", "一"], "玨": "王",
    "姦": "女", "焱": "火", "品": "口", "㠯": "己", "吉": ["士", "口"],
    "兵": ["斤", "一", "八"], "巳": "己", "簔": ["竹", "衣", "口", "一"], "百": ["一", "白"],
    "瑨": ["玉", "一", "八", "日"], "先": ["牛", "儿"], "𠂇": ["丿", "一"],
    "中": ["口", "丨"], "㦮": ["戈", "一"], "州": ["川", "丶"], "龶": ["土", "一"],
    "𢖻": ["心", "夊"], "隱": ['阝', '爫', '工', '彐', '心'], "冋": ["冂", "口"],
    "共": ["廾", "八"], "夫": ["大", "一"], "𠮦": ["口", "八"]}

# variations of radicals (geographical, handwritten, historical, etc.)
rad_variations = {"乀": "丿", "乁": "丿", "㇠": "乙", "乚": "乙", "乛": "乙", "㇖": "乙",
    "亻": "人", "丷": "八", "𠘨": "几", "刂": "刀", "⺈": "刀", "⺊": "卜",
    "⺍": "小", "⺌": "小", "尣": "尢", "巛": "川", "巜": "川", "彑": "彐",
    "⺕": "彐", "忄": "心", "龵": "手", "扌": "手", "⺙": "攴", "旡": "无",
    "龰": "止", "歺": "歹", "毋": "母", "氺": "水", "氵": "水", "灬": "火",
    "爫": "爪", "丬": "爿", "⺧": "牛", "牜": "牛", "犭": "犬", "王": "玉",
    "玊": "玉", "⺪": "疋", "礻": "示", "⺮": "竹", "糹": "糸", "⺳": "网",
    "罒": "网", "⺶": "羊", "耂": "老", "⺺": "聿", "⺻": "聿", "⺼": "肉",
    "艹": "艸", "衤": "衣", "覀": "襾", "西": "襾", "赱": "走", "辶": "辵",
    "镸": "長", "靑": "青", "𩙿": "食", "⺁": "厂", "⻭": "齒"}

# strokes that are not actually Chinese components
maybe_strokes = {"㇙", "㇛", "㇂", "㇈", "㇗", "㇉", "㇏", "㇒", "㇀", "㇝", "㇜",
                 "㇎", "㇇", "龴", "㇆", "㇅"}


def is_a_radical(component):
    """
    Checks if the component provided is a radical
    :param component: the component being checked
    :return: traditional radical form or None
    """
    if component in rad_variations.keys():
        component = rad_variations[component]
    if component in s_t_t.keys():
        component = s_t_t[component]
    if component in traditionals:
        return component
    return None


def find_decomp(character, comp_map, char_rads, wik_map):
    """
    Finds the immediate components of the character and filters out characters
    that don't have any of the main_radicals_for_ranker radicals.
    :param character: the character we want to break down into components
    :param comp_map: {char: (left, right)} decomposition map
    :param char_rads: {char: [rad1, rad2, ...]]} all of the components radicals
    :param wik_map: {char: (left, right)} decomposition map
    :return: a list of *relevant* components (contain one or more main radical)
    or None if it has no components that include them.
    """
    parts = set()

    # if character in main decomposition map break to components
    if character in comp_map.keys():
        # left components
        decomp = comp_map[character][0]
        if decomp is not None:  # might be None if our main radical isn't in the character provided to function
            for comp in decomp:
                if comp not in {"?", "*", "2", "7", None}:
                    parts.add(comp)
        # right components
        decomp = comp_map[character][1]
        if decomp is not None:
            for comp in decomp:
                if comp not in {"?", "*", "2", "7", None}:
                    parts.add(comp)

    # elif character in manual dictionary break to components
    elif character in mappodoufu:  # exception character
        for comp in mappodoufu[character]:
            parts.add(comp)

    # elif character in radical variations change to standard traditional form
    elif character in rad_variations.values():
        parts.add(rad_variations[character])

    # elif character is only found in the back up decomposition dictionary
    elif character in wik_map.keys():
        # left component
         for comp in wik_map[character][0]:
             if comp not in {"?", "*", "2", "7", None}:
                 parts.add(comp)
        # right component
         for comp in wik_map[character][1]:
             if comp not in {"?", "*", "2", "7", None}:
                 parts.add(comp)

    else:  # radicals and things that aren't in any of our character lists (can't be decomposed)
        return None

    # finalizes components and builds edges to immediate components (radicals are changed to traditional forms)
    final_comps = set()
    for comp in parts:
        # if the component is a radical add its *traditional* form
        radical = is_a_radical(comp) # returns traditional form of character or none
        if radical is not None and radical in main_radicals_for_ranker: # contains one of the main radicals
            final_comps.add(radical)

        # add non-radical components if they contain a main radical
        else:
            # comp is in the main decomposition dictionary
            if comp in char_rads.keys():
                for m_rad in main_radicals_for_ranker:  # contains one of the main radicals
                    if m_rad in char_rads[comp]:
                        final_comps.add(comp)
                        break
            # comp is in the back up decomposition dictionary
            elif comp in wik_map.keys():
                for m_rad in main_radicals_for_ranker:  # contains one of the main radicals
                    if comp_contains_ranker_main_rad(comp, m_rad, wik_map):
                        final_comps.add(comp)
                        break

    return final_comps


def comp_contains_ranker_main_rad(c, m_rad, comp_map):
    """
    Checks (recursively) if the provided component contains the main radical
    :param c: the component whose sub-components we are checking
    :param m_rad: the radical we want to check if exists within c's decompositions
    :param comp_map: {char: (left, right)} decomposition map
    :return: true if the component includes the radical in one of its components,
             false otherwise
    """
    complete_decomp_path = set() # path of edges from the character to its radicals

    #
    get_all_components(c, comp_map, complete_decomp_path)
    duplicate = copy.deepcopy(complete_decomp_path)
    for comp in duplicate:
        if comp in rad_variations.keys():
            complete_decomp_path.add(rad_variations[comp])
    if m_rad in complete_decomp_path:
        return True
    return False


def get_all_components(c, comp_map, visited):
    """
    provides all the sub-components of the provided component
    :param c: the character whose sub-components we are seeking
    :param comp_map: the dictionary used to find immediate components
    :param visited: the sub-components we have already visited
    :return: nothing, visited will have all the sub-components
    """
    visited.add(c)
    if c in mappodoufu.keys():
        for comp in mappodoufu[c]:
            visited.add(comp)
        return
    if c not in comp_map.keys():
        return

    parts = set()
    # left component
    for comp in comp_map[c][0]:
        if comp not in {"?", "*", "2", "7", None}:
            parts.add(comp)
    # right component
    for comp in comp_map[c][1]:
        if comp not in {"?", "*", "2", "7", None}:
            parts.add(comp)

    for part in parts:
        if part not in visited:
            get_all_components(part, comp_map, visited)


def build_decomposition_graph(char_list, comp_map, char_to_rads, wik_map):
    """
    Build a weighted directed graph from characters to their decomposed components.
    :param char_list: list of characters to analyze
    :param comp_map: {char: list} decomposition map
    :return: networkx.DiGraph
    """
    G = nx.DiGraph()

    def add_edges(c, visited):
        # stop recursion if we've seen the character/radical
        if c in visited:
            return
        visited.add(c)

        # check if relevant to our main radicals for ranker
        has_main_rads = False
        for rad in main_radicals_for_ranker:
            if c in char_to_rads.keys():
                if rad in char_to_rads[c]:
                    has_main_rads = True
                    break
            elif c in wik_map.keys():
                if comp_contains_ranker_main_rad(c, rad, wik_map):
                    has_main_rads = True
                    break
        if not has_main_rads:
            return

        # stop recursion if c has no decomposition
        components = find_decomp(c, comp_map, char_to_rads, wik_map) # returns only components relevant to main radical
        if components is None:
            return

        if c in components:  # avoid self-loops
            components.remove(c)

        # continues recursion if c has a relevant decomposition
        for comp in components:
            G.add_edge(c, comp, weight=1/len(components))
            if comp not in traditionals:  # makes radicals an "end sink"
                add_edges(comp, visited)
        return

    visited = set()
    for char in char_list:
        add_edges(char, visited)

    return G


def rank_characters(char_list, comp_map, char_to_rads, wik_map, damping=0.85):
    """
    Run weighted PageRank algorithm over decomposition graph.
    Visualizes the graph with normalized node sizes.
    :param char_list: list of characters to include as nodes
    :param comp_map: {char: (left, right)} immediate de-compositions map
    :param char_to_rads: {char: [rad1, rad2,...]]} character to end radicals map
    :param wik_map: {component: (left, right)} map (backup map)
    :param damping: damping factor for page rank algorithm (default 0.85)
    :return: {node: score} dict according to ranks, also graph evaluation and statistics as files
    """
    # build page rank graph
    G = build_decomposition_graph(char_list, comp_map, char_to_rads, wik_map)

    if do_evaluate:
        # evaluates the graph's node coverage, edge validity, ranking consistency, and baseline comparisons
        eval_results, pr_scores = evaluate(
            G, char_list, comp_map, char_to_rads, wik_map, is_a_radical,
            alphas=[0.8, 0.85, 0.9], k=10
        )

        print("Coverage:", eval_results["coverage"])
        print("Validity:", eval_results["validity"])
        print("Consistency:", eval_results["consistency"])
        print("Baseline overlap:", eval_results["baseline_overlap"])

    # compute page rank
    scores = nx.pagerank(G, alpha=damping, weight="weight")

    # top statistics
    plot_bar_graph_top_k(sorted(scores.items(), key=lambda x: x[1], reverse=True), num_nodes)
    print("saved bar graph in plots folder")

    # visualization
    plot_pagerank_graph(G, scores)

    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))


def plot_bar_graph_top_k(sorted_scores, k):
    """
    Plots both a bar chart and a network graph of the top k PageRank nodes,
    showing node, score, and meaning (if available).

    :param G: full NetworkX graph
    :param sorted_scores: list of tuples [(node, score), ...] sorted descending
    :param k: number of top nodes to visualize
    :param comp_map: dict of characters to their immediate components
    :return: dict of top k nodes with score and meaning
    """
    # --- Load meanings once ---
    radical_meanings = {}
    with open(DATA_DIR / "radicals_after_merge.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            english_text = row.get("English", "")
            if english_text:
                english_text = english_text.replace("；", ";")
                meaning = english_text.split(";", 1)[0].strip()
            else:
                meaning = ""
            radical_meanings[row.get("char", "")] = meaning
            radical_meanings[row.get("Variant", "")] = meaning

    char_meanings = {}
    with open(DATA_DIR / "chinese_characters_data3.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            char_meanings[row.get("char", "")] = row.get("meaning", "")

    # --- Annotate top k nodes ---
    topk = sorted_scores[:k]
    annotated_nodes = []
    for char, score in topk:
        meaning = radical_meanings.get(char) or char_meanings.get(char, "")
        annotated_nodes.append((char, score, meaning.split(";")[0]))
        # print(f"{char} ({meaning}) → {score:.6f}")

    # --- BAR CHART ---
    chars = [c for c, _, _ in annotated_nodes]
    scores = [s for _, s, _ in annotated_nodes]
    meanings = [m for _, _, m in annotated_nodes]

    plt.figure(figsize=(14, 7))
    cmap = plt.cm.Blues
    colors = [cmap(s / max(scores)) for s in scores]  # gradient color

    bars = plt.bar(chars, scores, color=colors, edgecolor='black')

    # Titles and labels
    plt.title("Top Chinese Components", fontname="Arial", fontsize=18,
              weight='bold')
    plt.xlabel("Character", fontname="Arial", fontsize=14, weight='bold')
    plt.ylabel("PageRank Score", fontname="Arial", fontsize=14, weight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for label in plt.gca().get_xticklabels():
        label.set_fontweight('bold')
        label.set_color('black')
        label.set_fontsize(14)

        label.set_path_effects([
            path_effects.Stroke(linewidth=0.5, foreground='black'),
            # border thickness & color
            path_effects.Normal()
        ])

    for label in plt.gca().get_yticklabels():
        label.set_fontname('Arial')
        label.set_fontweight('bold')
        label.set_color('black')

    # Annotate each bar with meaning and score
    for bar, m, s in zip(bars, meanings, scores):
        yval = bar.get_height()
        txt = plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + max(scores) * 0.01,
            f"{m}\n({s:.4f})",
            ha="center",
            va="bottom",
            fontsize=12,
            color="black",
            weight="bold"
        )

        # Add a thick white outline to text
        txt.set_path_effects([
            path_effects.Stroke(linewidth=1, foreground='black'),
            path_effects.Normal()
        ])

    # Use proper font for Chinese
    matplotlib.rcParams['font.sans-serif'] = ['Noto Sans SC']
    matplotlib.rcParams['axes.unicode_minus'] = False

    # Save plot
    plots_dir = ASSETS_DIR / "plots"
    os.makedirs(plots_dir, exist_ok=True)
    bar_path = os.path.join(plots_dir, f"top_{k}_pagerank_bar.png")
    plt.tight_layout()
    plt.savefig(bar_path, dpi=300)
    plt.close()

    # Return annotated top k
    return {c: {"score": s, "meaning": m} for c, s, m in annotated_nodes}


def plot_pagerank_graph(G, scores, output_file=None):
    """
    Plots the PageRank graph with professional styling and saves as an SVG file.
    Node size ∝ PageRank score, color ∝ PageRank intensity.
    :param G: NetworkX PageRank graph
    :param scores: dict {node: score}
    :param output_file: filename for the SVG
    """
    if output_file is None:
        output_file = str(ASSETS_DIR / "page_ranker.svg")

    # Convert to pydot graph
    pydot_graph = to_pydot(G)
    pydot_graph.set_splines("true")
    pydot_graph.set_overlap("false")
    pydot_graph.set_sep("+0.3")

    # Min/max for normalization
    min_score = min(scores.values())
    max_score = max(scores.values())

    # Log-scale size normalization
    def normalize(score, min_size=0.3, max_size=2.0):
        if max_score == min_score:
            return (min_size + max_size) / 2
        log_min = math.log(min_score + 1e-9)
        log_max = math.log(max_score + 1e-9)
        log_score = math.log(score + 1e-9)
        return min_size + (log_score - log_min) / (log_max - log_min) * (max_size - min_size)

    # Professional colormap: Yellow → Orange → Red
    cmap = matplotlib.colormaps["YlOrRd"]
    norm = mcolors.Normalize(vmin=min_score, vmax=max_score)

    def color_scale(score):
        r, g, b, _ = cmap(norm(score))
        return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

    # Node styling
    for node in G.nodes():
        score = scores.get(node, 0)
        size = normalize(score)
        color = color_scale(score)
        p = pydot_graph.get_node(node)[0]
        p.set_label(node)                  # show the Chinese character
        p.set_shape("circle")
        p.set_style("filled,setlinewidth(1.2)")
        p.set_fillcolor(color)
        p.set_color("#444444")             # border color
        p.set_fontsize("10")
        p.set_fontname("Noto Sans SC Bold")
        p.set_width(str(size))
        p.set_height(str(size))

    # Edge styling
    for edge in pydot_graph.get_edges():
        edge.set_color("#99999955")   # semi-transparent gray
        edge.set_penwidth("0.7")
        edge.set_arrowsize("0.4")

    # Save DOT file temporarily
    dot_file = str(DATA_DIR / "simple_ranker.dot")
    pydot_graph.write_raw(dot_file, encoding="utf-8")

    # Run Graphviz (sfdp layout for large graphs)
    try:
        subprocess.run([
            "sfdp",
            "-Tsvg",
            "-Goverlap=prism",
            "-Gsep=+1",
            "-GK=1.5",         # stronger repulsion → clearer layout
            "-Gratio=fill",
            "-Gdpi=150",       # better SVG scaling
            dot_file, "-o", output_file
        ], check=True)
        print(f" Graph saved to {output_file} with {len(G.nodes())} nodes")
    except Exception as e:
        print(f"Note: Graphviz (sfdp) visualization skipped or failed: {e}")


# ------------ graph evaluation ------------ #

# ------------------------------
# Coverage metrics
# ------------------------------
def parse_rsindex(filename=None):
    """
    Returns: dict mapping each character to its main radical.
    """
    if filename is None:
        filename = str(DATA_DIR / "RSIndex.txt")
    char_to_main_radical = {}
    main_radical_char = None
    with open(filename, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Split line into parts
            parts = line.split()
            if '.0' in parts[0]:
                # First unicode after the number.0
                main_radical_unicode = parts[1]
                main_radical_char = chr(int(main_radical_unicode[2:], 16))
                main_radical_char = is_a_radical(main_radical_char)

            # All characters after the first Unicode in that line
            for u in parts[1:]:
                if u.startswith("U+"):
                    char = chr(int(u[2:], 16))
                    char_to_main_radical[char] = main_radical_char

    return char_to_main_radical


def compute_overall_coverage(decomposition_dict, characters):
    """"
    Fraction of characters that have ANY decomposition at all.
    :param decomposition_dict: character to immediate components dictionary
    :param characters: nodes
    :return: fraction calculated
    """
    total = len(characters)
    with_decomp = sum(1 for ch in characters if decomposition_dict.get(ch))
    return with_decomp / total if total > 0 else 0.0


def transitive_main_radical_coverage_fast(G, char_to_main_radical):
    """
    Computes the fraction of characters whose main radical is reachable
    using precomputed reachable components.
    :param G: page rank graph
    :param char_to_main_radical: dictionary from character to the kangxi radical
    :return: fraction calculated

    """
    # setup
    characters = G.nodes()
    connections = G.edges()
    reachable_count = 0
    total_chars = len(characters)

    def decompose_by_edges(char, edges, relevant_edges):
        for u, v in edges:
            if u == char:
                relevant_edges.add(v)
                decompose_by_edges(v, edges, relevant_edges)

    # find all the nodes whose incoming edges stem from our original character
    for char in characters:
        relevant_edges = set()
        if is_a_radical(char) is not None:
            reachable_count += 1
            continue
        decompose_by_edges(char, connections, relevant_edges)
        main_rad = char_to_main_radical.get(char)
        if main_rad is not None and main_rad in relevant_edges:
            reachable_count += 1
        else:
            if main_rad is None:
                total_chars = total_chars - 1
                continue
            # print(char, main_rad, relevant_edges)

    return reachable_count / total_chars


# ------------------------------
# Graph validity (heuristic)
# ------------------------------
def compute_graph_validity(G, placeholders={"?", "*", "2", "7"}):
    """
    Checks that all edges have no trash values
    :param G: page ranker graph
    :param placeholders: trash values
    :return: search results (edges with place holder values)
    """
    total_edges = G.number_of_edges()
    invalid_edges = 0
    for u, v in G.edges():
        if v in placeholders:
            invalid_edges += 1
    return {
        "total_edges": total_edges,
        "invalid_edges": invalid_edges,
        "invalid_rate": invalid_edges / total_edges if total_edges > 0 else 0,
    }


# ------------------------------
# Ranking consistency
# ------------------------------
def jaccard(a, b, k=10):
    """
    computes jaccard algo.
    """
    top_a, top_b = set(a[:k]), set(b[:k])
    return len(top_a & top_b) / len(top_a | top_b) if top_a | top_b else 0


# ------------------------------
# Baselines
# ------------------------------
def compute_frequency_baseline(comp_map):
    """
    Count how often each component appears in decompositions.
    """
    freq = {}
    for ch, comps in comp_map.items():
        for c in comps:
            freq[c] = freq.get(c, 0) + 1
    return freq


def compute_degree_baseline(G):
    """
    Node degree as a naive importance score.
    """
    return dict(G.degree())


def compute_pagerank(G, alpha=0.85):
    """
    Wrapper for nx.pagerank so we keep interface consistent.
    """
    return nx.pagerank(G, alpha=alpha)


# ------------------------------
# General baseline comparison
# ------------------------------
def baseline_comparison(scores_a, scores_b, k=10):
    return compare_rankings(scores_a, scores_b, k=k)


def compare_rankings(scores_a, scores_b, k=10):
    ranks_a = sorted(scores_a, key=scores_a.get, reverse=True)
    ranks_b = sorted(scores_b, key=scores_b.get, reverse=True)

    # Top-k Jaccard
    topk_j = jaccard(ranks_a, ranks_b, k)

    # Kendall tau (for all common nodes)
    common = list(set(ranks_a) & set(ranks_b))
    if len(common) < 2:
        tau, pval = 0.0, 1.0
    else:
        vec_a = [ranks_a.index(x) for x in common]
        vec_b = [ranks_b.index(x) for x in common]
        tau, pval = kendalltau(vec_a, vec_b)

    return {"top{}_jaccard".format(k): topk_j, "kendall_tau": tau, "p_value": pval}


# ------------------------------
# Visualization helper
# ------------------------------
def save_plot(fig, fname, outdir=None):
    """
    Save a Matplotlib figure safely.
    """
    if outdir is None:
        outdir = str(ASSETS_DIR / "plots")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, fname)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_topk_comparison(pr_scores, baselines, k=10, alpha=0.85, outdir = "plots"):
    """
    Visualize top-k nodes for PageRank vs baselines.
    baselines: dict of {name: score_dict}
    """
    pr = pr_scores[alpha]
    pr_sorted = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:k]
    pr_nodes = [n for n, _ in pr_sorted]
    pr_vals = [v for _, v in pr_sorted]

    fig = plt.figure(figsize=(12, 5))  # assign to a variable
    plt.subplot(1, len(baselines) + 1, 1)
    plt.bar(range(k), pr_vals)
    plt.xticks(range(k), pr_nodes, rotation=45, ha="right")
    plt.title(f"PageRank (α={alpha})")

    for i, (name, scores) in enumerate(baselines.items(), start=2):
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        nodes = [n for n, _ in sorted_scores]
        vals = [v for _, v in sorted_scores]

        plt.subplot(1, len(baselines) + 1, i)
        plt.bar(range(k), vals)
        plt.xticks(range(k), nodes, rotation=45, ha="right")
        plt.title(f"{name} baseline")

    plt.tight_layout()
    save_plot(fig, f"topk_comparison_alpha{alpha}.png", outdir)


def plot_ranking_overlap(pr_scores, baselines, alpha=0.85, max_k=50, outdir="plots"):
    """
    Plot Jaccard and Kendall Tau as k grows.
    """
    pr = pr_scores[alpha]
    pr_ranked = sorted(pr, key=pr.get, reverse=True)

    ks = list(range(5, max_k+1, 5))
    fig = plt.figure(figsize=(12, 5))  # assign to a variable

    for name, scores in baselines.items():
        ranked = sorted(scores, key=scores.get, reverse=True)
        jaccards, taus = [], []

        for k in ks:
            # Top-k Jaccard
            top_pr = set(pr_ranked[:k])
            top_base = set(ranked[:k])
            j = len(top_pr & top_base) / len(top_pr | top_base) if (top_pr | top_base) else 0
            jaccards.append(j)

            # Kendall Tau (on common nodes)
            common = list(set(pr_ranked) & set(ranked))
            vec_a = [pr_ranked.index(x) for x in common]
            vec_b = [ranked.index(x) for x in common]
            tau, _ = kendalltau(vec_a, vec_b)
            taus.append(tau if tau is not None else 0)

        # Plot Jaccard
        plt.subplot(1, 2, 1)
        plt.plot(ks, jaccards, marker="o", label=name)

        # Plot Kendall Tau
        plt.subplot(1, 2, 2)
        plt.plot(ks, taus, marker="x", label=name)

    plt.subplot(1, 2, 1)
    plt.title("Top-k Jaccard with PageRank")
    plt.xlabel("k")
    plt.ylabel("Jaccard")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Kendall Tau with PageRank")
    plt.xlabel("k")
    plt.ylabel("Tau")
    plt.legend()

    plt.tight_layout()
    save_plot(fig, f"topk_comparison_alpha{alpha}.png", outdir)


# ------------------------------
# Main evaluation pipeline
# ------------------------------
def evaluate(G, char_list, comp_map, char_to_rads, wik_map, is_a_radical,
             alphas=[0.8,0.85,0.9], k=10, visualize=True):
    """evaluates the page rank graph"""
    results = {}

    # Coverage (split into two metrics)
    results["coverage"] = {
        "overall": compute_overall_coverage(comp_map, char_list),
        "main_radical": transitive_main_radical_coverage_fast(G, parse_rsindex()),
    }

    # Graph validity
    results["validity"] = compute_graph_validity(G)

    # Baselines
    freq_scores = compute_frequency_baseline(comp_map)
    deg_scores = compute_degree_baseline(G)

    # Rankings for different alphas
    pr_scores = {}
    for a in alphas:
        pr_scores[a] = compute_pagerank(G, alpha=a)

    # Pairwise consistency between different alphas
    consistency = {}
    for i in range(len(alphas)-1):
        a1, a2 = alphas[i], alphas[i+1]
        consistency[(a1,a2)] = compare_rankings(pr_scores[a1], pr_scores[a2], k=k)
    results["consistency"] = consistency

    # Baseline vs PageRank (using first alpha as default)
    pr_ref = pr_scores[alphas[0]]
    results["baseline_overlap"] = {
        "frequency": baseline_comparison(freq_scores, pr_ref, k=k),
        "degree": baseline_comparison(deg_scores, pr_ref, k=k),
    }

    # Visualization (optional)
    if visualize:
        baselines = {"frequency": freq_scores, "degree": deg_scores}
        plot_topk_comparison(pr_scores, baselines, k=k, alpha=alphas[0])
        plot_ranking_overlap(pr_scores, baselines, alpha=alphas[0], max_k=50)

    return results, pr_scores


if __name__ == "__main__":
    # --------- page ranker --------- #

    # set up the radicals we will be focusing to traditional
    main_rads = set()
    for rad in main_radicals_for_ranker:
        new_rad = rad
        if new_rad in rad_variations.keys():
            new_rad = rad_variations[new_rad]
        if new_rad in s_t_t.keys():
            new_rad = s_t_t[new_rad]
        main_rads.add(new_rad)
    main_radicals_for_ranker = main_rads

    # set up comp_maps (chinese_character_data3)
    comp_map = dict()
    char_to_rads = dict()
    with open(DATA_DIR / "chinese_characters_data3.csv", newline='',
              encoding='utf-8') as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [name.lstrip('\ufeff') for name in
                             reader.fieldnames]
        for row in reader:
            # save radicals in char_to_rads map ['_', '_', '_']
            all_rads = ast.literal_eval(row['all_radicals'])

            # fix bugs in data
            if row['char'] in mappodoufu.keys():  # '簔', '隱'
                all_rads = mappodoufu[row['char']]
            radicals = set()
            for comp in all_rads:
                if comp == '㇙':  # bug in data, stroke visual not radical
                    continue
                if comp in mappodoufu:  # bug in data 吉 counted as rads in data
                    comp = mappodoufu[comp]
                    for rad in comp:
                        radicals.add(rad)
                else:  # no bugs
                    radicals.add(comp)

            char_to_rads[row['char']] = radicals

            # if it is add it to the comp_map
            if row['components']:
                comp_map[row['char']] = ast.literal_eval(row['components'])
    f.close()

    for char in mappodoufu.keys():
        char_to_rads[char] = mappodoufu[char]

    # set up comp_maps backup (wikimedia_dcomposition.csv)
    wiktionary_map = dict()
    with open(DATA_DIR / "wikimedia_decomposition.csv", newline='',
              encoding='utf-8') as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [name.lstrip('\ufeff') for name in
                         reader.fieldnames]
        for row in reader:
            if row['Component'] not in comp_map:
                wiktionary_map[row['Component']] = (row['LeftComponent'], row['RightComponent'])

    # properly calculates and draws the page rank
    rank_characters(comp_map.keys(), comp_map, char_to_rads, wiktionary_map)

    print("Thank you for using the Chinese Components Page Ranker")
