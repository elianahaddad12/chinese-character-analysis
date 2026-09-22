import csv
import os
import re
import pandas as pd
from collections import defaultdict
import requests
from bs4 import BeautifulSoup
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
ASSETS_DIR = PROJECT_ROOT / "assets"

session = requests.Session()

# traditional to simplified radicals map
t_t_s = {'糸': '纟', '見': '见', '言': '讠', '貝': '贝', '車': '车', '長': '长',
         '門': '门', '韋': '韦', '頁': '页', '風': '风', '飛': '飞', '食': '饣', '馬': '马',
         '魚': '鱼', '鳥': '鸟', '麥': '麦', '黽': '黾', '鼠': '鼡', '齊': '齐', '齒': '齿',
         '龍': '龙', '龜': '龟', '金': '钅', '鹵': '卤', "戶": "户", "艸": "⺾", "黃": "黄"}

# traditional radicals
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

# simplified radicals
simplifieds = {'韦', '麦', '马', '鼡', '齐', '龙', '风', '见', '页', '黾', '讠',
               '⻊', '饣', '飞', '鱼', '长', '车', '鸟', '龟', '齿', '纟', '门',
               '贝', '钅', '户', '⺾', '黄'}

# map of manual corrections to our dataset, named after chinese food (look
# it up)
mappodoufu = {  # chars
    "誩": "言", "畕": "田", "刕": "刀", "兟": ["牛", "儿"], "吕": "口", "䖵": "虫",
    "聶": "耳", "丑": ["口", "一"],
    "赑": "贝", "幷": "干", "出": "山", "仌": "人", "乂": "丿", "騳": "馬",
    "芔": ["艸", "屮"], "鑫": "金", "豩": "豕", "𣏟": "木", "戔": "戈",
    "歰": ["止", "刀", "丶"], "众": "人", "孨": "子", "畾": "田", "𠈌": "人", "叒": "又",
    "从": "人", "多": "夕", "蟲": "虫", "垚": "土", "惢": "心", "串": "中", "燚": "火",
    "圭": "土", "喆": "吉", "皕": ["一", "白"], "矗": ["十", "目", "一"], "斦": "斤",
    "朋": "月", "驫": "馬", "轟": "車", "卯": "卩", "廿": "廾", "丱": ["丨", "㇙"],
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
    "兵": ["斤", "一", "八"], "巳": "己", "簔": ["竹", "衣 ", "口", "一"],
    "百": ["一", "白"],
    "瑨": ["玉", "一", "八", "日"], "先": ["牛", "儿"], "𠂇": ["丿", "一"],
    "中": ["口", "丨"], "㦮": ["戈", "一"], "州": ["川", "丶"], "龶": ["土", "一"],
    # what are thoseeeee:
    # "㇙": "?", "㇛": "?", "㇂": "?", "㇈": "?", "㇗": "?", "㇉": "?", "㇏": "?",
    # "㇒": "?", "㇀": "?", "㇝": "?", "㇜": "?", "㇎": "?", "㇇": "?", "龴": "?",
    # "㇆": "?", "㇅": "?",
    # radical variations
    "乀": "丿", "乁": "丿", "㇠": "乙", "乚": "乙", "乛": "乙", "㇖": "乙",
    "亻": "人", "丷": "八", "𠘨": "几", "刂": "刀", "⺈": "刀", "⺊": "卜",
    "⺍": "小", "⺌": "小", "尣": "尢", "巛": "川", "巜": "川", "彑": "彐",
    "⺕": "彐", "忄": "心", "龵": "手", "扌": "手", "⺙": "攴", "旡": "无",
    "龰": "止", "歺": "歹", "母": "毋", "氺": "水", "氵": "水", "灬": "火",
    "爫": "爪", "丬": "爿", "⺧": "牛", "牜": "牛", "犭": "犬", "王": "玉",
    "玊": "玉", "⺪": "疋", "礻": "示", "⺮": "竹", "糹": "糸", "⺳": "网",
    "罒": "网", "⺶": "羊", "耂": "老", "⺺": "聿", "⺻": "聿", "⺼": "肉",
    "艹": "艸", "衤": "衣", "覀": "襾", "西": "襾", "赱": "走", "辶": "辵",
    "镸": "長", "靑": "青", "𩙿": "食"}

# strokes without significant meaning
maybe_strokes = {"㇙", "㇛", "㇂", "㇈", "㇗", "㇉", "㇏", "㇒", "㇀", "㇝", "㇜",
                 "㇎", "㇇", "龴", "㇆", "㇅"}


def categorize_chars():
    """
    uses the file "Unihan_Variants.txt"
    builds sets for traditional and simplified characters.
    if a character has no variant, it will be categorized as both.
    :return: 2 sets: traditional_chars, simplified_chars
    """
    simplified_chars = set()
    traditional_chars = set()
    candidate_chars = set()

    with open(DATA_DIR / 'Unihan_Variants.txt', 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue

            codepoint, field, value = parts
            targets = value.split(' ')

            if field == 'kSimplifiedVariant':
                traditional_chars.add(codepoint)
                for target in targets:
                    simplified_chars.add(target)
            elif field == 'kTraditionalVariant':
                simplified_chars.add(codepoint)
                for target in targets:
                    traditional_chars.add(target)
            else:
                candidate_chars.add(codepoint)

    for c in candidate_chars:
        if c not in simplified_chars and c not in traditional_chars:
            simplified_chars.add(c)
            traditional_chars.add(c)

    return traditional_chars, simplified_chars


def parse_unihan_file(filepath):
    """
    receive a unihan file, parses it so that we will get only the lines in the
    format "unicode variantType unicode".
    :param filepath: the path to the file to be parsed
    :return: useable parsed data
    """
    data = {}
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            if line.startswith("U+"):
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                codepoint = parts[0]
                field = parts[1]
                value = parts[2]
                char = chr(int(codepoint[2:], 16))
                if char not in data:
                    data[char] = {}
                data[char][field] = value
    return data


# the range
def is_cjk(char):
    return '\u4E00' <= char <= '\u9FFF'


# clean the total_strokes data
def clean_strokes(raw):
    if not raw:
        return ""
    return raw.split(" ")[0].strip("()")


def categorize_helper():
    """
    uses the files "TSCharacters.txt", "STCharacters.txt"
    builds sets for traditional and simplified characters.
    :return: 2 sets: trad_set, simp_set
    """
    trad_set = set()
    simp_set = set()

    with open(DATA_DIR / "TSCharacters.txt", encoding="utf-8") as f1:
        for line in f1:
            line = line.strip()
            if not line:
                continue
            trad, simps = line.split("\t")
            trad_set.add(trad)
            simp_set.add(simps)

    with open(DATA_DIR / "STCharacters.txt", encoding="utf-8") as f2:
        for line in f2:
            line = line.strip()
            if not line:
                continue
            simp, trads = line.split("\t")
            simp_set.add(simp)
            trad_set.add(trads)
    return trad_set, simp_set


def load_components_map():
    """
    uses the file "wikimedia_decomposition.csv"
    creates a decomposition dictionary like so:
    { char: (left component, right component) }
    :return: comp_map
    """
    comp_map = {}
    with open(DATA_DIR / "wikimedia_decomposition.csv", newline='',
              encoding='utf-8') as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [name.lstrip('\ufeff') for name in
                             reader.fieldnames]
        for row in reader:
            char = row['Component']
            left = row['LeftComponent']
            right = row['RightComponent']
            if left != "*" or right != "*":
                comp_map[char] = (left if left != "*" else None,
                                  right if right != "*" else None)
    return comp_map


def get_radicals(char):
    """
    scrapes hanzicraft.com
    for a given chinese character, gets all the radicals it contains
    :param char: a given chinese character
    :return: a list of all the radicals char contains
    """
    url = f"https://hanzicraft.com/character/{char}"
    r = session.get(url, timeout=10)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # search for "Radical :"
    radical_title = soup.find("div", class_="decomptitle", string="Radical :")
    if not radical_title:
        return []

    # the following box contains the radicals
    radical_box = radical_title.find_next_sibling("div", class_="decompbox")
    radicals = [a.text.strip() for a in radical_box.find_all("a")]
    all_rads = [rad for rad in radicals if rad and "glyph" not in rad.lower()]
    for r in all_rads[:]:
        if r in maybe_strokes:
            all_rads.remove(r)
        if r in mappodoufu:
            all_rads.remove(r)
            decs = mappodoufu[r]
            if isinstance(decs, str):  # it's a string
                all_rads.append(decs)
            else:  # it's a list
                all_rads.extend(decs)
    return list(set(all_rads))  # remove duplicates caused by the cleaning


def wiktionary_categorize(char):
    """
    scrape Wiktionary to categorize a given Chinese character into Traditional,
    Simplified, Both, or Unknown (if not found)
    :param char: the character to categorize
    :return: the categorization (str)
    """
    headers = {"User-Agent": "BotForAssignment/1.0 (for educational purposes)"}
    url = f"https://en.wiktionary.org/wiki/{char}#Chinese"
    r = session.get(url, headers=headers)
    if r.status_code != 200:
        print(f"Error {r.status_code} for char {char}")
        return "Error: cannot access page"
    soup = BeautifulSoup(r.text, "html.parser")

    # search for categories
    trad = soup.find("td", class_="Hant")
    simp = soup.find("td", class_="Hans")
    both = None

    # look for a table with "simp. and trad."
    tables = soup.find_all("table", class_="floatright")
    for table in tables:
        header = table.find("th")
        if header and "simp." in header.text and "trad." in header.text:
            both = table.find("td", class_="Hani")
            break

    trad_chars = trad.text.strip() if trad else ""
    simp_chars = simp.text.strip() if simp else ""
    both_chars = both.text.strip() if both else ""

    # classify
    if both_chars:
        return "Both"
    elif trad_chars and simp_chars and trad_chars == simp_chars:
        return "Both"
    elif trad_chars:
        return "Traditional"
    elif simp_chars:
        return "Simplified"
    else:
        return "Unknown"


def build():
    """
    the main function that builds our database
    :return:
    """
    # read files
    meaning_data = parse_unihan_file(DATA_DIR / "Unihan_Readings.txt")
    source_data = parse_unihan_file(DATA_DIR / "Unihan_IRGSources.txt")

    # group characters per radical number
    radical_chars = defaultdict(set)  # {radical_number:[chars]}

    with open(DATA_DIR / "RSIndex.txt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue

            radical_number = parts[0].split(".")[0]
            char_codes = re.findall(r"U\+([0-9A-Fa-f]+)", parts[1])
            chars = [chr(int(cp, 16)) for cp in char_codes]

            for c in chars:
                radical_chars[radical_number].add(c)

    total_vals_lst = []
    rad_count_map = {}

    for rad in sorted(radical_chars.keys(), key=lambda x: int(x)):
        total_vals_lst.append(len(radical_chars[rad]))
        rad_count_map[rad] = len(radical_chars[rad])

    sorted_radicals = sorted(rad_count_map.items(), key=lambda item: item[1],
                             reverse=True)

    # use top 50 radicals only
    top_50_radicals = sorted_radicals[:50]

    output_rows = []
    traditional_chars, simplified_chars = categorize_chars()
    trad_help, simp_help = categorize_helper()
    seen_chars = set()
    components_map = load_components_map()

    for rad, rad_count in top_50_radicals:
        chars = radical_chars[rad]
        for char in chars:
            if char in seen_chars:  # no duplicates
                continue
            seen_chars.add(char)

            total_strokes_raw = source_data.get(char, {}).get("kTotalStrokes",
                                                              "")
            total_strokes = clean_strokes(total_strokes_raw)

            radical = source_data.get(char, {}).get("kRSUnicode", "")
            meaning = meaning_data.get(char, {}).get("kDefinition", "")

            # filter: only keep characters with meanings and in basic CJK block
            if not meaning or not is_cjk(char) \
                    or "Kangxi radical" in meaning or "(Cant.)" in meaning \
                    or "radical number" in meaning or "kwukyel" in meaning:
                continue

            in_trad = f"U+{ord(char):04X}" in traditional_chars
            in_simp = f"U+{ord(char):04X}" in simplified_chars
            in_trad_help = char in trad_help
            in_simp_help = char in simp_help

            category = ""

            if (in_trad and in_simp) or (in_trad_help and in_simp_help):
                category = "Both"
            elif in_trad or in_trad_help:
                category = "Traditional"
            elif in_simp or in_simp_help:
                category = "Simplified"
            else:
                category = "Unknown"  # will be handled separately later

            components = components_map.get(char, "")

            output_rows.append({
                "char": char,
                "total_strokes": total_strokes,
                "radical": radical,
                "meaning": meaning,
                "category": category,
                "components": components
            })

    # write to CVS file
    with open(DATA_DIR / "chinese_characters_data3.csv", "w", encoding="utf-8",
              newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["char", "total_strokes",
                                               "radical", "meaning",
                                               "category", "components"])
        writer.writeheader()
        writer.writerows(output_rows)


def add_all_radicals():
    """
    update our chinese_characters_data3.csv dataset:
    for every character, add the decomposition to radicals.
    uses the cache file decomposition_scraping.csv and the scraping function
    get_radicals(char).
    saves the scraped data to our dataset and to decomposition_scraping.csv.
    :return:
    """
    # --- load cache ---
    cache = {}
    if (DATA_DIR / "decomposition_scraping.csv").exists():
        with open(DATA_DIR / "decomposition_scraping.csv", newline="",
                  encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cache[row["char"]] = row["decomposition"]

    # --- update dataset ---
    df = pd.read_csv(DATA_DIR / "chinese_characters_data3.csv", encoding="utf-8")
    all_radicals_list = []
    timer_count = 0

    for idx, row in df.iterrows():
        char = row["char"]

        if char in cache:  # use cache
            all_radicals = cache[char]
        else:  # scrape
            all_radicals = get_radicals(char)
            cache[char] = all_radicals
            if timer_count % 100 == 0:
                time.sleep(0.4)
            timer_count += 1

        all_radicals_list.append(all_radicals)

    df["all_radicals"] = all_radicals_list  # add column

    # --- save cache ---
    with open(DATA_DIR / "decomposition_scraping.csv", "w", newline="",
              encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["char", "decomposition"])
        if f.tell() == 0:  # write header only for empty file
            writer.writeheader()
        for c, dec in cache.items():
            writer.writerow({"char": c, "decomposition": dec})

    # --- save dataset ---
    df.to_csv(DATA_DIR / "chinese_characters_data3.csv", index=False, encoding="utf-8")


def handle_unknowns():
    """
    update our chinese_characters_data3.csv dataset:
    categorize chars that has been categorized as unknown.
    uses the cache file wiktionary_cache.csv and the scraping function
    wiktionary_categorize(char).
    saves the scraped data to our dataset and to wiktionary_cache.csv.
    :return:
    """
    # --- load cache ---
    cache = {}
    if (DATA_DIR / "wiktionary_cache.csv").exists():
        with open(DATA_DIR / "wiktionary_cache.csv", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cache[row["char"]] = row["category"]

    # --- handle unknowns in our dataset ---
    df = pd.read_csv(DATA_DIR / "chinese_characters_data3.csv", encoding="utf-8")
    timer_count = 0

    for idx, row in df.iterrows():
        char = row["char"]
        category = row["category"]

        if category == "Unknown":
            if char in cache:  # use cache
                new_cat = cache[char]
            else:  # scrape
                new_cat = wiktionary_categorize(char)
                cache[char] = new_cat
                if timer_count % 10 == 0:
                    time.sleep(0.5)
                timer_count += 1

            df.at[idx, "category"] = new_cat

    # --- update cache ---
    with open(DATA_DIR / "wiktionary_cache.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["char", "category"])
        if f.tell() == 0:  # write header only for empty file
            writer.writeheader()
        for c, cat in cache.items():
            writer.writerow({"char": c, "category": cat})

    # --- update our dataset ---
    df.to_csv(DATA_DIR / "chinese_characters_data3.csv", index=False, encoding="utf-8")


def wikimedia_check():
    """
    perform a check on the file "wikimedia_decomposition.csv", and our dataset
    """
    print("\n\n--- running check on wikimedia_decomposition.csv ---")
    rights = {}
    lefts = {}
    with open(DATA_DIR / "wikimedia_decomposition.csv", newline='',
              encoding='utf-8') as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [name.lstrip('\ufeff') for name in
                             reader.fieldnames]
        for row in reader:
            char = row['Component']
            right = row['RightComponent']
            left = row['LeftComponent']
            if ("?" or "2" or "7") in right:
                rights[char] = right
            if ("?" or "2" or "7") in left:
                lefts[char] = left
    our_chars = set()
    with open(DATA_DIR / "chinese_characters_data3.csv", newline='',
              encoding='utf-8') as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [name.lstrip('\ufeff') for name in
                             reader.fieldnames]
        for row in reader:
            our_chars.add(row['char'])

    new_rights = {}
    new_lefts = {}
    for c in rights:
        if c in our_chars: new_rights[c] = rights[c]
    for c in lefts:
        if c in our_chars: new_lefts[c] = lefts[c]
    print("new_rights =", new_rights, "\nnew_rights len =", len(new_rights))
    print("new_rights_vals = ", new_rights.values())
    print("new_lefts =", new_lefts, "\nnew_lefts len =", len(new_lefts))
    print("new_lefts_vals = ", new_lefts.values())


if __name__ == "__main__":
    build()
    add_all_radicals()
    handle_unknowns()
