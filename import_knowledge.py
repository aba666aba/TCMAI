"""
中医药知识库批量导入脚本
导入来源：
  1. 中药大全.txt — 《中药学》教材格式（药名、性味归经、功效、主治）
  2. 中医常用经典方剂.docx — 150首经典方剂分类列表
  3. 常用中成药.txt — 中成药数据
"""

import re, sqlite3, logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "中医药方大全(免费)-0f4ca2a7f41fb7360b4c2e3f5727a5e9856a27"
DB_PATH = PROJECT_ROOT / "data" / "tcm_knowledge.db"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── 中药学教材章节分类（21类）──
HERB_CATEGORIES = [
    "解表药", "清热药", "泻下药", "祛风湿药", "化湿药",
    "利水渗湿药", "温里药", "理气药", "消食药", "驱虫药",
    "止血药", "活血化瘀药", "化痰止咳平喘药", "安神药",
    "平肝息风药", "开窍药", "补虚药", "收涩药", "涌吐药",
    "攻毒杀虫止痒药", "拔毒化腐生肌药",
]

SECTION_PATTERN = re.compile(r'^第[一二三四五六七八九十节]+节?\s+\S+')
# 已知的药材名（加载时从数据库读取）
KNOWN_HERB_NAMES: set = set()


# ════════════════════════════════════════════════
# 1. 解析中药教材文本（块结构解析法）
# ════════════════════════════════════════════════

def is_efficacy_line(line: str) -> bool:
    """判断是否为功效描述行（如'利水消肿'、'发汗解表，宣肺平喘'）"""
    if not line:
        return False
    if re.search(r'[用于①②③④⑤\d、]', line):
        return False
    if re.search(r'[辛甘苦酸咸]', line) and re.search(r'[寒热温凉平]', line):
        return False
    if re.match(r'^(?:肺|心|肝|脾|肾|胃|胆|大肠|小肠|膀胱|三焦|心包)+$', line):
        return False
    return True


def load_known_herbs(db_path):
    """从数据库加载已知药材名"""
    conn = sqlite3.connect(str(db_path))
    names = set(r[0] for r in conn.execute("SELECT name FROM herbs").fetchall())
    conn.close()
    return names


def parse_herb_text(text: str) -> list[dict]:
    """解析中药教材文本，返回药材列表"""
    lines = text.replace('\r\n', '\n').replace('\r', '\n').split('\n')
    lines = [l.replace('\x07', '').strip() for l in lines]
    lines = [l for l in lines if l]

    herbs = []
    current_category = ""
    current_section = ""
    i = 0

    while i < len(lines):
        line = lines[i]

        if line in HERB_CATEGORIES:
            current_category = line
            current_section = ""
            i += 1
            continue

        if SECTION_PATTERN.match(line):
            current_section = line
            i += 1
            continue

        if line in ('药名', '性味', '功效', '临床应用'):
            i += 1
            continue

        # 检测药材名：已知药材名 或 (2-4字纯中文，不在排除列表)
        is_known = line in KNOWN_HERB_NAMES
        is_candidate = bool(re.match(r'^[\u4e00-\u9fff]{2,4}$', line)) \
            and line not in HERB_CATEGORIES \
            and line not in ('药名','性味','功效','临床应用') \
            and line not in ('用于',)

        if is_known or is_candidate:
            name = line
            i += 1

            # 收集该药材的后续属性行
            props = []
            while i < len(lines):
                nl = lines[i]
                # 遇到下一个药材名则结束
                if nl in KNOWN_HERB_NAMES:
                    break
                if re.match(r'^[\u4e00-\u9fff]{2,4}$', nl) \
                    and nl not in HERB_CATEGORIES \
                    and nl not in ('药名','性味','功效','临床应用','用于') \
                    and nl not in KNOWN_HERB_NAMES:
                    break
                if nl in HERB_CATEGORIES or SECTION_PATTERN.match(nl) \
                    or nl in ('药名','性味','功效','临床应用'):
                    break
                props.append(nl)
                i += 1

            herb = parse_herb_properties(name, props, current_category, current_section)
            if herb:
                herbs.append(herb)
        else:
            i += 1

    return herbs


def parse_herb_properties(name: str, props: list[str], category: str, section: str) -> dict:
    """解析药材属性：性味归经、功效、主治（块解析）"""
    result = {
        "name": name,
        "category": category,
        "nature": "",
        "flavor": "",
        "meridian": "",
        "efficacy": "",
        "indications": "",
    }

    # 分离功效行和主治行
    efficacy_lines = []
    indication_lines = []

    for p in props:
        if re.match(r'^[\d①②③④⑤]+[、.．]\s*用于', p) \
            or re.match(r'^用于', p) \
            or re.match(r'^[\d①②③④⑤]+[、.．]', p):
            indication_lines.append(p)
        elif '用于' in p and len(indication_lines) > 0:
            indication_lines.append(p)
        else:
            efficacy_lines.append(p)

    if not efficacy_lines:
        return result

    first = efficacy_lines[0]

    # --- 解析性味归经（从第一行提取）---
    # 性味：辛、甘、苦、酸、咸、淡、微寒、温...等
    # 归经：肺、心、肝、脾、肾、胃、胆...等脏器名

    # 先从整体中提取归经（脏器名序列，通常在行尾）
    meridian_match = re.search(r'(肺|心|肝|脾|肾|胃|胆|大肠|小肠|膀胱|三焦|心包)', first)
    if meridian_match:
        meridian_start = meridian_match.start()
        meridian_raw = first[meridian_start:]
        meridian_raw = re.sub(r'[，。、\s经]', '', meridian_raw)
        # 按序去重
        seen = set()
        ordered = []
        for ch in meridian_raw:
            if ch not in seen:
                seen.add(ch)
                ordered.append(ch)
        result["meridian"] = '、'.join(ordered)
        nf_part = first[:meridian_start].strip('，。、\s')
    else:
        nf_part = first.strip('，。、\s')

    # 从 nf_part 提取性味
    natures = re.findall(r'[寒热温凉平]', nf_part)
    flavors = re.findall(r'[辛甘苦酸咸淡]', nf_part)
    if natures:
        result["nature"] = '、'.join(sorted(set(natures)))
    if flavors:
        result["flavor"] = '、'.join(sorted(set(flavors)))

    # --- 解析功效 ---
    if len(efficacy_lines) > 1:
        # 从第二行开始都是功效
        parts = []
        for el in efficacy_lines[1:]:
            el_clean = re.sub(r'[\x07]', '', el).strip('，。、\s')
            if el_clean:
                parts.append(el_clean)
        result["efficacy"] = '，'.join(parts)
    else:
        # 只有一行，去掉性味归经部分后剩下的
        eff = re.sub(r'^[\s、，,。]*[辛甘苦酸咸淡微寒热温凉平大]+[\s、，,。]*', '', first).strip('，。、\s\x07')
        # 去掉尾部归经
        eff = re.sub(r'(肺|心|肝|脾|肾|胃|胆|大肠|小肠|膀胱|三焦|心包)[经]?[\s、，,。]*$', '', eff).strip()
        if eff:
            result["efficacy"] = eff

    # --- 解析主治 ---
    if indication_lines:
        combined = ' '.join(indication_lines)
        combined = combined.replace('\x07', '').strip()
        # 美化：在序号前加换行
        combined = re.sub(r'(\d+[、.．])', r'\n\1', combined).strip()
        result["indications"] = combined

    return result


# ════════════════════════════════════════════════
# 2. 解析经典方剂
# ════════════════════════════════════════════════

def parse_formula_docx(filepath: Path) -> list[dict]:
    """解析中医常用经典方剂.docx"""
    import docx
    doc = docx.Document(str(filepath))
    lines = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

    formulas = []
    for l in lines[1:]:
        m = re.match(r'\d+[、.]\s*(.+?)-(.+)\s+(.+)$', l)
        if m:
            formulas.append({
                "name": m.group(3).strip(),
                "category": m.group(1).strip(),
                "subcategory": m.group(2).strip(),
            })
    return formulas


# ════════════════════════════════════════════════
# 3. 解析中成药文本
# ════════════════════════════════════════════════

def parse_patent_medicine_text(text: str) -> list[dict]:
    """解析常用中成药数据"""
    lines = text.replace('\r\n', '\n').replace('\r', '\n').split('\n')
    lines = [l.strip() for l in lines if l.strip()]

    medicines = []
    current_cat = ""
    for l in lines:
        if re.match(r'^[\d\s\.]+$', l):
            continue
        if len(l) < 4:
            continue
        if l.endswith('药') and len(l) <= 10:
            current_cat = l
            continue
        if re.search(r'[丸散膏丹片液]$', l) or '胶囊' in l or '口服液' in l or '颗粒' in l:
            medicines.append({"name": l, "category": current_cat})
    return medicines


# ════════════════════════════════════════════════
# 4. 导入到数据库
# ════════════════════════════════════════════════

def import_herbs(conn, herbs: list[dict]):
    """批量导入药材"""
    cur = conn.cursor()
    existing = set(r[0] for r in cur.execute("SELECT name FROM herbs").fetchall())

    new_count = 0
    update_count = 0
    for h in herbs:
        if h["name"] in existing:
            cur.execute("""
                UPDATE herbs SET
                    category = COALESCE(NULLIF(category, ''), ?),
                    nature = COALESCE(NULLIF(nature, ''), ?),
                    flavor = COALESCE(NULLIF(flavor, ''), ?),
                    meridian = COALESCE(NULLIF(meridian, ''), ?),
                    efficacy = COALESCE(NULLIF(efficacy, ''), ?),
                    indications = COALESCE(NULLIF(indications, ''), ?)
                WHERE name = ?
            """, (h["category"], h["nature"], h["flavor"], h["meridian"],
                  h["efficacy"], h["indications"], h["name"]))
            update_count += 1
        else:
            cur.execute("""
                INSERT INTO herbs (name, category, nature, flavor, meridian, efficacy, indications)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (h["name"], h["category"], h["nature"], h["flavor"],
                  h["meridian"], h["efficacy"], h["indications"]))
            new_count += 1

    conn.commit()
    return new_count, update_count


def import_formulas(conn, formulas: list[dict]):
    """批量导入方剂"""
    cur = conn.cursor()
    existing = set(r[0] for r in cur.execute("SELECT name FROM formulas").fetchall())

    new_count = 0
    for f in formulas:
        if f["name"] not in existing:
            cur.execute("""
                INSERT INTO formulas (name, category, functions)
                VALUES (?, ?, ?)
            """, (f["name"], f"{f['category']}-{f['subcategory']}",
                  f"{f['category']}类方剂，{f['subcategory']}"))
            new_count += 1

    conn.commit()
    return new_count


def import_patent_medicines(conn, medicines: list[dict]):
    """批量导入中成药"""
    cur = conn.cursor()
    existing = set(r[0] for r in cur.execute("SELECT name FROM patent_medicines").fetchall())

    new_count = 0
    for m in medicines:
        if m["name"] not in existing:
            cur.execute("""INSERT INTO patent_medicines (name, category) VALUES (?, ?)""",
                        (m["name"], m["category"]))
            new_count += 1

    conn.commit()
    return new_count


# ════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════

def main():
    global KNOWN_HERB_NAMES
    KNOWN_HERB_NAMES = load_known_herbs(DB_PATH)
    logger.info("已加载 %d 个已知药材名", len(KNOWN_HERB_NAMES))

    conn = sqlite3.connect(str(DB_PATH))

    # 1. 导入药材
    herb_txt = DATA_DIR / "_中药大全.txt"
    if herb_txt.exists():
        logger.info("=" * 50)
        logger.info("导入药材数据...")
        text = herb_txt.read_text(encoding='utf-8')
        herbs = parse_herb_text(text)
        logger.info("  解析到 %d 种药材", len(herbs))
        new, upd = import_herbs(conn, herbs)
        logger.info("  新增: %d, 更新: %d", new, upd)

    # 2. 导入经典方剂
    formula_docx = DATA_DIR / "中医常用经典方剂-cd2bb7b46b0203d8ce2f0066f5335a8103d26658.docx"
    if formula_docx.exists():
        logger.info("=" * 50)
        logger.info("导入方剂数据...")
        formulas = parse_formula_docx(formula_docx)
        logger.info("  解析到 %d 首方剂", len(formulas))
        new = import_formulas(conn, formulas)
        logger.info("  新增: %d", new)

    # 3. 导入中成药
    patent_txt = DATA_DIR / "_常用中成药.txt"
    if patent_txt.exists():
        logger.info("=" * 50)
        logger.info("导入中成药数据...")
        text = patent_txt.read_text(encoding='utf-8')
        meds = parse_patent_medicine_text(text)
        logger.info("  解析到 %d 种中成药", len(meds))
        new = import_patent_medicines(conn, meds)
        logger.info("  新增: %d", new)

    # 4. 统计
    cur = conn.cursor()
    logger.info("=" * 50)
    logger.info("导入后统计:")
    for table, desc in [("herbs", "药材"), ("formulas", "方剂"),
                         ("patent_medicines", "中成药"), ("folk_remedies", "验方")]:
        cnt = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        if table == "folk_remedies":
            uniq = cur.execute(f"SELECT COUNT(DISTINCT disease) FROM {table}").fetchone()[0]
        else:
            uniq = cur.execute(f"SELECT COUNT(DISTINCT name) FROM {table}").fetchone()[0]
        logger.info("  %s: %d 条 (唯一名: %d)", desc, cnt, uniq)

    conn.close()
    logger.info("全部完成!")


if __name__ == "__main__":
    main()
