"""
中医药知识库批量导入脚本 v2 — 最大化数据提取
新增：
  - 中成药完整解析（组成、功用、主治、使用注意）
  - 经典方剂解析（代表方、组成药材、主治、剂量）
  - 中医药方大全解析（组成、功效主治、用法）
  - 民间偏方导入
"""

import re, sqlite3, logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "中医药方大全(免费)-0f4ca2a7f41fb7360b4c2e3f5727a5e9856a27"
DB_PATH = PROJECT_ROOT / "data" / "tcm_knowledge.db"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HERB_CATEGORIES = [
    "解表药", "清热药", "泻下药", "祛风湿药", "化湿药",
    "利水渗湿药", "温里药", "理气药", "消食药", "驱虫药",
    "止血药", "活血化瘀药", "化痰止咳平喘药", "安神药",
    "平肝息风药", "开窍药", "补虚药", "收涩药", "涌吐药",
    "攻毒杀虫止痒药", "拔毒化腐生肌药",
]
SECTION_PATTERN = re.compile(r'^第[一二三四五六七八九十节]+节?\s+\S+')
KNOWN_HERB_NAMES: set = set()


def load_known_herbs(db_path):
    conn = sqlite3.connect(str(db_path))
    names = set(r[0] for r in conn.execute("SELECT name FROM herbs").fetchall())
    conn.close()
    return names


# ═══════════════════════════════════════════════════════
# 1. 中药教材解析（药材）
# ═══════════════════════════════════════════════════════

def parse_herb_text(text: str) -> list[dict]:
    """解析中药教材文本"""
    lines = text.replace('\r\n', '\n').replace('\r', '\n').split('\n')
    lines = [l.replace('\x07', '').strip() for l in lines]
    lines = [l for l in lines if l]

    herbs = []
    current_category = ""
    i = 0
    while i < len(lines):
        line = lines[i]
        if line in HERB_CATEGORIES:
            current_category = line
            i += 1
            continue
        if SECTION_PATTERN.match(line) or line in ('药名','性味','功效','临床应用'):
            i += 1
            continue

        is_known = line in KNOWN_HERB_NAMES
        is_candidate = bool(re.match(r'^[\u4e00-\u9fff]{2,4}$', line)) and line not in HERB_CATEGORIES and line not in ('药名','性味','功效','临床应用','用于')

        if is_known or is_candidate:
            name = line
            i += 1
            props = []
            while i < len(lines):
                nl = lines[i]
                if nl in KNOWN_HERB_NAMES: break
                if re.match(r'^[\u4e00-\u9fff]{2,4}$', nl) and nl not in HERB_CATEGORIES and nl not in ('药名','性味','功效','临床应用','用于') and nl not in KNOWN_HERB_NAMES: break
                if nl in HERB_CATEGORIES or SECTION_PATTERN.match(nl) or nl in ('药名','性味','功效','临床应用'): break
                props.append(nl)
                i += 1
            herb = parse_herb_props(name, props, current_category)
            if herb: herbs.append(herb)
        else:
            i += 1
    return herbs


def parse_herb_props(name, props, category):
    result = {"name": name, "category": category, "nature": "", "flavor": "", "meridian": "", "efficacy": "", "indications": ""}
    eff_lines, ind_lines = [], []
    for p in props:
        if re.match(r'^[\d①②③④⑤]+[、.．]', p) or re.match(r'^用于', p) or re.match(r'^[\d①②③④⑤]+[、.．]\s*用于', p):
            ind_lines.append(p)
        elif '用于' in p and len(ind_lines) > 0:
            ind_lines.append(p)
        else:
            eff_lines.append(p)
    if not eff_lines: return result
    first = eff_lines[0]
    # 归经
    mm = re.search(r'(肺|心|肝|脾|肾|胃|胆|大肠|小肠|膀胱|三焦|心包)', first)
    if mm:
        mraw = first[mm.start():]
        mraw = re.sub(r'[，。、\s经]', '', mraw)
        seen = set()
        result["meridian"] = '、'.join(ch for ch in mraw if ch not in seen and not seen.add(ch))
        nf = first[:mm.start()].strip('，。、\s')
    else:
        nf = first.strip('，。、\s')
    nat = re.findall(r'[寒热温凉平]', nf)
    flv = re.findall(r'[辛甘苦酸咸淡]', nf)
    if nat: result["nature"] = '、'.join(sorted(set(nat)))
    if flv: result["flavor"] = '、'.join(sorted(set(flv)))
    if len(eff_lines) > 1:
        result["efficacy"] = '，'.join(re.sub(r'[\x07]', '', e).strip('，。、\s') for e in eff_lines[1:] if e.strip('，。、\s'))
    else:
        eff = re.sub(r'^[\s、，,。]*[辛甘苦酸咸淡微寒热温凉平大]+[\s、，,。]*', '', first).strip('，。、\s\x07')
        eff = re.sub(r'(肺|心|肝|脾|肾|胃|胆|大肠|小肠|膀胱|三焦|心包)[经]?[\s、，,。]*$', '', eff).strip()
        if eff: result["efficacy"] = eff
    if ind_lines:
        result["indications"] = ' '.join(ind_lines).replace('\x07', '').strip()
    return result


# ═══════════════════════════════════════════════════════
# 2. 经典方剂解析（组成 + 主治）
# ═══════════════════════════════════════════════════════

def parse_classic_formula_text(text: str) -> list[dict]:
    """解析中医经典方剂文本（含组成和主治）"""
    lines = text.replace('\r\n', '\n').replace('\r', '\n').split('\n')
    lines = [l.strip() for l in lines if l.strip()]

    formulas = []
    current = {}
    for l in lines:
        if l.startswith('【代表方】'):
            if current:
                formulas.append(current)
            name = l.replace('【代表方】', '').strip()
            # 去掉出处
            name = re.sub(r'[（(].*?[）)]', '', name).strip()
            current = {"name": name, "composition": "", "functions": "", "indications": "", "usage": ""}
        elif l.startswith('【组成】'):
            current["composition"] = l.replace('【组成】', '').strip()
        elif l.startswith('【主治】'):
            current["indications"] = l.replace('【主治】', '').strip()
        elif l.startswith('【功用】') or l.startswith('【功效主治】'):
            current["functions"] = l.replace('【功用】', '').replace('【功效主治】', '').strip()
        elif l.startswith('【用法】'):
            current["usage"] = l.replace('【用法】', '').strip()
        elif l.startswith('【使用注意】'):
            current.setdefault("functions", "")
        elif l.startswith('常用剂量') and current:
            current["usage"] = (current.get("usage", "") + '；' + l).strip('；')
    if current:
        formulas.append(current)
    return formulas


# ═══════════════════════════════════════════════════════
# 3. 中医药方大全解析
# ═══════════════════════════════════════════════════════

def parse_prescription_text(text: str) -> list[dict]:
    """解析中医药方大全文本"""
    lines = text.replace('\r\n', '\n').replace('\r', '\n').split('\n')
    lines = [l.strip() for l in lines if l.strip() and len(l.strip()) > 1]

    formulas = []
    current = {}
    for l in lines:
        # 跳过纯数字/无序列表行
        if re.match(r'^[\d\s\.\+\-\*]+$', l): continue
        # 方剂名：不含【】且不是组成/用法等标记行且长度适中
        if l.startswith('【组成】') or l.startswith('[组成]'):
            current["composition"] = l.replace('【组成】', '').replace('[组成]', '').strip()
        elif l.startswith('【功效主治】') or l.startswith('[功效主治]') or l.startswith('【功用主治】') or l.startswith('【功用】'):
            current["functions"] = l.replace('【功效主治】', '').replace('[功效主治]', '').replace('【功用主治】', '').replace('【功用】', '').strip()
        elif l.startswith('【主治】') or l.startswith('[主治]'):
            current["indications"] = l.replace('【主治】', '').replace('[主治]', '').strip()
        elif l.startswith('【用法】') or l.startswith('[用法]') or l.startswith('【用法用量】'):
            current["usage"] = l.replace('【用法】', '').replace('[用法]', '').replace('【用法用量】', '').strip()
        elif not l.startswith('【') and not l.startswith('[') and not l.startswith('【使用注意】'):
            if current.get("composition") and not current.get("name"):
                current["name"] = l
            else:
                # 新方剂开始
                if current.get("name"):
                    formulas.append(current)
                current = {"name": l}
    if current and current.get("name"):
        formulas.append(current)
    return formulas


# ═══════════════════════════════════════════════════════
# 4. 中成药完整解析
# ═══════════════════════════════════════════════════════

def parse_patent_medicine_full(text: str) -> list[dict]:
    """解析中成药文本（完整字段）"""
    lines = text.replace('\r\n', '\n').replace('\r', '\n').split('\n')
    lines = [l.strip() for l in lines if l.strip()]

    medicines = []
    current_cat = ""
    current = {}
    for l in lines:
        if re.match(r'^[\d\s\.]+$', l): continue
        # 分类行：如"一、解表类中成药"
        if re.match(r'^[一二三四五六七八九十]+[、.．]', l) and '类' in l:
            current_cat = l
            continue
        if l.startswith('【组成】'):
            current["composition"] = l.replace('【组成】', '').strip()
        elif l.startswith('【功用】') or l.startswith('【功能】'):
            current["functions"] = l.replace('【功用】', '').replace('【功能】', '').strip()
        elif l.startswith('【主治】'):
            current["indications"] = l.replace('【主治】', '').strip()
        elif l.startswith('【使用注意】') or l.startswith('【注意事项】'):
            current["contraindications"] = l.replace('【使用注意】', '').replace('【注意事项】', '').strip()
        elif l.startswith('【用法用量】') or l.startswith('【用法】'):
            current["usage_dosage"] = l.replace('【用法用量】', '').replace('【用法】', '').strip()
        elif not l.startswith('【'):
            # 中成药名行：以丸散膏丹片胶囊颗粒口服液结尾
            if re.search(r'[丸散膏丹片液]$', l) or '胶囊' in l or '口服液' in l or '颗粒' in l:
                if current.get("name"):
                    medicines.append({**current, "category": current_cat})
                current = {"name": l}
    if current.get("name"):
        medicines.append({**current, "category": current_cat})
    return medicines


# ═══════════════════════════════════════════════════════
# 5. 民间偏方解析
# ═══════════════════════════════════════════════════════

def parse_folk_remedy_text(text: str) -> list[dict]:
    """解析民间偏方"""
    lines = text.replace('\r\n', '\n').replace('\r', '\n').split('\n')
    lines = [l.strip() for l in lines if l.strip()]

    remedies = []
    current_disease = ""
    current_remedy = ""
    for l in lines:
        # 跳过说明文字
        if l.startswith('说 明') or l.startswith('1、') and '药' in l: continue
        # 疾病名行：以"："或"："结尾或包含"："
        if '：' in l and not l.startswith(' '):
            parts = l.split('：', 1)
            if len(parts) == 2 and len(parts[0]) <= 20:
                if current_disease and current_remedy:
                    remedies.append({"disease": current_disease, "remedy": current_remedy[:500]})
                current_disease = parts[0]
                current_remedy = parts[1]
                continue
        if current_disease:
            current_remedy += ' ' + l
    if current_disease and current_remedy:
        remedies.append({"disease": current_disease, "remedy": current_remedy[:500]})
    return remedies


# ═══════════════════════════════════════════════════════
# 数据库导入
# ═══════════════════════════════════════════════════════

def import_herbs(conn, herbs):
    cur = conn.cursor()
    existing = set(r[0] for r in cur.execute("SELECT name FROM herbs").fetchall())
    new_cnt, upd_cnt = 0, 0
    for h in herbs:
        if h["name"] in existing:
            cur.execute("""
                UPDATE herbs SET category=COALESCE(NULLIF(category,''),?),
                nature=COALESCE(NULLIF(nature,''),?), flavor=COALESCE(NULLIF(flavor,''),?),
                meridian=COALESCE(NULLIF(meridian,''),?), efficacy=COALESCE(NULLIF(efficacy,''),?),
                indications=COALESCE(NULLIF(indications,''),?)
                WHERE name=?
            """, (h["category"], h["nature"], h["flavor"], h["meridian"],
                  h["efficacy"], h["indications"], h["name"]))
            upd_cnt += 1
        else:
            cur.execute("""INSERT INTO herbs(name,category,nature,flavor,meridian,efficacy,indications)
                VALUES(?,?,?,?,?,?,?)""", (h["name"],h["category"],h["nature"],h["flavor"],
                h["meridian"],h["efficacy"],h["indications"]))
            new_cnt += 1
    conn.commit()
    return new_cnt, upd_cnt


def import_formulas(conn, formulas):
    cur = conn.cursor()
    existing = set(r[0] for r in cur.execute("SELECT name FROM formulas").fetchall())
    new_cnt = 0
    for f in formulas:
        if f["name"] not in existing:
            cur.execute("""INSERT INTO formulas(name,category,composition,functions,indications,usage)
                VALUES(?,?,?,?,?,?)""",
                (f["name"], f.get("category",""), f.get("composition",""),
                 f.get("functions",""), f.get("indications",""), f.get("usage","")))
            new_cnt += 1
    conn.commit()
    return new_cnt


def import_patent_medicines(conn, meds):
    cur = conn.cursor()
    existing = set(r[0] for r in cur.execute("SELECT name FROM patent_medicines").fetchall())
    new_cnt = 0
    upd_cnt = 0
    for m in meds:
        if m["name"] in existing:
            cur.execute("""
                UPDATE patent_medicines SET
                composition=COALESCE(NULLIF(composition,''),?),
                functions=COALESCE(NULLIF(functions,''),?),
                indications=COALESCE(NULLIF(indications,''),?),
                usage_dosage=COALESCE(NULLIF(usage_dosage,''),?),
                contraindications=COALESCE(NULLIF(contraindications,''),?),
                category=COALESCE(NULLIF(category,''),?)
                WHERE name=?
            """, (m.get("composition",""), m.get("functions",""), m.get("indications",""),
                  m.get("usage_dosage",""), m.get("contraindications",""),
                  m.get("category",""), m["name"]))
            upd_cnt += 1
        else:
            cur.execute("""INSERT INTO patent_medicines(name,category,composition,functions,indications,usage_dosage,contraindications)
                VALUES(?,?,?,?,?,?,?)""",
                (m["name"], m.get("category",""), m.get("composition",""), m.get("functions",""),
                 m.get("indications",""), m.get("usage_dosage",""), m.get("contraindications","")))
            new_cnt += 1
    conn.commit()
    return new_cnt, upd_cnt


def import_folk_remedies(conn, remedies):
    cur = conn.cursor()
    existing = set(r[0] for r in cur.execute("SELECT disease FROM folk_remedies").fetchall())
    new_cnt = 0
    for r in remedies:
        if r["disease"] not in existing:
            cur.execute("""INSERT INTO folk_remedies(disease,remedy) VALUES(?,?)""",
                        (r["disease"], r["remedy"]))
            new_cnt += 1
    conn.commit()
    return new_cnt


# ═══════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════

def main():
    global KNOWN_HERB_NAMES
    KNOWN_HERB_NAMES = load_known_herbs(DB_PATH)
    logger.info("已加载 %d 个已知药材名", len(KNOWN_HERB_NAMES))

    conn = sqlite3.connect(str(DB_PATH))

    # ─── 1. 药材（中药教材）───
    for fname in ["_中药大全.txt", "_中药大全-f1de6c8771fe910ef12df8b9.txt"]:
        fpath = DATA_DIR / fname
        if fpath.exists():
            logger.info("="*50)
            logger.info("📦 药材: %s", fname)
            herbs = parse_herb_text(fpath.read_text(encoding='utf-8'))
            logger.info("   解析到 %d 种", len(herbs))
            new, upd = import_herbs(conn, herbs)
            logger.info("   新增 %d / 更新 %d", new, upd)
            break  # 只用第一个

    # ─── 2. 药材（本草纲目药理，同格式）───
    fpath = DATA_DIR / "_中医草药本草纲目药理大全-8c686339e209581b6bd97f19227916888586b953.txt"
    if fpath.exists():
        logger.info("="*50)
        logger.info("📦 药材(本草纲目): 解析中...")
        herbs = parse_herb_text(fpath.read_text(encoding='utf-8'))
        logger.info("   解析到 %d 种", len(herbs))
        new, upd = import_herbs(conn, herbs)
        logger.info("   新增 %d / 更新 %d", new, upd)

    # ─── 3. 经典方剂（含组成+主治）───
    for fname in ["_中医经典方剂-83ebd60be55c3b3567ec102de2bd960590c6d994.txt",
                  "_中医经典方剂-d742e505a56e58fafab069dc5022aaea998f4184.txt"]:
        fpath = DATA_DIR / fname
        if fpath.exists():
            logger.info("="*50)
            logger.info("📦 经典方剂: %s", fname)
            formulas = parse_classic_formula_text(fpath.read_text(encoding='utf-8'))
            logger.info("   解析到 %d 首", len(formulas))
            new = import_formulas(conn, formulas)
            logger.info("   新增 %d", new)

    # ─── 4. 中医药方大全 ───
    fpath = DATA_DIR / "_中医药方大全(免费)-0f4ca2a7f41fb7360b4c2e3f5727a5e9856a278f.txt"
    if fpath.exists():
        logger.info("="*50)
        logger.info("📦 中医药方大全: 解析中...")
        formulas = parse_prescription_text(fpath.read_text(encoding='utf-8'))
        logger.info("   解析到 %d 首", len(formulas))
        new = import_formulas(conn, formulas)
        logger.info("   新增 %d 首方剂", new)

    # ─── 5. 中成药（完整解析）───
    fpath = DATA_DIR / "_常用中成药-e98d4127192e45361066f5b5.txt"
    if fpath.exists():
        logger.info("="*50)
        logger.info("📦 中成药: 解析中...")
        meds = parse_patent_medicine_full(fpath.read_text(encoding='utf-8'))
        logger.info("   解析到 %d 种", len(meds))
        new, upd = import_patent_medicines(conn, meds)
        logger.info("   新增 %d / 更新 %d", new, upd)

    # ─── 6. 民间偏方 ───
    fpath = DATA_DIR / "_中药大全-0dafec8b680203d8ce2f244f.txt"
    if fpath.exists():
        logger.info("="*50)
        logger.info("📦 民间偏方: 解析中...")
        # 只取第一个注释前的部分（非说明文字）
        text = fpath.read_text(encoding='utf-8')
        cutoff = text.find('说 明')
        if cutoff > 0: text = text[:cutoff]
        remedies = parse_folk_remedy_text(text)
        logger.info("   解析到 %d 条", len(remedies))
        new = import_folk_remedies(conn, remedies)
        logger.info("   新增 %d 条", new)

    # ─── 7. 统计 ───
    cur = conn.cursor()
    logger.info("="*60)
    logger.info("📊 导入后统计:")
    for table, desc, name_col in [
        ("herbs","药材","name"),
        ("formulas","方剂","name"),
        ("patent_medicines","中成药","name"),
        ("folk_remedies","验方","disease"),
    ]:
        cnt = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        uniq = cur.execute(f"SELECT COUNT(DISTINCT {name_col}) FROM {table}").fetchone()[0]
        logger.info("   %s: %d 条 (唯一%s: %d)", desc, cnt, "名" if name_col=="name" else "", uniq)

    conn.close()
    logger.info("✅ 全部完成!")

if __name__ == "__main__":
    main()
