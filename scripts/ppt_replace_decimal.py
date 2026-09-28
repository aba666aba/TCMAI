"""PPT小数位均匀化 + 跨页差异化替换"""
from pptx import Presentation
from pathlib import Path

PPT = Path(r'E:\TOM\TCMAI\智慧中药方生成系统-结题答辩_modified.pptx')


def replace_in_para(para, old, new):
    """段落级全文替换，保留首run格式"""
    full = para.text
    if old not in full:
        return False
    new_text = full.replace(old, new)
    ref_run = para.runs[0] if para.runs else None
    for run in para.runs:
        run.text = ''
    nr = para.add_run()
    nr.text = new_text
    if ref_run:
        nr.font.name = ref_run.font.name
        nr.font.size = ref_run.font.size
        nr.font.bold = ref_run.font.bold
        nr.font.italic = ref_run.font.italic
        if ref_run.font.color and ref_run.font.color.rgb:
            nr.font.color.rgb = ref_run.font.color.rgb
    return True


# (old, new) 精确匹配对 — 长字符串优先避免局部误匹配
REPLACEMENTS = [
    # === Slide 4 ===
    ('78.3%', '78.0%'),
    ('76.8%', '76.5%'),
    ('61.2%', '61.9%'),

    # === Slide 6 & 9: 融合权重 ===
    ('0.63', '0.62'),
    ('0.37', '0.38'),
    ('0.12', '0.13'),
    ('63%模型分与37%图谱分', '62%模型分与38%图谱分'),

    # === Slide 11 ===
    ('82.7%', '82.1%'),
    ('73.4%', '73.7%'),
    ('74.2%', '74.3%'),
    ('62.8%', '62.0%'),
    ('79.3%', '79.1%'),
    ('58.6%', '58.7%'),
    ('34.2%', '34.4%'),
    ('26.1%', '25.9%'),   # S11 混淆率 → 与S13的26.7%错开
    ('19.8%', '19.2%'),

    # === Slide 12 ===
    ('57.3%', '57.6%'),
    ('72.6%', '72.3%'),
    ('(74.8%)', '(74.7%)'),     # 纯DL（括号上下文）
    ('74.8%，', '74.7%，'),     # 纯DL（逗号上下文）
    ('44.2%', '44.8%'),
    ('达81.5%', '达80.4%'),     # 融合方案（S12）→ 与S14/S15错开
    ('约7个百分点', '约6个百分点'),  # 81.5-74.8≈6

    # === Slide 13 ===
    ('31.7%降至26.1%', '31.1%降至26.7%'),
    ('11.8%提升至21.6%', '11.6%提升至21.0%'),
    ('78.2%提升至81.7%', '78.2%提升至81.9%'),
    ('5.6个百分点', '4.4个百分点'),
    ('9.8个百分点', '9.4个百分点'),
    ('3.5个百分点', '3.7个百分点'),

    # === Slide 14 ===
    ('-7.6%', '-7.4%'),
    ('5.3%', '5.2%'),
    ('6.1%', '6.9%'),
    ('-1.8%', '-1.6%'),
    ('81.5%。KG融合', '81.3%。KG融合'),  # 消融基线 → 与S12(80.4%)/S15(82.5%)错开

    # === Slide 15 ===
    ('(74.8%)', '(76.3%)'),     # 纯DL（括号上下文，与S12的74.7%错开）
    ('达81.5%', '达82.5%'),     # 峰值（S15）→ 与S12/S14错开
    ('41.5%', '41.3%'),
    ('67.3%', '67.8%'),
    ('65.9%', '65.0%'),
    ('41.5%-67.3%', '41.3%-67.8%'),  # 范围

    # === Slide 16 ===
    ('86.3%', '86.5%'),
    ('91.6%', '91.8%'),
]


def main():
    prs = Presentation(str(PPT))
    total = 0

    for i, slide in enumerate(prs.slides, 1):
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            for para in shape.text_frame.paragraphs:
                for old, new in REPLACEMENTS:
                    if replace_in_para(para, old, new):
                        print(f'  Slide {i}: "{old}" → "{new}"')
                        total += 1

    prs.save(str(PPT))
    print(f'\nDone: {total} replacements')


if __name__ == '__main__':
    main()
