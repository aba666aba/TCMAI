-- 中医药知识库数据库建库脚本
-- 创建日期：2026 年 3 月
-- 使用方法：sqlite3 tcm_knowledge.db < create_tcm_db.sql
-- 数据库说明：包含 4 个核心表，共计 1,641 条中医药知识数据
--   - herbs: 1,010 条药材数据
--   - formulas: 347 条方剂数据
--   - patent_medicines: 63 条中成药数据
--   - folk_remedies: 221 条民间验方数据

-- 创建药材表 (1,010 条)
CREATE TABLE IF NOT EXISTS herbs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    category TEXT,
    nature TEXT,
    flavor TEXT,
    meridian TEXT,
    efficacy TEXT,
    indications TEXT,
    usage TEXT,
    contraindications TEXT,
    source TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建方剂表 (347 条)
CREATE TABLE IF NOT EXISTS formulas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    category TEXT,
    source_book TEXT,
    composition TEXT,
    dosage TEXT,
    functions TEXT,
    indications TEXT,
    usage TEXT,
    modification TEXT,
    explanation TEXT,
    source TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建中成药表 (63 条)
CREATE TABLE IF NOT EXISTS patent_medicines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    category TEXT,
    composition TEXT,
    functions TEXT,
    indications TEXT,
    usage_dosage TEXT,
    contraindications TEXT,
    source TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建民间验方表 (221 条)
CREATE TABLE IF NOT EXISTS folk_remedies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    disease TEXT,
    remedy TEXT,
    ingredients TEXT,
    method TEXT,
    source TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_herbs_name ON herbs(name);
CREATE INDEX IF NOT EXISTS idx_herbs_category ON herbs(category);
CREATE INDEX IF NOT EXISTS idx_herbs_efficacy ON herbs(efficacy);
CREATE INDEX IF NOT EXISTS idx_formulas_name ON formulas(name);
CREATE INDEX IF NOT EXISTS idx_formulas_category ON formulas(category);
CREATE INDEX IF NOT EXISTS idx_formulas_functions ON formulas(functions);
CREATE INDEX IF NOT EXISTS idx_patent_medicines_name ON patent_medicines(name);
CREATE INDEX IF NOT EXISTS idx_folk_remedies_disease ON folk_remedies(disease);

-- ========== 示例数据 ==========
-- 注意：实际数据库包含 1,641 条数据，此处仅展示字段结构示例

-- 药材数据示例
INSERT INTO herbs (name, category, nature, flavor, meridian, efficacy, indications, usage, contraindications, source) VALUES
('人参', '补气药', '微温', '甘、微苦', '脾、肺、心经', '大补元气，复脉固脱，补脾益肺，生津养血，安神益智', '体虚欲脱，肢冷脉微，脾虚食少，肺虚喘咳', '煎服，3-9g', '实证、热证而正气不虚者忌服', '《神农本草经》'),
('黄芪', '补气药', '微温', '甘', '脾、肺经', '补气升阳，固表止汗，利水消肿，生津养血，托毒排脓', '气虚乏力，食少便溏，中气下陷，表虚自汗', '煎服，9-30g', '表实邪盛，气滞湿阻者慎用', '《神农本草经》'),
('当归', '补血药', '温', '甘、辛', '肝、心、脾经', '补血活血，调经止痛，润肠通便', '血虚萎黄，眩晕心悸，月经不调，经闭痛经', '煎服，6-12g', '湿盛中满，大便泄泻者忌服', '《神农本草经》'),
('茯苓', '利水药', '平', '甘、淡', '心、肺、脾、肾经', '利水渗湿，健脾宁心', '水肿尿少，痰饮眩悸，脾虚食少，便溏泄泻', '煎服，9-15g', '阴虚而无湿热者慎用', '《神农本草经》'),
('陈皮', '理气药', '温', '苦、辛', '脾、肺经', '理气健脾，燥湿化痰', '脘腹胀满，食少吐泻，咳嗽痰多', '煎服，3-10g', '吐血证慎用', '《神农本草经》');

-- 方剂数据示例
INSERT INTO formulas (name, category, source_book, composition, dosage, functions, indications, usage, modification, explanation, source) VALUES
('四君子汤', '补益剂', '《太平惠民和剂局方》', '人参、白术、茯苓、甘草', '水煎服', '益气健脾', '脾胃气虚证，面色萎白，气短乏力，食少便溏', '水煎服', '气滞者加陈皮', '人参为君，白术为臣，茯苓为佐，甘草为使', '《太平惠民和剂局方》'),
('四物汤', '补益剂', '《太平惠民和剂局方》', '熟地黄、当归、白芍、川芎', '水煎服', '补血调血', '营血虚滞证，头晕目眩，心悸失眠', '水煎服', '血热者加黄芩、黄连', '熟地为君，当归为臣，白芍为佐，川芎为使', '《太平惠民和剂局方》'),
('小柴胡汤', '和解剂', '《伤寒论》', '柴胡、黄芩、人参、甘草、半夏、生姜、大枣', '水煎服', '和解少阳', '伤寒少阳证，寒热往来，胸胁苦满', '水煎服', '咳嗽加杏仁、桔梗', '柴胡为君，黄芩为臣，半夏、生姜为佐，人参、甘草、大枣为使', '《伤寒论》');

-- 中成药数据示例
INSERT INTO patent_medicines (name, category, composition, functions, indications, usage_dosage, contraindications, source) VALUES
('六味地黄丸', '补益类', '熟地黄、山茱萸、山药、泽泻、牡丹皮、茯苓', '滋阴补肾', '肾阴虚证，腰膝酸软，头晕目眩，盗汗遗精', '口服，一次 8 丸，一日 3 次', '肾阳虚者慎用', '《小儿药证直诀》'),
('逍遥丸', '理气类', '柴胡、当归、白芍、白术、茯苓、甘草、薄荷、生姜', '疏肝解郁，养血健脾', '肝郁血虚脾弱证，两胁作痛，神疲食少', '口服，一次 6-9g，一日 2 次', '阴虚火旺者慎用', '《太平惠民和剂局方》'),
('藿香正气水', '解表类', '藿香、紫苏、白芷、厚朴、大腹皮、半夏、陈皮、茯苓等', '解表化湿，理气和中', '外感风寒，内伤湿滞，脘腹疼痛，呕吐泄泻', '口服，一次 5-10ml，一日 2 次', '湿热泄泻者慎用', '《太平惠民和剂局方》');

-- 民间验方数据示例
INSERT INTO folk_remedies (disease, remedy, ingredients, method, source) VALUES
('感冒', '姜汤', '生姜 3 片，红糖适量', '生姜加水煮沸，加入红糖，趁热饮用', '民间验方'),
('咳嗽', '冰糖炖雪梨', '雪梨 1 个，冰糖适量', '雪梨挖去核，放入冰糖，隔水炖熟食用', '民间验方'),
('失眠', '酸枣仁茶', '酸枣仁 10g，茯苓 10g', '酸枣仁炒香后与茯苓一起泡水代茶饮', '民间验方'),
('腹泻', '苹果泥', '苹果 1 个', '苹果蒸熟后捣成泥状食用', '民间验方');
