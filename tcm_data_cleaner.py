import json
import re
from typing import List, Dict, Any, Set
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TCMGreedySegmenter:
    """
    基于贪心算法的中药术语分词器
    用于对中药相关的文本进行专业术语的分词处理
    """
    
    def __init__(self):
        """
        初始化分词器
        """
        self.dictionary = set()  # 词典
        self.max_word_length = 0  # 词典中最长词的长度
    
    def add_words(self, words: List[str]) -> None:
        """
        向词典中添加词汇
        
        Args:
            words (List[str]): 词汇列表
        """
        for word in words:
            if word:
                self.dictionary.add(word)
                self.max_word_length = max(self.max_word_length, len(word))
    
    def segment(self, text: str) -> List[str]:
        """
        使用贪心算法对文本进行分词
        
        Args:
            text (str): 待分词的文本
            
        Returns:
            List[str]: 分词结果
        """
        if not text:
            return []
        
        result = []
        position = 0
        
        while position < len(text):
            # 贪心匹配：从最长可能的词开始匹配
            matched = False
            for i in range(min(self.max_word_length, len(text) - position), 0, -1):
                word = text[position:position + i]
                if word in self.dictionary:
                    result.append(word)
                    position += i
                    matched = True
                    break
            
            # 如果没有匹配到词典中的词，则将单个字符作为词
            if not matched:
                result.append(text[position])
                position += 1
        
        return result


class TCMDataCleaner:
    """
    中药数据清洗和规整类
    用于清洗和规整从网站爬取的中药数据
    """
    
    def __init__(self, min_name_length: int = 1, max_name_length: int = 50):
        """
        初始化数据清洗器
        
        Args:
            min_name_length (int): 药材名称最小长度
            max_name_length (int): 药材名称最大长度
        """
        self.min_name_length = min_name_length
        self.max_name_length = max_name_length
        
        # 定义标准的药材性质
        self.standard_natures = {
            "寒", "热", "温", "凉", "平"
        }
        
        # 定义标准的药材味道
        self.standard_tastes = {
            "辛", "甘", "酸", "苦", "咸", "淡", "涩"
        }
        
        # 定义标准的经络名称
        self.standard_channels = {
            "肺经", "大肠经", "胃经", "脾经", "心经", "小肠经",
            "膀胱经", "肾经", "心包经", "三焦经", "胆经", "肝经"
        }
        
        # 定义毒性等级
        self.toxicity_levels = {
            "无毒", "有毒", "小毒", "大毒", "剧毒"
        }
        
        # 初始化分词器
        self.segmenter = TCMGreedySegmenter()
        self._init_segmenter()
    
    def _init_segmenter(self) -> None:
        """
        初始化分词器词典
        """
        # 添加标准术语到分词器
        terms = list(self.standard_natures) + list(self.standard_tastes) + \
                list(self.standard_channels) + list(self.toxicity_levels)
        self.segmenter.add_words(terms)
        
        # 可以添加更多中药专业术语
        additional_terms = [
            # 常见证状
            "头痛", "发热", "咳嗽", "胸闷", "腹痛", "腹泻", "便秘", "失眠", "烦躁",
            # 常见疾病
            "感冒", "咳嗽", "胃炎", "肺炎", "高血压", "糖尿病",
            # 常见功效
            "清热", "解毒", "止咳", "化痰", "健脾", "养胃", "补气", "养血", "活血",
            "祛风", "除湿", "利水", "消肿", "止痛", "安神", "补肾"
        ]
        self.segmenter.add_words(additional_terms)
    
    def load_raw_data(self, filename: str) -> List[Dict[str, Any]]:
        """
        加载原始爬取数据
        
        Args:
            filename (str): 数据文件名
            
        Returns:
            List[Dict[str, Any]]: 原始数据列表
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"成功加载 {len(data)} 条原始数据")
            return data
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            return []
    
    def clean_herb_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        清洗药材数据
        
        Args:
            raw_data (List[Dict[str, Any]]): 原始数据列表
            
        Returns:
            List[Dict[str, Any]]: 清洗后的数据列表
        """
        cleaned_data = []
        
        for herb in raw_data:
            try:
                # 创建清洗后的药材数据
                cleaned_herb = {
                    "herb_id": self._generate_herb_id(herb),
                    "name": self._clean_name(herb.get("name", "")),
                    "pinyin": self._clean_pinyin(herb.get("pinyin", "")),
                    "english_name": self._clean_english_name(herb.get("english_name", "")),
                    "latin_name": self._clean_latin_name(herb.get("latin_name", "")),
                    "nature": self._normalize_nature(herb.get("nature", "")),
                    "taste": self._normalize_taste(herb.get("taste", "")),
                    "toxicity": self._normalize_toxicity(herb.get("toxicity", "")),
                    "channel_tropism": self._normalize_channels(herb.get("channel_tropism", "")),
                    "effect": self._clean_effect(herb.get("effect", "")),
                    "indication": self._clean_indication(herb.get("indication", "")),
                    "contraindication": self._clean_contraindication(herb.get("contraindication", "")),
                    "description": self._clean_description(herb.get("description", "")),
                    "image_url": self._clean_image_url(herb.get("image_url", "")),
                    "source": herb.get("source", ""),
                    "interactions": self._extract_interactions(herb)
                }
                
                # 验证必要字段
                if self._validate_herb_data(cleaned_herb):
                    cleaned_data.append(cleaned_herb)
                else:
                    logger.warning(f"药材数据验证失败，跳过: {cleaned_herb['name']}")
                    
            except Exception as e:
                logger.error(f"清洗药材数据失败: {herb.get('name', '未知')}, 错误: {e}")
                continue
        
        logger.info(f"清洗完成，有效数据 {len(cleaned_data)} 条")
        return cleaned_data
        
    def clean_prescription_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        清洗药方数据
        
        Args:
            raw_data (List[Dict[str, Any]]): 原始药方数据列表
            
        Returns:
            List[Dict[str, Any]]: 清洗后的药方数据列表
        """
        cleaned_data = []
        
        for prescription in raw_data:
            try:
                # 创建清洗后的药方数据
                cleaned_prescription = {
                    "prescription_id": self._generate_prescription_id(prescription),
                    "name": self._clean_name(prescription.get("name", "")),
                    "pinyin": self._clean_pinyin(prescription.get("pinyin", "")),
                    "english_name": self._clean_english_name(prescription.get("english_name", "")),
                    "source": self._clean_source(prescription.get("source", "")),
                    "composition": self._clean_composition(prescription.get("composition", [])),
                    "dosage": self._clean_dosage(prescription.get("dosage", "")),
                    "indications": self._clean_indications(prescription.get("indications", "")),
                    "excerpt": self._clean_excerpt(prescription.get("excerpt", "")),
                    "diseases": self._clean_diseases(prescription.get("diseases", [])),
                    "description": self._clean_description(prescription.get("description", "")),
                    "url": prescription.get("url", ""),
                }
                
                # 验证必要字段
                if self._validate_prescription_data(cleaned_prescription):
                    cleaned_data.append(cleaned_prescription)
                else:
                    logger.warning(f"药方数据验证失败，跳过: {cleaned_prescription['name']}")
                    
            except Exception as e:
                logger.error(f"清洗药方数据失败: {prescription.get('name', '未知')}, 错误: {e}")
                continue
        
        logger.info(f"药方数据清洗完成，有效数据 {len(cleaned_data)} 条")
        return cleaned_data
    
    def _generate_herb_id(self, herb: Dict[str, Any]) -> str:
        """
        生成药材唯一ID
        
        Args:
            herb (Dict[str, Any]): 药材数据
            
        Returns:
            str: 药材唯一ID
        """
        name = herb.get("name", "")
        # 使用名称的拼音首字母和部分数字作为ID
        pinyin_part = self._get_pinyin_initials(herb.get("pinyin", ""))
        # 简单哈希处理
        hash_part = hash(name) % 10000
        return f"HERB_{pinyin_part}_{hash_part:04d}"
        
    def _generate_prescription_id(self, prescription: Dict[str, Any]) -> str:
        """
        生成药方唯一ID
        
        Args:
            prescription (Dict[str, Any]): 药方数据
            
        Returns:
            str: 药方唯一ID
        """
        name = prescription.get("name", "")
        # 使用名称的拼音首字母和部分数字作为ID
        pinyin_part = self._get_pinyin_initials(prescription.get("pinyin", ""))
        # 简单哈希处理
        hash_part = hash(name) % 10000
        return f"PRESC_{pinyin_part}_{hash_part:04d}"
    
    def _get_pinyin_initials(self, pinyin: str) -> str:
        """
        获取拼音首字母
        
        Args:
            pinyin (str): 拼音字符串
            
        Returns:
            str: 拼音首字母
        """
        if not pinyin:
            return "UNKNOWN"
        # 提取每个词的首字母
        initials = "".join([word[0].upper() if word else "" for word in pinyin.split() if word])
        return initials[:8] if initials else "UNKNOWN"
    
    def _clean_name(self, name: str) -> str:
        """
        清洗药材名称
        
        Args:
            name (str): 原始名称
            
        Returns:
            str: 清洗后的名称
        """
        if not name:
            return "未知"
        # 去除多余空格和特殊字符
        cleaned = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\u3000\s]', '', name).strip()
        # 合并多个空格
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # 检查名称长度
        if len(cleaned) < self.min_name_length or len(cleaned) > self.max_name_length:
            return "未知"
            
        return cleaned or "未知"
    
    def _clean_pinyin(self, pinyin: str) -> str:
        """
        清洗拼音
        
        Args:
            pinyin (str): 原始拼音
            
        Returns:
            str: 清洗后的拼音
        """
        if not pinyin:
            return ""
        # 去除多余空格和特殊字符，保留字母和声调数字
        cleaned = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\u02C7\u02CA\s]', '', pinyin).strip()
        return re.sub(r'\s+', ' ', cleaned)
    
    def _clean_english_name(self, english_name: str) -> str:
        """
        清洗英文名
        
        Args:
            english_name (str): 原始英文名
            
        Returns:
            str: 清洗后的英文名
        """
        if not english_name:
            return ""
        # 去除多余空格和特殊字符，保留字母和空格
        cleaned = re.sub(r'[^a-zA-Z\s\-]', '', english_name).strip()
        return re.sub(r'\s+', ' ', cleaned)
    
    def _clean_latin_name(self, latin_name: str) -> str:
        """
        清洗拉丁名
        
        Args:
            latin_name (str): 原始拉丁名
            
        Returns:
            str: 清洗后的拉丁名
        """
        if not latin_name:
            return ""
        # 去除多余空格和特殊字符，保留字母、空格和标点
        cleaned = re.sub(r'[^a-zA-Z\s\.\-\(\)]', '', latin_name).strip()
        return re.sub(r'\s+', ' ', cleaned)
    
    def _normalize_nature(self, nature: str) -> str:
        """
        标准化药材性质
        
        Args:
            nature (str): 原始性质
            
        Returns:
            str: 标准化后的性质
        """
        if not nature:
            return ""
        
        # 转换为标准形式
        nature = nature.strip()
        
        # 匹配标准性质
        for standard in self.standard_natures:
            if standard in nature:
                return standard
                
        # 处理复合性质
        if "温" in nature and "热" in nature:
            return "热"
        if "凉" in nature and "寒" in nature:
            return "寒"
            
        return nature
    
    def _normalize_taste(self, taste: str) -> str:
        """
        标准化药材味道
        
        Args:
            taste (str): 原始味道
            
        Returns:
            str: 标准化后的味道
        """
        if not taste:
            return ""
        
        # 转换为标准形式
        taste = taste.strip()
        
        # 提取标准味道
        normalized_tastes = []
        for standard in self.standard_tastes:
            if standard in taste:
                normalized_tastes.append(standard)
                
        return ",".join(normalized_tastes) if normalized_tastes else taste
    
    def _normalize_toxicity(self, toxicity: str) -> str:
        """
        标准化毒性等级
        
        Args:
            toxicity (str): 原始毒性描述
            
        Returns:
            str: 标准化后的毒性等级
        """
        if not toxicity:
            return ""
        
        # 转换为标准形式
        toxicity = toxicity.strip()
        
        # 匹配标准毒性等级
        for standard in self.toxicity_levels:
            if standard in toxicity:
                return standard
                
        return toxicity
    
    def _normalize_channels(self, channels: str) -> str:
        """
        标准化归经信息
        
        Args:
            channels (str): 原始归经信息
            
        Returns:
            str: 标准化后的归经信息
        """
        if not channels:
            return ""
        
        # 转换为标准形式
        channels = channels.strip()
        
        # 提取标准经络
        normalized_channels = []
        for standard in self.standard_channels:
            if standard in channels:
                normalized_channels.append(standard)
                
        return ",".join(normalized_channels) if normalized_channels else channels
    
    def _clean_effect(self, effect: str) -> str:
        """
        清洗功效信息
        
        Args:
            effect (str): 原始功效信息
            
        Returns:
            str: 清洗后的功效信息
        """
        if not effect:
            return ""
        # 去除多余空格和特殊字符
        cleaned = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\u3000\s\-\，\、\,\;]', '', effect).strip()
        return re.sub(r'\s+', ' ', cleaned)
    
    def _clean_indication(self, indication: str) -> str:
        """
        清洗主治信息
        
        Args:
            indication (str): 原始主治信息
            
        Returns:
            str: 清洗后的主治信息
        """
        if not indication:
            return ""
        # 去除多余空格和特殊字符
        cleaned = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\u3000\s\-\，\、\,\;]', '', indication).strip()
        return re.sub(r'\s+', ' ', cleaned)
    
    def _clean_contraindication(self, contraindication: str) -> str:
        """
        清洗禁忌信息
        
        Args:
            contraindication (str): 原始禁忌信息
            
        Returns:
            str: 清洗后的禁忌信息
        """
        if not contraindication:
            return ""
        # 去除多余空格和特殊字符
        cleaned = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\u3000\s\-\，\、\,\;]', '', contraindication).strip()
        return re.sub(r'\s+', ' ', cleaned)
    
    def _clean_source(self, source: str) -> str:
        """
        清洗来源信息
        
        Args:
            source (str): 原始来源信息
            
        Returns:
            str: 清洗后的来源信息
        """
        if not source:
            return ""
        # 去除多余空格和特殊字符
        cleaned = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\u3000\s\-\，\、\,\;]', '', source).strip()
        return re.sub(r'\s+', ' ', cleaned)
        
    def _clean_composition(self, composition: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        清洗组成成分
        
        Args:
            composition (List[Dict[str, str]]): 原始组成成分列表
            
        Returns:
            List[Dict[str, str]]: 清洗后的组成成分列表
        """
        if not composition:
            return []
            
        cleaned_composition = []
        for item in composition:
            if isinstance(item, dict):
                cleaned_item = {
                    "name": self._clean_name(item.get("name", "")),
                    "dosage": self._clean_dosage_text(item.get("dosage", ""))
                }
                # 只有当名称不为空时才添加
                if cleaned_item["name"]:
                    cleaned_composition.append(cleaned_item)
                    
        return cleaned_composition
        
    def _clean_dosage_text(self, dosage: str) -> str:
        """
        清洗用量文本
        
        Args:
            dosage (str): 原始用量文本
            
        Returns:
            str: 清洗后的用量文本
        """
        if not dosage:
            return ""
        # 去除多余空格和特殊字符，保留数字、单位和常见中文字符
        cleaned = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\u3000\s\-\，\、\,\;克gG]', '', dosage).strip()
        return re.sub(r'\s+', ' ', cleaned)
        
    def _clean_dosage(self, dosage: str) -> str:
        """
        清洗用法用量信息
        
        Args:
            dosage (str): 原始用法用量信息
            
        Returns:
            str: 清洗后的用法用量信息
        """
        if not dosage:
            return ""
        # 去除多余空格和特殊字符
        cleaned = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\u3000\s\-\，\、\,\;\.\。\!\！\?\？克gG]', '', dosage).strip()
        return re.sub(r'\s+', ' ', cleaned)
        
    def _clean_indications(self, indications: str) -> str:
        """
        清洗功能主治信息
        
        Args:
            indications (str): 原始功能主治信息
            
        Returns:
            str: 清洗后的功能主治信息
        """
        if not indications:
            return ""
        # 去除多余空格和特殊字符
        cleaned = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\u3000\s\-\，\、\,\;\.\。\!\！\?\？]', '', indications).strip()
        return re.sub(r'\s+', ' ', cleaned)
        
    def _clean_excerpt(self, excerpt: str) -> str:
        """
        清洗摘录信息
        
        Args:
            excerpt (str): 原始摘录信息
            
        Returns:
            str: 清洗后的摘录信息
        """
        if not excerpt:
            return ""
        # 去除多余空格和特殊字符
        cleaned = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\u3000\s\-\，\、\,\;\.\。\!\！\?\？]', '', excerpt).strip()
        return re.sub(r'\s+', ' ', cleaned)
        
    def _clean_diseases(self, diseases: List[str]) -> List[str]:
        """
        清洗相关疾病列表
        
        Args:
            diseases (List[str]): 原始相关疾病列表
            
        Returns:
            List[str]: 清洗后的相关疾病列表
        """
        if not diseases:
            return []
            
        cleaned_diseases = []
        for disease in diseases:
            cleaned_disease = self._clean_name(disease)
            if cleaned_disease and cleaned_disease != "未知":
                cleaned_diseases.append(cleaned_disease)
                
        return cleaned_diseases
    
    def _clean_image_url(self, image_url: str) -> str:
        """
        清洗图片URL
        
        Args:
            image_url (str): 原始图片URL
            
        Returns:
            str: 清洗后的图片URL
        """
        if not image_url:
            return ""
        # 去除多余空格
        cleaned = image_url.strip()
        # 验证是否为有效的URL格式
        if re.match(r'^https?://', cleaned):
            return cleaned
        return ""
    
    def _extract_interactions(self, herb: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        提取药材相互作用信息
        
        Args:
            herb (Dict[str, Any]): 药材数据
            
        Returns:
            Dict[str, List[str]]: 相互作用信息
        """
        # TODO: 实现从描述中提取相互作用信息的逻辑
        # 目前返回空字典，需要根据实际数据格式和需求完善实现
        return {
            "synergizes_with": [],  # 相使
            "counteracts": [],      # 相恶
            "folks_with": [],       # 相畏
            "incompatible_with": [] # 相反
        }
    
    def _validate_herb_data(self, herb: Dict[str, Any]) -> bool:
        """
        验证药材数据完整性
        
        Args:
            herb (Dict[str, Any]): 药材数据
            
        Returns:
            bool: 验证是否通过
        """
        # 检查必要字段
        required_fields = ["herb_id", "name"]
        for field in required_fields:
            if not herb.get(field):
                logger.warning(f"缺少必要字段: {field}")
                return False
        
        # 检查名称是否有效
        if herb["name"] == "未知":
            logger.warning("药材名称无效")
            return False
            
        return True
        
    def _validate_prescription_data(self, prescription: Dict[str, Any]) -> bool:
        """
        验证药方数据完整性
        
        Args:
            prescription (Dict[str, Any]): 药方数据
            
        Returns:
            bool: 验证是否通过
        """
        # 检查必要字段
        required_fields = ["prescription_id", "name"]
        for field in required_fields:
            if not prescription.get(field):
                logger.warning(f"缺少必要字段: {field}")
                return False
        
        # 检查名称是否有效
        if prescription["name"] == "未知":
            logger.warning("药方名称无效")
            return False
            
        return True
    
    def extract_symptoms_diseases(self, cleaned_data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        从清洗后的药材数据中提取症状和疾病信息
        
        Args:
            cleaned_data (List[Dict[str, Any]]): 清洗后的药材数据
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: 症状和疾病信息
        """
        symptoms = {}
        diseases = {}
        
        # 从主治信息中提取症状和疾病
        for herb in cleaned_data:
            indication = herb.get("indication", "")
            if indication:
                # 简单分词提取症状和疾病名称
                terms = re.split(r'[\s\,\，\、\;\；]', indication)
                for term in terms:
                    term = term.strip()
                    if term:
                        # 假设长度较短的为症状，较长的为疾病
                        if len(term) <= 6:
                            symptom_id = f"SYM_{hash(term) % 10000:04d}"
                            if symptom_id not in symptoms:
                                symptoms[symptom_id] = {
                                    "symptom_id": symptom_id,
                                    "name": term,
                                    "pinyin": "",
                                    "description": "",
                                    "related_herbs": [herb["herb_id"]]
                                }
                            else:
                                symptoms[symptom_id]["related_herbs"].append(herb["herb_id"])
                        else:
                            disease_id = f"DIS_{hash(term) % 10000:04d}"
                            if disease_id not in diseases:
                                diseases[disease_id] = {
                                    "disease_id": disease_id,
                                    "name": term,
                                    "pinyin": "",
                                    "description": "",
                                    "related_herbs": [herb["herb_id"]]
                                }
                            else:
                                diseases[disease_id]["related_herbs"].append(herb["herb_id"])
        
        return {
            "symptoms": list(symptoms.values()),
            "diseases": list(diseases.values())
        }
    
    def extract_effects(self, cleaned_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        从清洗后的药材数据中提取功效信息
        
        Args:
            cleaned_data (List[Dict[str, Any]]): 清洗后的药材数据
            
        Returns:
            List[Dict[str, Any]]: 功效信息
        """
        effects = {}
        
        # 从功效信息中提取功效
        for herb in cleaned_data:
            effect_text = herb.get("effect", "")
            if effect_text:
                # 简单分词提取功效名称
                terms = re.split(r'[\s\,\，\、\;\；]', effect_text)
                for term in terms:
                    term = term.strip()
                    if term:
                        effect_id = f"EFF_{hash(term) % 10000:04d}"
                        if effect_id not in effects:
                            effects[effect_id] = {
                                "effect_id": effect_id,
                                "name": term,
                                "pinyin": "",
                                "description": "",
                                "related_herbs": [herb["herb_id"]]
                            }
                        else:
                            effects[effect_id]["related_herbs"].append(herb["herb_id"])
        
        return list(effects.values())
    
    def save_to_json(self, data: List[Dict[str, Any]], filename: str):
        """
        保存数据到JSON文件
        
        Args:
            data (List[Dict[str, Any]]): 要保存的数据
            filename (str): 文件名
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"数据已保存到 {filename}")
        except Exception as e:
            logger.error(f"保存数据失败: {e}")

    def save_cleaned_data(self, data: Dict[str, Any], filename: str):
        """
        保存清洗后的数据
        
        Args:
            data (Dict[str, Any]): 清洗后的数据
            filename (str): 文件名
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"清洗后的数据已保存到 {filename}")
        except Exception as e:
            logger.error(f"保存清洗数据失败: {e}")


def main():
    """
    主函数，演示如何使用TCMDataCleaner
    """
    # 创建数据清洗器实例
    cleaner = TCMDataCleaner()
    
    # 加载原始数据
    raw_data = cleaner.load_raw_data("tcm_herbs_data.json")
    
    if not raw_data:
        logger.error("未加载到原始数据，退出")
        return
    
    # 清洗药材数据
    cleaned_herbs = cleaner.clean_herb_data(raw_data)
    
    # 提取症状和疾病信息
    symptoms_diseases = cleaner.extract_symptoms_diseases(cleaned_herbs)
    
    # 提取功效信息
    effects = cleaner.extract_effects(cleaned_herbs)
    
    # 组织所有数据
    cleaned_data = {
        "herbs": cleaned_herbs,
        "symptoms": symptoms_diseases["symptoms"],
        "diseases": symptoms_diseases["diseases"],
        "effects": effects
    }
    
    # 保存清洗后的数据
    cleaner.save_cleaned_data(cleaned_data, "tcm_cleaned_data.json")
    
    # 打印部分清洗后的数据作为示例
    print("\n清洗后的部分药材数据:")
    for herb in cleaned_herbs[:3]:
        print(f"药材ID: {herb['herb_id']}")
        print(f"药材名称: {herb['name']}")
        print(f"性味归经: {herb['nature']}, {herb['taste']}, {herb['channel_tropism']}")
        print(f"功效: {herb['effect']}")
        print("-" * 50)


if __name__ == "__main__":
    main()