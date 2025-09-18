#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
中药数据采集和处理流水线
整合爬虫、数据清洗、知识图谱构建和训练启动功能
"""

import os
import sys
import json
import logging
import argparse
import subprocess
from datetime import datetime
from typing import List, Dict, Any

# 添加项目路径到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入项目模块
try:
    from tcm_scraper import TCMDataScraper
    SCRAPER_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入tcm_scraper模块: {e}")
    SCRAPER_AVAILABLE = False
    
try:
    from tcm_data_cleaner import TCMDataCleaner
    CLEANER_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入tcm_data_cleaner模块: {e}")
    CLEANER_AVAILABLE = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_collection_pipeline.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TCMDataCollectionPipeline:
    """中药数据采集和处理流水线"""
    
    def __init__(self, config_file: str = "config.json"):
        """
        初始化数据采集流水线
        
        Args:
            config_file (str): 配置文件路径
        """
        self.config_file = config_file
        self.config = self._load_config()
        self.scraper = None
        self.cleaner = None
    
    def _load_config(self) -> Dict[str, Any]:
        """
        加载配置文件
        
        Returns:
            Dict[str, Any]: 配置信息
        """
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"配置文件 {self.config_file} 不存在，使用默认配置")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        获取默认配置
        
        Returns:
            Dict[str, Any]: 默认配置信息
        """
        return {
            "scraper": {
                "delay_range": [2, 5],
                "max_retries": 3,
                "max_herbs": 1000
            },
            "cleaner": {
                "min_name_length": 1,
                "max_name_length": 50
            },
            "data_paths": {
                "raw_data": "data/raw/tcm_herbs_data.json",
                "cleaned_data": "data/cleaned/tcm_herbs_cleaned.json",
                "kg_data": "data/kg/tcm_kg_data.json",
                "train_data": "data/train/tcm_train_data.json"
            }
        }
    
    def run_scraper(self, max_herbs: int = None) -> bool:
        """
        运行爬虫采集数据
        
        Args:
            max_herbs (int): 最大采集药材数量
            
        Returns:
            bool: 是否成功
        """
        logger.info("开始运行爬虫采集数据...")
        
        # 检查爬虫模块是否可用
        if not SCRAPER_AVAILABLE:
            logger.error("爬虫模块不可用，请检查tcm_scraper.py文件是否存在且正确实现")
            return False
            
        try:
            # 创建爬虫实例
            scraper_config = self.config.get("scraper", {})
            delay_range = tuple(scraper_config.get("delay_range", [2, 5]))
            max_retries = scraper_config.get("max_retries", 3)
            
            self.scraper = TCMDataScraper(
                delay_range=delay_range,
                max_retries=max_retries
            )
            
            # 设置最大采集数量
            if max_herbs is None:
                max_herbs = scraper_config.get("max_herbs", 1000)
            
            # 爬取数据
            herbs_data = self.scraper.scrape_all_herbs(max_herbs=max_herbs)
            
            # 保存原始数据
            raw_data_path = self.config["data_paths"]["raw_data"]
            os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
            
            self.scraper.save_to_json(herbs_data, raw_data_path)
            logger.info(f"爬虫采集完成，共获得 {len(herbs_data)} 条数据，已保存到 {raw_data_path}")
            
            return True
        except Exception as e:
            logger.error(f"爬虫运行失败: {e}")
            return False
    
    def run_cleaner(self) -> bool:
        """
        运行数据清洗
        
        Returns:
            bool: 是否成功
        """
        logger.info("开始运行数据清洗...")
        
        # 检查清洗器模块是否可用
        if not CLEANER_AVAILABLE:
            logger.error("数据清洗模块不可用，请检查tcm_data_cleaner.py文件是否存在且正确实现")
            return False
            
        try:
            # 创建清洗器实例
            cleaner_config = self.config.get("cleaner", {})
            self.cleaner = TCMDataCleaner(
                min_name_length=cleaner_config.get("min_name_length", 1),
                max_name_length=cleaner_config.get("max_name_length", 50)
            )
            
            # 加载原始数据
            raw_data_path = self.config["data_paths"]["raw_data"]
            if not os.path.exists(raw_data_path):
                logger.error(f"原始数据文件不存在: {raw_data_path}")
                return False
            
            with open(raw_data_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            # 清洗数据
            cleaned_data = self.cleaner.clean_herb_data(raw_data)
            
            # 保存清洗后的数据
            cleaned_data_path = self.config["data_paths"]["cleaned_data"]
            os.makedirs(os.path.dirname(cleaned_data_path), exist_ok=True)
            
            self.cleaner.save_to_json(cleaned_data, cleaned_data_path)
            logger.info(f"数据清洗完成，清洗后剩余 {len(cleaned_data)} 条数据，已保存到 {cleaned_data_path}")
            
            return True
        except Exception as e:
            logger.error(f"数据清洗失败: {e}")
            return False
    
    def build_knowledge_graph_data(self) -> bool:
        """
        构建知识图谱数据
        
        Returns:
            bool: 是否成功
        """
        logger.info("开始构建知识图谱数据...")
        
        try:
            # 加载清洗后的数据
            cleaned_data_path = self.config["data_paths"]["cleaned_data"]
            if not os.path.exists(cleaned_data_path):
                logger.error(f"清洗后数据文件不存在: {cleaned_data_path}")
                return False
            
            with open(cleaned_data_path, 'r', encoding='utf-8') as f:
                cleaned_data = json.load(f)
            
            # 初始化知识图谱数据结构
            kg_data = {
                "herbs": [],
                "symptoms": [],
                "diseases": [],
                "effects": [],
                "relations": []
            }
            
            # 如果是列表格式，直接使用
            if isinstance(cleaned_data, list):
                kg_data["herbs"] = cleaned_data
            # 如果是字典格式，提取各个部分
            elif isinstance(cleaned_data, dict):
                kg_data["herbs"] = cleaned_data.get("herbs", [])
                kg_data["symptoms"] = cleaned_data.get("symptoms", [])
                kg_data["diseases"] = cleaned_data.get("diseases", [])
                kg_data["effects"] = cleaned_data.get("effects", [])
                
                # 构建实体间的关系
                self._build_entity_relations(kg_data)
            
            # 保存知识图谱数据
            kg_data_path = self.config["data_paths"]["kg_data"]
            os.makedirs(os.path.dirname(kg_data_path), exist_ok=True)
            
            with open(kg_data_path, 'w', encoding='utf-8') as f:
                json.dump(kg_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"知识图谱数据构建完成，已保存到 {kg_data_path}")
            return True
        except Exception as e:
            logger.error(f"知识图谱数据构建失败: {e}")
            return False
    
    def _generate_sample_prescriptions(self, kg_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        生成示例处方数据（实际应用中应该从真实数据或专业规则生成）
        
        Args:
            kg_data (Dict[str, Any]): 知识图谱数据
            
        Returns:
            List[Dict[str, Any]]: 示例处方数据
        """
        # TODO: 实现真实的处方生成逻辑
        # 这里应该基于真实的中医药理论和知识图谱中的实体关系来生成处方
        # 目前保留示例代码，但需要根据实际需求修改
        
        herbs = kg_data.get("herbs", [])
        sample_prescriptions = []
        
        # 为前10种药材生成示例处方
        for i, herb in enumerate(herbs[:10]):
            sample_prescriptions.append({
                "id": f"prescription_{i}",
                "name": f"示例处方{i+1}",
                "herbs": [herb["herb_id"]] + [h["herb_id"] for h in herbs[i+1:i+4]],  # 包含当前药材和后续3种药材
                "symptoms": [f"symptom_{j}" for j in range(i*2, i*2+3)],  # 生成示例症状
                "indications": herb.get("indication", ""),
                "source": "generated_sample"
            })
        
        return sample_prescriptions
    
    def _build_entity_relations(self, kg_data: Dict[str, Any]) -> None:
        """
        构建知识图谱中实体间的关系
        
        Args:
            kg_data (Dict[str, Any]): 知识图谱数据
        """
        # TODO: 实现实体间关系的构建逻辑
        # 这里需要根据药材、症状、疾病、功效之间的实际关联关系来构建
        # 目前留空，需要根据数据特点和领域知识来实现
        pass
    
    def prepare_training_data(self) -> bool:
        """
        准备训练数据
        
        Returns:
            bool: 是否成功
        """
        logger.info("开始准备训练数据...")
        
        try:
            # 加载知识图谱数据
            kg_data_path = self.config["data_paths"]["kg_data"]
            if not os.path.exists(kg_data_path):
                logger.error(f"知识图谱数据文件不存在: {kg_data_path}")
                return False
            
            with open(kg_data_path, 'r', encoding='utf-8') as f:
                kg_data = json.load(f)
            
            # 准备训练数据（这里简化处理，实际应用中可能需要更复杂的处理）
            train_data = {
                "herbs": kg_data.get("herbs", []),
                "symptoms": kg_data.get("symptoms", []),
                "relations": kg_data.get("relations", []),
                "prescriptions": self._generate_sample_prescriptions(kg_data)
            }
            
            # 保存训练数据
            train_data_path = self.config["data_paths"]["train_data"]
            os.makedirs(os.path.dirname(train_data_path), exist_ok=True)
            
            with open(train_data_path, 'w', encoding='utf-8') as f:
                json.dump(train_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"训练数据准备完成，已保存到 {train_data_path}")
            return True
        except Exception as e:
            logger.error(f"训练数据准备失败: {e}")
            return False
    
    def start_training(self) -> bool:
        """
        启动模型训练
        
        Returns:
            bool: 是否成功
        """
        logger.info("开始启动模型训练...")
        
        try:
            # 使用subprocess运行训练脚本
            train_script = "tcm_gnn_real_training.py"
            if not os.path.exists(train_script):
                logger.error(f"训练脚本不存在: {train_script}")
                return False
            
            # 构建命令
            cmd = [sys.executable, train_script, "--mode", "train"]
            
            # 运行训练脚本
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            
            if result.returncode == 0:
                logger.info("模型训练启动成功")
                logger.info(f"训练输出:\n{result.stdout}")
                return True
            else:
                logger.error(f"模型训练启动失败: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"启动模型训练失败: {e}")
            return False


    def run_full_pipeline(self, max_herbs: int = None) -> bool:
        """
        运行完整的数据采集和处理流水线
        
        Args:
            max_herbs (int): 最大采集药材数量
            
        Returns:
            bool: 是否成功
        """
        logger.info("开始运行完整的数据采集和处理流水线...")
        
        # 1. 运行爬虫
        if not self.run_scraper(max_herbs):
            logger.error("爬虫运行失败，流水线中断")
            return False
        
        # 2. 运行数据清洗
        if not self.run_cleaner():
            logger.error("数据清洗失败，流水线中断")
            return False
        
        # 3. 构建知识图谱数据
        if not self.build_knowledge_graph_data():
            logger.error("知识图谱数据构建失败，流水线中断")
            return False
        
        # 4. 准备训练数据
        if not self.prepare_training_data():
            logger.error("训练数据准备失败，流水线中断")
            return False
        
        logger.info("完整的数据采集和处理流水线运行完成")
        return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="中药数据采集和处理流水线")
    parser.add_argument("--max-herbs", type=int, help="最大采集药材数量")
    parser.add_argument("--skip-training", action="store_true", help="跳过模型训练")
    parser.add_argument("--only-scrape", action="store_true", help="仅运行爬虫")
    parser.add_argument("--config", default="config.json", help="配置文件路径")
    
    args = parser.parse_args()
    
    # 创建流水线实例
    pipeline = TCMDataCollectionPipeline(config_file=args.config)
    
    # 根据参数决定运行哪些步骤
    if args.only_scrape:
        # 仅运行爬虫
        success = pipeline.run_scraper(args.max_herbs)
    else:
        # 运行完整流水线
        success = pipeline.run_full_pipeline(args.max_herbs)
        
        # 如果不跳过训练且流水线成功，则启动训练
        if success and not args.skip_training:
            success = pipeline.start_training()
    
    if success:
        logger.info("程序执行成功完成")
        return 0
    else:
        logger.error("程序执行失败")
        return 1