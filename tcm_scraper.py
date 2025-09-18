#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
中药数据爬虫模块
负责从网络爬取中药相关数据
"""

import os
import sys
import json
import time
import random
import logging
import re
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# 添加项目路径到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TCMDataScraper:
    """中药数据爬虫类"""
    
    def __init__(self, delay_range: tuple = (1, 3), max_retries: int = 3):
        """
        初始化爬虫
        
        Args:
            delay_range (tuple): 请求延迟范围（秒）
            max_retries (int): 最大重试次数
        """
        self.delay_range = delay_range
        self.max_retries = max_retries
        self.session = requests.Session()
        # 设置User-Agent以模拟浏览器访问
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        # 设置基础URL
        self.base_url = "http://www.cnzyao.com"
        
    def _get_page(self, url: str) -> Optional[requests.Response]:
        """
        获取网页内容
        
        Args:
            url (str): 目标URL
            
        Returns:
            Optional[requests.Response]: 响应对象，如果失败则返回None
        """
        for attempt in range(self.max_retries):
            try:
                # 随机延迟，避免请求过于频繁
                delay = random.uniform(*self.delay_range)
                time.sleep(delay)
                
                response = self.session.get(url, timeout=10)
                response.raise_for_status()  # 检查HTTP错误
                # 设置正确的编码
                response.encoding = 'utf-8'
                return response
            except requests.RequestException as e:
                logger.warning(f"请求失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    logger.error(f"请求最终失败: {url}")
                    return None
        return None

    def scrape_single_herb(self, url: str) -> Optional[Dict[str, Any]]:
        """
        爬取单个药材信息
        
        Args:
            url (str): 药材详情页URL
            
        Returns:
            Optional[Dict[str, Any]]: 药材信息字典，如果失败则返回None
        """
        logger.info(f"正在爬取药材信息: {url}")
        response = self._get_page(url)
        if not response:
            return None
            
        try:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 初始化药材数据
            herb_data = {
                "name": "",  # 药材名称
                "pinyin": "",  # 拼音
                "english_name": "",  # 英文名
                "latin_name": "",  # 拉丁学名
                "nature": "",  # 性味
                "taste": "",  # 味道
                "channel_tropism": "",  # 归经
                "effect": "",  # 功效
                "indication": "",  # 主治
                "contraindication": "",  # 禁忌
                "toxicity": "",  # 毒性
                "description": "",  # 描述
                "image_url": "",  # 图片URL
                "source": url  # 数据来源
            }
            
            # 提取药材名称 (从<h1>标签)
            name_element = soup.find('h1')
            if name_element:
                herb_data["name"] = name_element.get_text(strip=True)
            
            # 提取各字段信息
            dls = soup.find_all('dl')
            for dl in dls:
                dt = dl.find('dt')
                dd = dl.find('dd')
                if dt and dd:
                    dt_text = dt.get_text(strip=True)
                    dd_text = dd.get_text(strip=True)
                    
                    # 根据不同的dt_text提取对应的信息
                    if "英文名" in dt_text:
                        herb_data["english_name"] = dd_text
                    elif "拼音" in dt_text:
                        herb_data["pinyin"] = dd_text
                    elif "别名" in dt_text:
                        # 别名暂时不处理，可以作为扩展
                        pass
                    elif "性味" in dt_text:
                        herb_data["nature"] = dd_text
                    elif "归经" in dt_text:
                        herb_data["channel_tropism"] = dd_text
                    elif "入药部分" in dt_text:
                        # 入药部分可以作为描述的一部分
                        herb_data["description"] += f"入药部分: {dd_text} "
                    elif "功效与作用" in dt_text:
                        herb_data["effect"] = dd_text
                    elif "用法用量" in dt_text:
                        # 用法用量可以作为描述的一部分
                        herb_data["description"] += f"用法用量: {dd_text} "
                    elif "注意事项" in dt_text:
                        herb_data["contraindication"] = dd_text
                    elif "形态特征" in dt_text:
                        herb_data["description"] += f"形态特征: {dd_text} "
                    elif "产地和分布" in dt_text:
                        herb_data["description"] += f"产地和分布: {dd_text} "
            
            # 提取图片URL
            img_element = soup.find('img')
            if img_element and img_element.get('src'):
                img_src = img_element['src']
                # 如果是相对路径，转换为绝对路径
                if img_src.startswith('/'):
                    herb_data["image_url"] = urljoin(self.base_url, img_src)
                else:
                    herb_data["image_url"] = img_src
            
            # 提取描述 (meta标签中的description)
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc and meta_desc.get('content'):
                herb_data["indication"] = meta_desc['content']
            
            # 如果没有从meta标签获取到主治信息，则从功效与作用中提取
            if not herb_data["indication"] and herb_data["effect"]:
                # 简单地将功效与作用作为主治
                herb_data["indication"] = herb_data["effect"]
                
            return herb_data
        except Exception as e:
            logger.error(f"解析药材信息失败: {e}")
            return None

    def scrape_all_herbs(self, max_herbs: int = 100) -> List[Dict[str, Any]]:
        """
        爬取所有药材信息
        
        Args:
            max_herbs (int): 最大爬取药材数量
            
        Returns:
            List[Dict[str, Any]]: 药材信息列表
        """
        logger.info("开始爬取所有药材信息...")
        
        herbs_data = []
        
        # 定义药材分类URL列表
        category_urls = [
            "http://www.cnzyao.com/yaocai/gj/",   # 根茎类
            "http://www.cnzyao.com/yaocai/qc/",   # 全草类
            "http://www.cnzyao.com/yaocai/gszz/", # 果实/种子类
            "http://www.cnzyao.com/yaocai/tm/",   # 藤木类
            "http://www.cnzyao.com/yaocai/dw/",   # 动物类
            "http://www.cnzyao.com/yaocai/zysp/", # 枝叶/树皮类
            "http://www.cnzyao.com/yaocai/jz/",   # 菌藻类
            "http://www.cnzyao.com/yaocai/kw/",   # 矿物类
            "http://www.cnzyao.com/yaocai/qt/"    # 其他
        ]
        
        # 遍历每个分类
        for category_url in category_urls:
            logger.info(f"正在爬取分类: {category_url}")
            
            # 获取分类页面
            response = self._get_page(category_url)
            if not response:
                continue
                
            # 解析分类页面，提取药材链接
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 查找所有药材链接
            herb_links = []
            # 查找所有<a>标签
            links = soup.find_all('a', href=True)
            for link in links:
                href = link['href']
                # 检查链接是否指向药材详情页 (以数字结尾的.html)
                if re.match(r'/yaocai/[a-z]+/\d+\.html', href):
                    full_url = urljoin(self.base_url, href)
                    herb_links.append(full_url)
            
            # 去重
            herb_links = list(set(herb_links))
            
            logger.info(f"在分类 {category_url} 中找到 {len(herb_links)} 个药材链接")
            
            # 爬取每个药材的信息
            for i, herb_url in enumerate(herb_links):
                if len(herbs_data) >= max_herbs:
                    break
                    
                # 爬取单个药材信息
                herb_data = self.scrape_single_herb(herb_url)
                if herb_data and herb_data.get("name"):
                    herbs_data.append(herb_data)
                    logger.info(f"已爬取药材: {herb_data['name']}")
                else:
                    logger.warning(f"爬取药材失败: {herb_url}")
                
                # 显示进度
                if (len(herbs_data) > 0) and (len(herbs_data) % 10 == 0):
                    logger.info(f"已爬取 {len(herbs_data)} 个药材")
        
        logger.info(f"药材信息爬取完成，共获得 {len(herbs_data)} 条数据")
        return herbs_data

    def save_to_json(self, data: List[Dict[str, Any]], file_path: str):
        """
        将数据保存为JSON文件
        
        Args:
            data (List[Dict[str, Any]]): 要保存的数据
            file_path (str): 文件路径
        """
        try:
            # 如果提供了目录路径，则确保目录存在
            if os.path.dirname(file_path):
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # 保存数据
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"数据已保存到 {file_path}")
        except Exception as e:
            logger.error(f"保存数据失败: {e}")


def main():
    """主函数，用于测试爬虫功能"""
    # 创建爬虫实例
    scraper = TCMDataScraper(delay_range=(1, 2))
    
    # 爬取数据
    herbs_data = scraper.scrape_all_herbs(max_herbs=50)
    
    # 保存数据到当前目录
    output_file = os.path.join(os.path.dirname(__file__), "tcm_herbs_data.json") if __file__ else "tcm_herbs_data.json"
    scraper.save_to_json(herbs_data, output_file)
    
    # 打印结果
    print(json.dumps(herbs_data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()