#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
中药药方数据爬虫模块（修复版）
负责从网络爬取中药方剂相关数据
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
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import requests
from bs4 import BeautifulSoup

# 添加项目路径到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('prescription_scraper.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TCMPrescriptionScraper:
    """中药药方数据爬虫类（修复版）"""
    
    def __init__(self, delay_range: tuple = (1, 3), max_retries: int = 5, proxies=None, max_workers=3):
        """
        初始化爬虫
        
        Args:
            delay_range (tuple): 请求延迟范围（秒）
            max_retries (int): 最大重试次数
            proxies (list): 代理IP列表，格式如 [{'http': 'http://ip:port', 'https': 'https://ip:port'}]
            max_workers (int): 并发线程数
        """
        self.delay_range = delay_range
        self.max_retries = max_retries
        self.proxies = proxies or []
        self.max_workers = max_workers
        self.current_proxy_index = 0
        self.session = requests.Session()
        
        # 线程锁，用于保护代理索引等共享资源
        self.lock = threading.Lock()
        
        # User-Agent池，模拟不同浏览器访问
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        ]
        
        # 设置基础URL
        self.base_url = "http://www.cnzyao.com"
        
        # 初始化会话头信息
        self._set_random_headers()
    
    def _set_random_headers(self):
        """设置随机请求头，模拟不同浏览器"""
        user_agent = random.choice(self.user_agents)
        self.session.headers.update({
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
    
    def _get_random_proxy(self):
        """获取随机代理"""
        if not self.proxies:
            return None
        with self.lock:
            proxy = random.choice(self.proxies)
        return proxy
    
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
                # 根据尝试次数动态调整延迟时间，越往后越长
                base_delay = random.uniform(*self.delay_range)
                delay = base_delay * (1.5 ** attempt)  # 指数退避
                time.sleep(delay)
                
                # 随机更换User-Agent
                if attempt > 0:  # 第一次使用初始化的User-Agent
                    self._set_random_headers()
                
                # 获取代理
                proxy = self._get_random_proxy()
                
                # 发送请求
                response = self.session.get(url, timeout=15, proxies=proxy)
                response.raise_for_status()  # 检查HTTP错误
                
                # 检查响应内容是否正常
                if "验证码" in response.text or "blocked" in response.text.lower():
                    logger.warning(f"遇到验证码或被阻止，尝试切换代理或User-Agent")
                    # 如果有代理则切换代理，否则继续使用指数退避
                    if self.proxies:
                        # 旋转到下一个代理
                        with self.lock:
                            self.current_proxy_index = (self.current_proxy_index + 1) % len(self.proxies)
                    continue  # 重试
                
                # 设置正确的编码
                response.encoding = 'utf-8'
                return response
            except requests.RequestException as e:
                logger.warning(f"请求失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                # 根据错误类型调整策略
                if isinstance(e, requests.exceptions.ConnectionError):
                    # 连接错误时切换代理
                    if self.proxies:
                        with self.lock:
                            self.current_proxy_index = (self.current_proxy_index + 1) % len(self.proxies)
                elif isinstance(e, requests.exceptions.Timeout):
                    # 超时错误时增加延迟
                    self.delay_range = (self.delay_range[0] * 1.2, self.delay_range[1] * 1.2)
                if attempt == self.max_retries - 1:
                    logger.error(f"请求最终失败: {url}")
                    return None
        return None

    def scrape_single_prescription(self, url: str) -> Optional[Dict[str, Any]]:
        """
        爬取单个药方信息
        
        Args:
            url (str): 药方详情页URL
            
        Returns:
            Optional[Dict[str, Any]]: 药方信息字典，如果失败则返回None
        """
        logger.info(f"正在爬取药方信息: {url}")
        response = self._get_page(url)
        if not response:
            return None
            
        try:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 初始化药方数据
            prescription_data = {
                "name": "",  # 方剂名称
                "pinyin": "",  # 拼音名
                "english_name": "",  # 英文名
                "source": "",  # 来源
                "composition": [],  # 组成成分（药材列表）
                "dosage": "",  # 用法用量
                "indications": "",  # 功能主治
                "excerpt": "",  # 摘录
                "diseases": [],  # 相关疾病
                "description": "",  # 详细描述
                "url": url  # 数据来源URL
            }
            
            # 提取方剂名称 (从<h1>标签)
            name_element = soup.find('h1')
            if name_element:
                prescription_data["name"] = name_element.get_text(strip=True) if hasattr(name_element, 'get_text') and name_element.get_text else ""
            
            # 提取各字段信息
            dls = soup.find_all('dl')
            for dl in dls:
                dt = dl.find('dt')
                dd = dl.find('dd')
                # 确保元素存在且有get_text方法
                if dt and dd and hasattr(dt, 'get_text') and hasattr(dd, 'get_text'):
                    dt_text = dt.get_text(strip=True) if hasattr(dt, 'get_text') and dt.get_text else ""
                    dd_text = dd.get_text(strip=True) if hasattr(dd, 'get_text') and dd.get_text else ""
                    
                    # 根据不同的dt_text提取对应的信息
                    if "方剂名称" in dt_text:
                        prescription_data["name"] = dd_text
                    elif "别名" in dt_text:
                        # 别名可以作为名称的一部分
                        prescription_data["name"] += f" ({dd_text})" if prescription_data["name"] else dd_text
                    elif "拼音" in dt_text:
                        prescription_data["pinyin"] = dd_text
                    elif "英文名" in dt_text:
                        prescription_data["english_name"] = dd_text
                    elif "来源" in dt_text or "方剂出处" in dt_text:
                        prescription_data["source"] = dd_text
                    elif "组成" in dt_text or "配方组成" in dt_text:
                        prescription_data["composition"] = self._parse_composition(dd_text)
                    elif "用法用量" in dt_text or "使用方法" in dt_text:
                        prescription_data["dosage"] = dd_text
                    elif "功能主治" in dt_text or "功效和作用" in dt_text:
                        prescription_data["indications"] = dd_text
                    elif "摘录" in dt_text:
                        prescription_data["excerpt"] = dd_text
                    elif "相关疾病" in dt_text:
                        prescription_data["diseases"] = [d.strip() for d in dd_text.split() if d and d.strip()]
                    elif "方剂歌诀" in dt_text:
                        prescription_data["description"] += f"方剂歌诀: {dd_text} "
                    elif "方解" in dt_text:
                        prescription_data["description"] += f"方解: {dd_text} "
                    elif "临床应用" in dt_text:
                        prescription_data["description"] += f"临床应用: {dd_text} "
                    elif "注意事项" in dt_text:
                        prescription_data["description"] += f"注意事项: {dd_text} "
            
            # 提取详细描述 (如果没有从其他字段获取到相关信息)
            if not prescription_data["description"]:
                # 查找描述性段落
                paragraphs = soup.find_all('p')
                for p in paragraphs:
                    # 确保元素存在且有get_text方法
                    if p and hasattr(p, 'get_text'):
                        text = p.get_text(strip=True) if hasattr(p, 'get_text') and p.get_text else ""
                        if text and len(text) > 20:  # 过滤掉太短的文本
                            prescription_data["description"] = text
                            break
            
            # 如果仍然没有描述，尝试从meta标签获取
            if not prescription_data["description"]:
                meta_desc = soup.find('meta', attrs={'name': 'description'})
                if meta_desc and meta_desc.get('content'):
                    prescription_data["description"] = meta_desc['content']
            
            # 确保name字段不为空
            if not prescription_data["name"]:
                # 如果没有提取到名称，尝试从URL中提取
                import re
                match = re.search(r'/([^/]+)\.html$', url)
                if match:
                    prescription_data["name"] = match.group(1)
            
            return prescription_data
        except Exception as e:
            logger.error(f"解析药方信息失败: {e}")
            return None

    def _parse_composition(self, composition_text: str) -> List[Dict[str, str]]:
        """
        解析组成成分文本
        
        Args:
            composition_text (str): 组成成分文本
            
        Returns:
            List[Dict[str, str]]: 药材列表，每个药材包含名称和用量
        """
        herbs = []
        if not composition_text:
            return herbs
            
        # 先尝试处理带有HTML链接的格式，如：
        # <a href="http://www.cnzyao.com/yaocai/gj/2535.html" target="_blank" >党参</a>12克
        # 或者 <a href="http://www.cnzyao.com/yaocai/gj/dangshen.html" target="_blank" >党参</a>12克
        # 匹配药材链接和用量
        pattern = r'<a[^>]*>([^<]+)</a>\s*([0-9.]+克?)?'
        matches = re.findall(pattern, composition_text)
        
        if matches:
            # 处理匹配到的药材
            for name, dosage in matches:
                # 清理名称，去除多余的字符
                clean_name = re.sub(r'[，,、]', '', name.strip()) if name else ""
                clean_dosage = dosage.strip() if dosage else ""
                herbs.append({"name": clean_name, "dosage": clean_dosage})
        else:
            # 如果没有匹配到HTML链接格式，使用更优化的解析方法
            # 首先尝试处理类似 "党参12克，当归9克" 的格式
            # 先按常见分隔符分割
            parts = re.split(r'[，,、]', composition_text)
            
            # 如果分割后的部分很少，可能是没有正确分割的情况
            if len(parts) <= 1:
                # 尝试更复杂的分割方式
                # 匹配类似于 "党参12克 当归9克" 或 "党参12克,当归9克" 的格式
                complex_pattern = r'([^0-9，,、]+?)((?:\d+(?:\.\d+)?(?:-(?:\d+(?:\.\d+)?)?)?[克gG钱两]|适量|少许|少许许|各等分)(?:\s*[，,、]?)?)'
                complex_matches = re.findall(complex_pattern, composition_text)
                
                if complex_matches:
                    for name, dosage in complex_matches:
                        clean_name = name.strip() if name else ""
                        clean_dosage = dosage.strip() if dosage else ""
                        # 去除剂量中的分隔符
                        clean_dosage = re.sub(r'[，,、]', '', clean_dosage)
                        herbs.append({"name": clean_name, "dosage": clean_dosage})
                else:
                    # 如果仍然没有匹配到，使用原来的简单分割方式
                    for part in parts:
                        part = part.strip() if part else ""
                        if part:
                            # 尝试提取药材名称和用量
                            # 支持更复杂的用量格式，如：12克、9g、适量等
                            dosage_pattern = r'(.*?)(?:\s+|)($(?:\d+|\d+\.\d+|\d+-\d+)[克gG]|\d+-?\d*钱|适量|少许|少许许|各等分)'
                            match = re.match(dosage_pattern, part)
                            if match:
                                name = match.group(1).strip() if match.group(1) else ""
                                dosage = match.group(2).strip() if match.group(2) else ""
                                herbs.append({"name": name, "dosage": dosage})
                            else:
                                # 如果没有匹配到用量格式，整段作为名称
                                herbs.append({"name": part, "dosage": ""})
            else:
                # 按分隔符分割后处理
                for part in parts:
                    part = part.strip() if part else ""
                    if part:
                        # 尝试提取药材名称和用量
                        # 支持更复杂的用量格式，如：12克、9g、适量等
                        dosage_pattern = r'(.*?)(?:\s+|)($(?:\d+|\d+\.\d+|\d+-\d+)[克gG]|\d+-?\d*钱|适量|少许|少许许|各等分)'
                        match = re.match(dosage_pattern, part)
                        if match:
                            name = match.group(1).strip() if match.group(1) else ""
                            dosage = match.group(2).strip() if match.group(2) else ""
                            herbs.append({"name": name, "dosage": dosage})
                        else:
                            # 如果没有匹配到用量格式，整段作为名称
                            herbs.append({"name": part, "dosage": ""})
        return herbs

    def scrape_prescriptions_from_category(self, category_url: str, max_prescriptions: int = 100) -> List[Dict[str, Any]]:
        """
        从指定分类爬取药方信息
        
        Args:
            category_url (str): 分类页面URL
            max_prescriptions (int): 最大爬取药方数量
            
        Returns:
            List[Dict[str, Any]]: 药方信息列表
        """
        logger.info(f"开始爬取分类中的药方信息: {category_url}")
        
        prescriptions_data = []
        prescription_links = []
        
        # 处理分页情况，爬取多页数据
        page = 1
        max_pages = 10  # 最大爬取页数，防止无限循环
        
        while page <= max_pages and len(prescription_links) < max_prescriptions * 2:  # 爬取更多链接以确保数量
            # 构造分页URL
            if page == 1:
                current_url = category_url
            else:
                current_url = f"{category_url}list_25_{page}.html"
            
            logger.info(f"正在爬取分类页面: {current_url}")
            
            # 获取分类页面
            response = self._get_page(current_url)
            if not response:
                break
            
            # 解析分类页面，提取药方链接
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 查找所有药方链接
            # 查找所有<a>标签
            links = soup.find_all('a', href=True)
            page_links = []
            
            for link in links:
                href = link['href']
                # 检查链接是否指向药方详情页
                # 支持多种格式的URL：
                # 1. 数字ID格式：/fangji/nk/3304.html
                # 2. 拼音格式：/fangji/nk/ainaxiang.html
                # 3. 带目录的格式：/fangji/nk/jingfang/3304.html
                # 4. 完整URL格式：http://www.cnzyao.com/fangji/nk/3304.html
                if re.match(r'/fangji/nk/\d+\.html', href) or \
                   re.match(r'/fangji/nk/[a-zA-Z][a-zA-Z0-9]*\.html', href) or \
                   re.match(r'/fangji/nk/(?:[a-z]+/)+\d+\.html', href) or \
                   re.match(r'/fangji/nk/(?:[a-z]+/)?(?:\d+|[a-zA-Z][a-zA-Z0-9]*|[a-zA-Z0-9]+)\.html', href):
                    full_url = urljoin(self.base_url, href)
                    page_links.append(full_url)
                # 也检查完整URL格式
                elif re.match(r'http://www\.cnzyao\.com/fangji/nk/\d+\.html', href) or \
                     re.match(r'http://www\.cnzyao\.com/fangji/nk/[a-zA-Z][a-zA-Z0-9]*\.html', href) or \
                     re.match(r'http://www\.cnzyao\.com/fangji/nk/(?:[a-z]+/)+\d+\.html', href) or \
                     re.match(r'http://www\.cnzyao\.com/fangji/nk/(?:[a-z]+/)?(?:\d+|[a-zA-Z][a-zA-Z0-9]*|[a-zA-Z0-9]+)\.html', href):
                    page_links.append(href)
            
            # 如果当前页没有找到新的链接，说明已经到最后一页
            if not page_links:
                logger.info(f"分类页面 {current_url} 没有找到药方链接，停止爬取分页")
                break
            
            # 添加到总链接列表中
            prescription_links.extend(page_links)
            logger.info(f"在分类页面 {current_url} 中找到 {len(page_links)} 个药方链接")
            
            page += 1
            
            # 如果是第一页，检查是否有分页链接来判断是否需要继续爬取
            if page == 2:
                # 查找分页链接，判断是否还有下一页
                pagination = soup.find('div', class_='page') or soup.find('div', class_='pages') or soup.find('div', {'class': re.compile(r'.*page.*', re.I)})
                if not pagination:
                    logger.info("未找到分页控件，停止爬取分页")
                    break
        
        # 去重
        prescription_links = list(set(prescription_links))
        
        logger.info(f"在分类 {category_url} 中总共找到 {len(prescription_links)} 个药方链接")
        
        # 使用线程池并发爬取药方信息
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交任务
            future_to_url = {executor.submit(self.scrape_single_prescription, url): url for url in prescription_links[:max_prescriptions]}
            
            # 获取结果
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    prescription_data = future.result()
                    if prescription_data and prescription_data.get("name"):
                        prescriptions_data.append(prescription_data)
                        logger.info(f"已爬取药方: {prescription_data['name']}")
                    else:
                        logger.warning(f"爬取药方失败: {url}")
                except Exception as e:
                    logger.error(f"爬取药方时发生异常 {url}: {e}")
                
                # 显示进度
                if (len(prescriptions_data) > 0) and (len(prescriptions_data) % 10 == 0):
                    logger.info(f"已爬取 {len(prescriptions_data)} 个药方")
        
        # 截取到最大数量
        prescriptions_data = prescriptions_data[:max_prescriptions]
        logger.info(f"分类 {category_url} 中的药方信息爬取完成，共获得 {len(prescriptions_data)} 条数据")
        return prescriptions_data

    def scrape_all_prescriptions(self, max_prescriptions: int = 100) -> List[Dict[str, Any]]:
        """
        爬取所有药方信息
        
        Args:
            max_prescriptions (int): 最大爬取药方数量
            
        Returns:
            List[Dict[str, Any]]: 药方信息列表
        """
        logger.info("开始爬取所有药方信息...")
        
        prescriptions_data = []
        
        # 定义药方分类URL列表 - 修正为正确的URL
        category_urls = [
            "http://www.cnzyao.com/fangji/nk/",    # 内科方剂
            "http://www.cnzyao.com/fangji/wk/",    # 外科方剂
            "http://www.cnzyao.com/fangji/fk/",    # 妇科方剂
            "http://www.cnzyao.com/fangji/ek/",    # 儿科方剂
            "http://www.cnzyao.com/fangji/wgk/",   # 五官科方剂
            "http://www.cnzyao.com/fangji/pfk/"    # 皮肤病方剂
        ]
        
        # 计算每个分类应该爬取的最大数量
        max_per_category = max_prescriptions // len(category_urls) + 1
        
        # 遍历每个分类
        for category_url in category_urls:
            logger.info(f"正在爬取分类: {category_url}")
            
            # 爬取该分类下的药方
            category_prescriptions = self.scrape_prescriptions_from_category(
                category_url, max_prescriptions=max_per_category)
            
            prescriptions_data.extend(category_prescriptions)
            
            # 如果已达到最大数量，停止爬取
            if len(prescriptions_data) >= max_prescriptions:
                break
            
            # 分类之间随机等待较短时间（但仍要保持礼貌）
            wait_time = random.uniform(30, 60)  # 30秒到1分钟
            logger.info(f"等待 {wait_time:.1f} 秒后再开始爬取下一个分类...")
            remaining = wait_time
            while remaining > 0:
                sleep_time = min(30, remaining)
                time.sleep(sleep_time)
                remaining -= sleep_time
                logger.info(f"爬取分类等待剩余 {remaining:.1f} 秒")
        
        # 截取到最大数量
        prescriptions_data = prescriptions_data[:max_prescriptions]
        logger.info(f"药方信息爬取完成，共获得 {len(prescriptions_data)} 条数据")
        return prescriptions_data

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
    # 代理IP列表（示例，实际使用时需要替换为有效的代理IP）
    proxies = [
        # {'http': 'http://proxy1.example.com:8080', 'https': 'https://proxy1.example.com:8080'},
        # {'http': 'http://proxy2.example.com:8080', 'https': 'https://proxy2.example.com:8080'}
    ]
    
    # 创建爬虫实例，使用更短的延迟范围以提高速度，并启用并发
    scraper = TCMPrescriptionScraper(delay_range=(0.5, 2), proxies=proxies, max_workers=5)

    # 爬取数据
    prescriptions_data = scraper.scrape_all_prescriptions(max_prescriptions=150)

    # 构建保存路径：当前脚本目录下的 data/raw 文件夹
    base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    output_folder = os.path.join(base_dir, "data", "raw")
    os.makedirs(output_folder, exist_ok=True)  # 自动创建目录

    # 文件名
    output_file = os.path.join(output_folder, "tcm_prescriptions_data_fixed.json")

    # 保存数据
    scraper.save_to_json(prescriptions_data, output_file)

if __name__ == "__main__":
    main()