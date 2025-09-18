#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aho-Corasick多模式匹配算法实现
用于中药术语的高效字符串匹配和实体识别
"""

import collections
from typing import List, Dict, Set, Tuple, Optional, Any


class ACAutomaton:
    """
    Aho-Corasick自动机实现
    用于多模式字符串匹配，在中药术语识别中具有重要作用
    """
    
    def __init__(self):
        """
        初始化AC自动机
        """
        self.trie = {}  # 字典树
        self.fail = {}  # 失败指针
        self.output = {}  # 输出函数
        self.patterns = []  # 模式串列表
        
    def add_pattern(self, pattern: str, pattern_id: Any = None) -> None:
        """
        添加模式串到自动机中
        
        Args:
            pattern (str): 模式串
            pattern_id (Any): 模式串的标识符，如果为None则使用pattern本身
        """
        if not pattern:
            return
            
        # 构建字典树
        node = self.trie
        for char in pattern:
            if char not in node:
                node[char] = {}
            node = node[char]
        
        # 标记单词结束
        end_key = "##END##"
        if end_key not in node:
            node[end_key] = []
        node[end_key].append(pattern_id if pattern_id is not None else pattern)
        
        # 保存模式串
        self.patterns.append(pattern)
    
    def build_fail_function(self) -> None:
        """
        构建失败指针函数
        """
        # 初始化失败指针
        self.fail = {id(node): node for node in self._get_all_nodes()}
        self.fail[id(self.trie)] = self.trie
        
        # 使用BFS构建失败指针
        queue = collections.deque()
        
        # 初始化第一层节点的失败指针
        for char, child in self.trie.items():
            if char != "##END##":
                self.fail[id(child)] = self.trie
                queue.append(child)
        
        # BFS构建失败指针
        while queue:
            node = queue.popleft()
            
            for char, child in node.items():
                if char != "##END##":
                    # 查找失败指针
                    fail_node = self.fail[id(node)]
                    while fail_node != self.trie and char not in fail_node:
                        fail_node = self.fail[id(fail_node)]
                    
                    if char in fail_node and fail_node[char] != child:
                        self.fail[id(child)] = fail_node[char]
                    else:
                        self.fail[id(child)] = self.trie
                    
                    # 合并输出
                    if "##END##" in self.fail[id(child)]:
                        if "##END##" not in child:
                            child["##END##"] = []
                        child["##END##"].extend(self.fail[id(child)]["##END##"])
                    
                    queue.append(child)
    
    def _get_all_nodes(self) -> List[Dict]:
        """
        获取字典树中的所有节点
        
        Returns:
            List[Dict]: 所有节点列表
        """
        nodes = []
        stack = [self.trie]
        
        while stack:
            node = stack.pop()
            nodes.append(node)
            
            for char, child in node.items():
                if char != "##END##":
                    stack.append(child)
        
        return nodes
    
    def search(self, text: str) -> List[Tuple[int, int, Any]]:
        """
        在文本中搜索所有模式串
        
        Args:
            text (str): 待搜索的文本
            
        Returns:
            List[Tuple[int, int, Any]]: 匹配结果列表，每个元素为(起始位置, 结束位置, 模式串标识符)
        """
        if not self.fail:
            self.build_fail_function()
        
        result = []
        node = self.trie
        pos = 0
        
        while pos < len(text):
            char = text[pos]
            
            # 如果当前字符在当前节点的子节点中
            if char in node:
                node = node[char]
                
                # 如果到达单词结尾，添加匹配结果
                if "##END##" in node:
                    for pattern_id in node["##END##"]:
                        start_pos = pos - len(pattern_id) + 1
                        result.append((start_pos, pos + 1, pattern_id))
                
                pos += 1
            else:
                # 如果不在，转移到失败指针指向的节点
                if node == self.trie:
                    pos += 1
                else:
                    node = self.fail[id(node)]
        
        return result
    
    def search_first(self, text: str) -> Optional[Tuple[int, int, Any]]:
        """
        在文本中搜索第一个匹配的模式串
        
        Args:
            text (str): 待搜索的文本
            
        Returns:
            Optional[Tuple[int, int, Any]]: 第一个匹配结果，格式为(起始位置, 结束位置, 模式串标识符)
        """
        matches = self.search(text)
        return matches[0] if matches else None
    
    def search_all_patterns(self, text: str) -> Dict[str, List[Tuple[int, int]]]:
        """
        搜索文本中所有模式串的出现位置
        
        Args:
            text (str): 待搜索的文本
            
        Returns:
            Dict[str, List[Tuple[int, int]]]: 每个模式串及其在文本中的位置列表
        """
        matches = self.search(text)
        result = {}
        
        for start, end, pattern_id in matches:
            pattern = pattern_id if isinstance(pattern_id, str) else self.patterns[pattern_id]
            if pattern not in result:
                result[pattern] = []
            result[pattern].append((start, end))
        
        return result


class TCMEntityRecognizer:
    """
    中药实体识别器
    基于Aho-Corasick算法识别中药文本中的实体
    """
    
    def __init__(self):
        """
        初始化中药实体识别器
        """
        self.herb_automaton = ACAutomaton()  # 药材实体识别自动机
        self.symptom_automaton = ACAutomaton()  # 症状实体识别自动机
        self.disease_automaton = ACAutomaton()  # 疾病实体识别自动机
        self.effect_automaton = ACAutomaton()  # 功效实体识别自动机
        self.interaction_automaton = ACAutomaton()  # 相互作用实体识别自动机
        
        # 实体类型映射
        self.entity_types = {
            "herb": self.herb_automaton,
            "symptom": self.symptom_automaton,
            "disease": self.disease_automaton,
            "effect": self.effect_automaton,
            "interaction": self.interaction_automaton
        }
    
    def add_entities(self, entity_type: str, entities: List[str]) -> None:
        """
        添加实体到对应的自动机中
        
        Args:
            entity_type (str): 实体类型 ("herb", "symptom", "disease", "effect", "interaction")
            entities (List[str]): 实体列表
        """
        if entity_type not in self.entity_types:
            raise ValueError(f"不支持的实体类型: {entity_type}")
        
        automaton = self.entity_types[entity_type]
        for i, entity in enumerate(entities):
            automaton.add_pattern(entity, i)
    
    def build_automatons(self) -> None:
        """
        构建所有自动机的失败指针函数
        """
        for automaton in self.entity_types.values():
            automaton.build_fail_function()
    
    def recognize_entities(self, text: str) -> Dict[str, List[Tuple[int, int, str]]]:
        """
        识别文本中的所有实体
        
        Args:
            text (str): 待识别的文本
            
        Returns:
            Dict[str, List[Tuple[int, int, str]]]: 识别到的实体，格式为{实体类型: [(起始位置, 结束位置, 实体名称)]}
        """
        result = {}
        
        for entity_type, automaton in self.entity_types.items():
            matches = automaton.search(text)
            if matches:
                result[entity_type] = matches
        
        return result
    
    def extract_herb_interactions(self, text: str) -> List[Dict[str, Any]]:
        """
        从文本中提取药材相互作用关系
        
        Args:
            text (str): 包含药材相互作用描述的文本
            
        Returns:
            List[Dict[str, Any]]: 相互作用关系列表
        """
        # 常见的相互作用关键词
        interaction_keywords = {
            "相使": "synergizes",
            "相恶": "counteracts",
            "相畏": "folks",
            "相反": "incompatible"
        }
        
        # 添加相互作用关键词到自动机
        for keyword in interaction_keywords:
            self.interaction_automaton.add_pattern(keyword, keyword)
        
        # 识别文本中的实体
        entities = self.recognize_entities(text)
        
        # 提取相互作用关系
        interactions = []
        if "interaction" in entities and "herb" in entities:
            interaction_matches = entities["interaction"]
            herb_matches = entities["herb"]
            
            # 简单的相互作用关系提取逻辑
            for inter_start, inter_end, inter_type in interaction_matches:
                # 查找附近的药材实体
                nearby_herbs = [
                    (start, end, herb) for start, end, herb in herb_matches
                    if abs(start - inter_end) < 20 or abs(end - inter_start) < 20
                ]
                
                if len(nearby_herbs) >= 2:
                    interactions.append({
                        "herbs": [herb for _, _, herb in nearby_herbs[:2]],  # 取前两个药材
                        "interaction_type": interaction_keywords.get(inter_type, "unknown"),
                        "description": text[max(0, inter_start-10):min(len(text), inter_end+10)]
                    })
        
        return interactions


def main():
    """
    主函数，演示Aho-Corasick算法的使用
    """
    # 创建AC自动机
    ac = ACAutomaton()
    
    # 添加模式串
    patterns = ["人参", "当归", "甘草", "黄芪", "枸杞"]
    for pattern in patterns:
        ac.add_pattern(pattern)
    
    # 构建失败指针
    ac.build_fail_function()
    
    # 测试文本
    text = "人参和当归是常用的补气药材，甘草可以调和诸药，黄芪和枸杞具有很好的滋补作用。"
    
    # 搜索匹配
    matches = ac.search(text)
    print("匹配结果:")
    for start, end, pattern in matches:
        print(f"  位置 {start}-{end}: '{pattern}'")
    
    # 创建中药实体识别器
    recognizer = TCMEntityRecognizer()
    
    # 添加药材实体
    herbs = ["人参", "当归", "甘草", "黄芪", "枸杞", "白术", "茯苓"]
    recognizer.add_entities("herb", herbs)
    
    # 添加相互作用关键词
    interactions = ["相使", "相恶", "相畏", "相反"]
    recognizer.add_entities("interaction", interactions)
    
    # 构建自动机
    recognizer.build_automatons()
    
    # 识别实体
    entities = recognizer.recognize_entities(text)
    print("\n实体识别结果:")
    for entity_type, matches in entities.items():
        print(f"  {entity_type}:")
        for start, end, entity_id in matches:
            entity_name = herbs[entity_id] if entity_type == "herb" else interactions[entity_id]
            print(f"    位置 {start}-{end}: '{entity_name}'")
    
    # 提取相互作用关系
    interactions = recognizer.extract_herb_interactions(text)
    print("\n相互作用关系:")
    for interaction in interactions:
        print(f"  {interaction['herbs'][0]} 和 {interaction['herbs'][1]}: {interaction['interaction_type']}")


if __name__ == "__main__":
    main()