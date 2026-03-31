#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

# 添加项目路径到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tcm_prescription_scraper_fixed import TCMPrescriptionScraper

def main():
    # 创建爬虫实例，使用更短的延迟范围以提高速度，并启用并发
    scraper = TCMPrescriptionScraper(delay_range=(0.5, 2), max_workers=5)

    # 专门爬取内科方剂分类
    print("开始爬取内科方剂...")
    prescriptions_data = scraper.scrape_prescriptions_from_category(
        "http://www.cnzyao.com/fangji/nk/", max_prescriptions=150)

    print(f"内科方剂爬取完成，共获得 {len(prescriptions_data)} 条数据")

    # 构建保存路径：当前脚本目录下的 data/raw 文件夹
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(base_dir, "data", "raw")
    os.makedirs(output_folder, exist_ok=True)  # 自动创建目录

    # 文件名
    output_file = os.path.join(output_folder, "tcm_prescriptions_nk_final.json")

    # 保存数据
    scraper.save_to_json(prescriptions_data, output_file)

    print(f"数据已保存到: {output_file}")

    # 显示前几个药方的名称
    print("\n前10个药方名称:")
    for i, prescription in enumerate(prescriptions_data[:10]):
        print(f"{i+1}. {prescription.get('name', '未知名称')}")

    # 检查是否达到了预期的数量
    if len(prescriptions_data) >= 130:
        print(f"\n✅ 成功！爬取到 {len(prescriptions_data)} 条药方，达到了预期的136条左右。")
    else:
        print(f"\n⚠️  注意：只爬取到 {len(prescriptions_data)} 条药方，少于预期的136条。")

if __name__ == "__main__":
    main()