#!/usr/bin/env python3
"""
合并 results/one_round 目录中带有 part1 和 part2 且前缀后缀相同的 JSON 文件
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


def find_part_files(directory: Path) -> Dict[str, Dict[str, Path]]:
    """
    查找所有 part1 和 part2 文件，按前缀和后缀分组
    
    Returns:
        Dict[base_name, Dict['part1'|'part2', Path]]
    """
    part_files = defaultdict(dict)
    
    # 匹配文件名格式: {prefix}_part_{1|2}_{suffix}.json
    pattern = re.compile(r'^(.+)_part_([12])_(.+\.json)$')
    
    for file_path in directory.glob('*_part_*.json'):
        # 跳过 metrics 目录中的文件
        if 'metrics' in str(file_path):
            continue
            
        match = pattern.match(file_path.name)
        if match:
            prefix = match.group(1)
            part_num = match.group(2)
            suffix = match.group(3)
            
            # 生成基础名称（去掉 part1/part2）
            base_name = f"{prefix}_{suffix}"
            
            if part_num == '1':
                part_files[base_name]['part1'] = file_path
            elif part_num == '2':
                part_files[base_name]['part2'] = file_path
    
    return part_files


def merge_json_files(part1_path: Path, part2_path: Path, output_path: Path) -> None:
    """
    合并两个 JSON 文件
    
    Args:
        part1_path: part1 文件路径
        part2_path: part2 文件路径
        output_path: 输出文件路径
    """
    print(f"合并文件:")
    print(f"  Part1: {part1_path.name}")
    print(f"  Part2: {part2_path.name}")
    
    # 读取 part1 文件
    with open(part1_path, 'r', encoding='utf-8') as f:
        data1 = json.load(f)
    
    # 读取 part2 文件
    with open(part2_path, 'r', encoding='utf-8') as f:
        data2 = json.load(f)
    
    # 合并结果
    merged_data = {
        "dataset_path": data1.get("dataset_path", "").replace("_part_1_", "_merged_").replace("_part_2_", "_merged_"),
        "results": data1.get("results", []) + data2.get("results", [])
    }
    
    # 如果原数据有其他字段，也保留
    for key in data1:
        if key not in ["dataset_path", "results"]:
            merged_data[key] = data1[key]
    
    # 保存合并后的文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    print(f"  ✓ 已保存: {output_path.name}")
    print(f"  ✓ 合并了 {len(data1.get('results', []))} + {len(data2.get('results', []))} = {len(merged_data['results'])} 条结果\n")


def main():
    # 设置目录路径
    script_dir = Path(__file__).parent
    results_dir = script_dir
    
    if not results_dir.exists():
        print(f"错误: 目录不存在: {results_dir}")
        return
    
    print(f"扫描目录: {results_dir}\n")
    
    # 查找所有 part 文件
    part_files = find_part_files(results_dir)
    
    if not part_files:
        print("未找到需要合并的 part1 和 part2 文件对")
        return
    
    # 统计信息
    matched_pairs = []
    missing_pairs = []
    
    for base_name, files in part_files.items():
        if 'part1' in files and 'part2' in files:
            matched_pairs.append((files['part1'], files['part2'], base_name))
        else:
            missing = []
            if 'part1' not in files:
                missing.append('part1')
            if 'part2' not in files:
                missing.append('part2')
            missing_pairs.append((base_name, missing))
    
    print(f"找到 {len(matched_pairs)} 对匹配的文件\n")
    
    if missing_pairs:
        print(f"警告: 以下文件缺少对应的 part:")
        for base_name, missing in missing_pairs:
            print(f"  {base_name}: 缺少 {', '.join(missing)}")
        print()
    
    # 合并文件
    for part1_path, part2_path, base_name in matched_pairs:
        # 生成输出文件名（去掉 part1/part2，或替换为 merged）
        output_name = base_name.replace('_part_1_', '_merged_').replace('_part_2_', '_merged_')
        # 如果还有 part 标记，直接去掉
        output_name = re.sub(r'_part_[12]_', '_merged_', output_name)
        output_path = results_dir / output_name
        
        merge_json_files(part1_path, part2_path, output_path)
    
    print(f"完成! 共合并了 {len(matched_pairs)} 对文件")


if __name__ == "__main__":
    main()