#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : data_convert.py
@Author  : wenge-research
'''

import os, json
from argparse import ArgumentParser
from tqdm import tqdm

def convert_inst_to_chat(path):
    """
    Usage: 将 `指令数据格式` 转换为 `对话数据格式`.
    """
    results = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            try:
                inst_data = json.loads(line)
                results.append({
                    "system": "", 
                    "conversations": [
                        {"from": "human","value": str(inst_data['instruction'].strip()+"\n"+inst_data['input'].strip()).strip()},
                        {"from": "yayi","value": inst_data["output"].strip()}]})
            except:
                continue

        print(f"Data num: {len(results)}")    
        print(f"Example:\n{json.dumps(results[0], ensure_ascii=False, indent=2) if results else 0}")
    
    save_path = path.split('.')[0]+'_chat.json'
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"Save to {save_path}")


def merge_multi_chat_files(path):
    """
    Usage: 合并多个 `对话数据格式` 文件.
    """
    results = []
    for filepath in path.split(","):
        print(f"Loading {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            results.extend(json.load(f))

    save_path = path.split('.')[0]+'_merged.json'
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"Save to {save_path}")


if __name__ == "__main__":

    parser = ArgumentParser(description="Convert inst to chat")
    parser.add_argument("--mode", type=str, default="inst2chat", help="`inst2chat`: instruction to chat; `merge`: merge multi chat files.")
    parser.add_argument("--path", type=str, help="input file file, split with `,`")
    args = parser.parse_args()

    if args.mode=="inst2chat" and (args.path is None or not os.path.exists(args.path)):
        print("*** File path not exists. ***")
        exit(0)

    if args.mode == "inst2chat":
        convert_inst_to_chat(args.path)
    elif args.mode == "merge":
        merge_multi_chat_files(args.path)
    else:
        print("*** Unknown mode. ***")
