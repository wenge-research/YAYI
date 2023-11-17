# 相关脚本

| 脚本名称 | 用途 | 使用说明 |
| --- | --- | --- |
| [data_convert.py](data_convert.py) | 数据格式转换与合并 | `python data_convert.py --mode inst2chat --path {YOUR_JSON_PATH}` 用于将指令数据格式转换为多轮对话数据格式；</br> `python data_convert.py --mode chat2inst --path {YOUR_JSON_PATH}` 用于将对话格式数据(仅单轮)转换为指令数据格式；</br> `python data_convert.py --mode merge --path {YOUR_JSON_PATH_1,YOUR_JSON_PATH_2,...}` 用于多个数据文件的合并。 |
