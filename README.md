# 雅意大模型

<div align="center">
<img src="./assets/yayi_dark_small.png" alt="YaYi" style="width: 30%; display: block; margin: auto;">
<br>

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-brightgreen.svg)](./LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC_BY_NC_4.0-red.svg)](./LICENSE_DATA)
[![Model License](https://img.shields.io/badge/Model%20License-YaYi_7B-blue.svg)](./LICENSE_MODEL)

[[📖README](./README.md)] 
[[🤗HF Repo](https://huggingface.co/wenge-research)]
[[🔗网页端](https://yayi.wenge.com)]

</div>

## 介绍

[雅意大模型](https://www.wenge.com/yayi/index.html)在百万级人工构造的高质量领域数据上进行指令微调得到，训练数据覆盖媒体宣传、舆情分析、公共安全、金融风控、城市治理等五大领域，上百种自然语言指令任务。雅意大模型从预训练初始化权重到领域模型的迭代过程中，我们逐步增强了它的中文基础能力和领域分析能力，并增加了多轮对话和部分插件能力。同时，经过数百名用户内测过程中持续不断的人工反馈优化，我们进一步提升了模型性能和安全性。

通过雅意大模型的开源为促进中文预训练大模型开源社区的发展，贡献自己的一份力量，通过开源，与每一位合作伙伴共建雅意大模型生态。

*News: 🔥 雅意大模型已开源基于 LLaMA 2 的中文优化模型版本，探索适用于中文多领域任务的最新实践。*


## 模型地址

|  模型名称  | 🤗HF模型标识 |  下载地址  |
| --------- | ---------    | --------- |
|  YaYi-7B  | wenge-research/yayi-7b  | [模型下载](https://huggingface.co/wenge-research/yayi-7b)  |
| YaYi-7B-Llama2 | wenge-research/yayi-7b-llama2 | [模型下载](https://huggingface.co/wenge-research/yayi-7b-llama2) |
| YaYi-13B-Llama2 | wenge-research/yayi-13b-llama2 | [模型下载](https://huggingface.co/wenge-research/yayi-13b-llama2) |



## 运行方式

### 环境安装
1. 下载本仓库内容至本地/远程服务器

```bash
git clone https://github.com/wenge-research/YaYi.git
cd YaYi
```

2. 创建conda环境

```bash
conda create --name yayi python=3.8
conda activate yayi
```

3. 安装依赖

```bash
pip install -r requirements.txt
```
其中 `torch` 和 `transformers` 版本不建议低于推荐版本。

### 模型推理

模型权重（7b版本）已在我们的 [Huggingface 模型仓库](https://huggingface.co/wenge-research) 开源，欢迎下载使用。以下是一个简单调用 `yayi-7b` 进行下游任务推理的示例代码，可在单张 A100/A800/3090 等GPU运行，使用FP16精度推理时约占用 20GB 显存：

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch

yayi_7b_path = "wenge-research/yayi-7b"
tokenizer = AutoTokenizer.from_pretrained(yayi_7b_path)
model = AutoModelForCausalLM.from_pretrained(yayi_7b_path, device_map="auto", torch_dtype=torch.bfloat16)

prompt = "你好"
formatted_prompt = f"<|System|>:\nA chat between a human and an AI assistant named YaYi.\nYaYi is a helpful and harmless language model developed by Beijing Wenge Technology Co.,Ltd.\n\n<|Human|>:\n{prompt}\n\n<|YaYi|>:"
inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

eos_token_id = tokenizer("<|End|>").input_ids[0]
generation_config = GenerationConfig(
    eos_token_id=eos_token_id,
    pad_token_id=eos_token_id,
    do_sample=True,
    max_new_tokens=100,
    temperature=0.3,
    repetition_penalty=1.1,
    no_repeat_ngram_size=0
)
response = model.generate(**inputs, generation_config=generation_config)
print(tokenizer.decode(response[0]))
```

注意，模型训练时添加了 special token `<|End|>` 作为结束符，因此上述代码 `GenerationConfig` 里将 `eos_token_id` 设置为该结束符对应的 token id。

### 模型微调

本项目基于 `deepspeed` 框架进行模型训练，配置完环境后执行以下命令行即可开始模型微调（单机多卡）。

```
deepspeed --num_gpus=8 \
    --module training.trainer \
    --data-path ./data/yayi_train_example.json \
    --input-model ./checkpoints/yayi-7b \
    --deepspeed ./config/deepspeed_zero2_bf16.json \
    --epochs 2 \
    --local-output-dir ./checkpoints \
    --per-device-train-batch-size 8 \
    --per-device-eval-batch-size 8 \
    --logging-steps 1 \
    --save-steps 100 \
    --save-total-limit 10 \
    --eval-steps 100 \
    --warmup-steps 100 \
    --test-size 400 \
    --lr 5e-7 \
    --seed 515
```


## 训练数据

雅意大模型基于中科闻歌百万级高质量领域指令微调数据集训练而来，我们本次开源 5w 条训练数据集，可在我们的 [Huggingface 数据仓库](https://huggingface.co/wenge-research) 下载。数据集主要涵盖了金融、安全、舆情、媒体等几大领域，我们为各领域任务大部分指令数据添加了离散 prompt 前缀，以区分各领域数据。


## 相关协议

### 局限性
基于当前数据和基础模型训练得到的SFT模型，在效果上仍存在以下问题：

1. 在涉及事实性的指令上可能会产生违背事实的错误回答。
2. 对于具备危害性的指令无法很好的鉴别，可能会产生危害性言论。
3. 在一些涉及推理、代码、多轮对话等场景下模型的能力仍有待提高。

### 免责声明

基于以上模型局限性，我们要求开发者仅将我们开源的代码、数据、模型及后续用此项目生成的衍生物用于研究目的，不得用于商业用途，以及其他会对社会带来危害的用途。请谨慎鉴别和使用雅意大模型生成的内容，请勿将生成的有害内容传播至互联网。若产生不良后果，由传播者自负。

本项目仅可应用于研究目的，项目开发者不承担任何因使用本项目（包含但不限于数据、模型、代码等）导致的危害或损失。详细请参考[免责声明](DISCLAIMER)。

### 开源协议

本项目中的代码依照 [Apache-2.0](LICENSE) 协议开源，数据采用 [CC BY-NC 4.0](LICENSE_DATA) 协议，YaYi 系列模型权重的使用则需要遵循 [Model License](LICENSE_MODEL)。

## 更新日志
- [2023/07/22] 更新中文领域知识增强的 YaYi-6B-Llama2 和 YaYi-13B-Llama2 模型权重。
- [2023/07/14] 升级模型安全性和拒识能力，新增模型 int8 量化。
- [2023/06/29] 升级和优化中英文多轮对话能力。
- [2023/06/03] 雅意大模型正式对外发布并开源 7B 版本模型权重。

## 致谢
- 本项目分别使用了 BigScience  [bloomz-7b1-mt](https://huggingface.co/bigscience/bloomz-7b1-mt) 以及 Meta [Llama 2](https://huggingface.co/meta-llama) 系列的模型权重作为初始化权重，并进行词表扩展；
- 本项目训练代码参考了 Databricks 的 [dolly](https://github.com/databrickslabs/dolly) 项目及 Huggingface [transformers](https://github.com/huggingface/transformers) 库；
- 本项目分布式训练使用了 Microsoft 的 [DeepSpeed](https://github.com/microsoft/deepspeed) 分布式训练工具及 Huggingface transformers 文档中的 [ZeRO stage 2](https://huggingface.co/docs/transformers/main_classes/deepspeed#zero2-config) 配置文件；


## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=wenge-research/YaYi&type=Date)](https://star-history.com/#wenge-research/YaYi&Date)