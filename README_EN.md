# YaYi

<div align="center">
<img src="./assets/yayi_dark_small.png" alt="YaYi" style="width: 30%; display: block; margin: auto;">
<br>

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-brightgreen.svg)](./LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC_BY_NC_4.0-red.svg)](./LICENSE_DATA)
[![Model License](https://img.shields.io/badge/Model%20License-YaYi-blue.svg)](./LICENSE_MODEL)

[[📖README](./README.md)] 
[[🤗HF Repo](https://huggingface.co/wenge-research)]
[[🔗WEB](https://yayi.wenge.com)]

English | [中文](./README.md)

</div>

## Introduction

[YaYi](https://www.wenge.com/yayi/index.html) was fine-tuned on millions of artificially constructed high-quality domain data. This training data covers five key domains: media publicity, public opinion analysis, public safety, financial risk control, and urban governance, encompassing over a hundred natural language instruction tasks. Throughout the iterative development process of the YaYi, starting from pre-training initialization weights and progressing to domain-specific model, we have steadily enhanced its foundational Chinese language capabilities and domain analysis capabilities. We've also introduced multi-turn conversation enhancements and integrated various plug-in capabilities. Furthermore, through continuous manual feedback and optimization from hundreds of users during the internal testing phase, we've meticulously refined the model's performance and security.

By open-sourcing the YaYi model, we will contribute our own efforts to the development of the Chinese pre-trained large language model open-source community. Through this open-source initiative, we seek to collaborate with every partner to build the YaYi model ecosystem together.

*News: 🔥 YaYi has open sourced the Chinese optimization model version based on LLaMA 2 to explore the latest practices suitable for Chinese multi-domain tasks.*


## Model download

|  Model  | 🤗HF Model Name |  Download Links  |
| --------- | ---------    | --------- |
|  YaYi-7B  | wenge-research/yayi-7b  | [Download](https://huggingface.co/wenge-research/yayi-7b)  |
| YaYi-7B-Llama2 | wenge-research/yayi-7b-llama2 | [Download](https://huggingface.co/wenge-research/yayi-7b-llama2) |
| YaYi-13B-Llama2 | wenge-research/yayi-13b-llama2 | [Download](https://huggingface.co/wenge-research/yayi-13b-llama2) |



## Run

### Setup
1. Download this repository to your local/remote server.

```bash
git clone https://github.com/wenge-research/YaYi.git
cd YaYi
```

2. Create conda environment

```bash
conda create --name yayi python=3.8
conda activate yayi
```

3. Install requirements

```bash
pip install -r requirements.txt
```
The `torch` and `transformers` versions are not recommended to be lower than the recommended version.

### Inference

Model weights (7b version) have been open-sourced in our [Huggingface model repository](https://huggingface.co/wenge-research). Feel free to download and use them. Below is a simple example code for invoking yayi-7b for downstream task inference. It can run on a single GPU like A100/A800/3090, and it occupies approximately 20GB of GPU memory when performing inference with FP16 precision:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch

yayi_7b_path = "wenge-research/yayi-7b"
tokenizer = AutoTokenizer.from_pretrained(yayi_7b_path)
model = AutoModelForCausalLM.from_pretrained(yayi_7b_path, device_map="auto", torch_dtype=torch.bfloat16)

prompt = "hello!"
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

Please note that a special token `<|End|>` was added as an end-of-sequence marker during model training. Therefore, in the `GenerationConfig` provided above, you should set `eos_token_id` to the token id corresponding to this end-of-sequence marker. The inference code for models fine-tuned based on LlaMA2 instructions may vary slightly; for specific details, please refer to the corresponding version in our [Huggingface model repository](https://huggingface.co/wenge-research).

### Model fine-tuned

This project utilizes the `deepspeed` framework for model training. After setting up the environment, you can execute the corresponding scripts to commence training. It supports full-parameter fine-tuning on instruction data, LoRA fine-tuning on instruction data, full-parameter fine-tuning on multi-turn dialogue data, and LoRA fine-tuning on multi-turn dialogue data.

#### 1. Full-parameter fine-tuning on instruction data.

- **Data format**: Refer to [`data/yayi_train_example.json`](data/yayi_train_example.json), which follows the jsonline data format from the Alpaca project, with one JSON data entry per line. Each entry consists of `"instruction"`、`"input"`、`"output"`. `"instruction"` and `"input"` represent the instruction input, while `"output"` represents the output answer. 
- **Instructions**: Running the following command will initiate full-parameter fine-tuning of the YaYi model. This command supports training on a single machine with multiple GPUs. If you need to configure multi-machine multi-GPU training, please refer to the official deepspeed documentation. It is recommended to use hardware configurations with 4 or more A100 GPUs (80GB each) or higher.

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
        --lr 5e-6 \
        --seed 515
    ```

#### 2. LoRA fine-tuning on instruction data

- **Data format**: Same as above, refer to [`data/yayi_train_example.json`](data/yayi_train_example.json).
- **Instructions**: LoRA is an efficient low-resource fine-tuning method that can train models with hundreds of billions of parameters on a single GPU. This project primarily implements LoRA fine-tuning using [`peft`](https://huggingface.co/docs/peft/index). You can start LoRA fine-tuning of the YaYi model by running the following command. It is possible to complete fine-tuning using a single A100 (80GB) GPU, and you can adjust the learning rate to a higher value. The `--lora-dim`  sets the rank of the update matrix, where a larger value results in more parameters being trained. And the `--lora-module-name` specifies the module for the LoRA update matrix and can be changed based on the model type.

    ```
    deepspeed --num_gpus=1 \
        --module training.trainer_lora \
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
        --lr 5e-4 \
        --seed 515 \
        --lora-dim 16 \
        --lora-module-name query_key_value
    ```

#### 3. Full-parameter fine-tuning on multi-turn dialogue data

- **Data format**: Refer to [`data/yayi_train_example_multi_rounds.json`](data/yayi_train_example_multi_rounds.json), which is a standard JSON file. Each data entry consists of `"system"` and `"conversations"`. `"system"` contains global role-setting information and can be an empty string. `"conversations"` contains multi-turn dialogue content conducted alternately between human and YaYi roles.
- **Instructions**: Running the following command will initiate full-parameter fine-tuning of the YaYi model. For multi-turn dialogue data, it calculates the loss only for model-generated responses. This command supports training on a single machine with multiple GPUs. If you need to configure multi-machine multi-GPU training, please refer to the official deepspeed documentation. It is recommended to use hardware configurations with 4 or more A100 GPUs (80GB each) or higher.

    ```
    deepspeed --num_gpus=8 \
        --module training.trainer_multi_rounds \
        --data-path ./data/yayi_train_example_multi_rounds.json \
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

#### 4. LoRA fine-tuning on multi-turn dialogue data

- **Data format**: Same as above, refer to [`data/yayi_train_example_multi_rounds.json`](data/yayi_train_example_multi_rounds.json).
- **Instructions**: Refer to the data loading method for full-parameter fine-tuning on multi-turn dialogue data, as well as the LoRA fine-tuning method for instruction data.


## Training data

The YaYi model was trained on a high-quality domain-specific instruction fine-tuning dataset, which consists of millions of instances provided by Wenge Research. For this open-source release, we have made available a training dataset containing 50,000 samples, which can be downloaded from our [Huggingface data repository](https://huggingface.co/wenge-research). This dataset primarily covers several domains, including finance, security, public opinion analysis, media, and more. We have added discrete prompt prefixes to most of the instruction data to differentiate between various domain-specific data. Additionally, the training data includes some security-enhanced data, plugin capability data, and multi-turn dialogue data.

## Related agreements

### Limitations
The SFT model trained based on the current data and base model still exhibits the following issues in terms of performance:

1. It may generate factually incorrect responses for factual instructions.
2. It struggles to effectively identify harmful instructions, potentially leading to harmful content generation.
3. Its capabilities in scenarios involving logical reasoning, code generation, scientific computation, and similar tasks still require improvement.

### Disclaimer

Due to the limitations of the model mentioned above, we request that developers use the code, data, models, and any derivatives generated from this project solely for research purposes and refrain from using them for commercial or any other potentially harmful purposes to society. Please exercise caution in evaluating and utilizing content generated by the YaYi model, and do not propagate harmful content on the internet. Any adverse consequences resulting from such actions are the responsibility of the disseminator.

This project is intended for research purposes only, and the project developers bear no responsibility for any harm or losses incurred due to the use of this project, including but not limited to data, models, code, etc. For more details, please refer to the [Disclaimer](DISCLAIMER).

### License

The code in this project is open-source under the [Apache-2.0](LICENSE) license, the data follows the [CC BY-NC 4.0](LICENSE_DATA) license, and the usage of YaYi series model weights must adhere to the [Model License](LICENSE_MODEL).

## Update log
- [2023/08/09] Updated LoRA fine-tuning code and multi-turn dialogue format data training code.
- [2023/07/22] Updated YaYi-7B-Llama2 and YaYi-13B-Llama2 model weights with enhanced Chinese domain knowledge.
- [2023/07/14] Enhanced model security and anti-denial capabilities, introducing model int8 quantization.
- [2023/06/29] Improved and optimized multi-turn dialogue capabilities in both Chinese and English.
- [2023/06/03] Officially released and open-sourced the 7B version of the YaYi model.

## Acknowledgements
- In this project, we used model weights from BigScience's [bloomz-7b1-mt](https://huggingface.co/bigscience/bloomz-7b1-mt) and Meta's [Llama 2](https://huggingface.co/meta-llama) series as initialization weights, along with vocabulary expansion.
- The training code in this project was inspired by Databricks' [dolly](https://github.com/databrickslabs/dolly) project and Huggingface's [transformers](https://github.com/huggingface/transformers) library.
- Distributed training in this project utilized Microsoft's [DeepSpeed](https://github.com/microsoft/deepspeed) distributed training tool and configuration files from Huggingface transformers' [ZeRO stage 2](https://huggingface.co/docs/transformers/main_classes/deepspeed#zero2-config).


## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=wenge-research/YaYi&type=Date)](https://star-history.com/#wenge-research/YaYi&Date)