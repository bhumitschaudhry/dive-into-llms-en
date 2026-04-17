# Fine-tuning and Deployment of Pre-trained Language Models

> Overview: This section introduces fine-tuning for pre-trained models
Want to improve pre-trained model performance on specific tasks? Let's select an appropriate pre-trained model, fine-tune it for your target task, and deploy the fine-tuned model as an accessible Demo!

## Tutorial Objectives:
1. Become familiar with the Transformers library
2. Master fine-tuning and inference for pre-trained models (decoupled customizable version & default integrated version)
3. Master Demo deployment using Gradio Spaces
4. Understand model selection criteria and use cases for different pre-trained models

## Tutorial Contents:

### 1. Preparation:

#### 1.1 Introduction: Transformers Library
https://github.com/huggingface/transformers

> 🤗 Transformers provides APIs and tools to easily download and train state-of-the-art pre-trained models. Using pre-trained models reduces computational costs, carbon emissions, and saves the time and resources required for training from scratch. These models support common tasks across different modalities:
📝 Natural Language Processing: text classification, named entity recognition, question answering, language modeling, summarization, translation, multiple choice, and text generation.
🖼️ Computer Vision: image classification, object detection, and semantic segmentation.
🗣️ Audio: automatic speech recognition and audio classification.
🐙 Multimodal: table question answering, optical character recognition, information extraction from scanned documents, video classification, and visual question answering.

Official Documentation: https://huggingface.co/docs/transformers/main/en/index

![huggingface](./assets/huggingface.PNG)

#### 1.2 Environment Setup: Text Classification Example (e.g., Fake News Detection)
1. Navigate to the text classification examples directory, review the readme for key parameters, and download `requirements.txt` and `run_classification.py`

https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification

2. Install environment:
   - 1. Create new conda environment: `conda create -n llm python=3.9`
   - 2. Activate virtual environment: `conda activate llm`
   - 3. Install transformers: `pip install transformers`
   - 4. Remove torch auto-install from requirements.txt, then run: `pip install -r requirements.txt`

> For slow download speeds, use domestic mirror: `pip [Packages] -i https://pypi.tuna.tsinghua.edu.cn/simple`

> Note: Installing PyTorch via domestic pip mirrors will default to CPU version, which cannot run on GPU:
   - 5. Install PyTorch via conda: `conda install pytorch`

> For slow conda downloads, configure conda mirrors following this guide: https://blog.csdn.net/weixin_42797483/article/details/132048218

3. Prepare Dataset: We use the Kaggle Fake Tweets dataset as an example: https://www.kaggle.com/c/nlp-getting-started/data

#### 1.3 Prepared Project Packages (Demo Code + Data)
(1) Decoupled Customizable Version
Key modules are separated for easy understanding. Allows custom data loading, model architecture, evaluation metrics, etc.

[Download TextClassificationCustom](https://pan.quark.cn/s/00dae5c2b128)

(2) Default Integrated Version
More comprehensive and complex codebase, intended for direct hyperparameter usage with moderate development threshold.

[Download TextClassification](https://pan.quark.cn/s/9d0510f1c98d)


### 2. Custom Development with Decoupled Version (Minimum Viable Product)
Three main files:
- `main.py`: Main execution program
- `utils_data.py`: Data loading and processing
- `modeling_bert.py`: Model architecture implementation

![project structure](./assets/0.png)

#### 2.1 Key Module Overview
1. Data Loading & Processing (`utils_data.py`)
![utils_data.py](./assets/1.png)

2. Model Loading (`modeling_bert.py`)

![modeling_bert_1.py](./assets/2.png)

![modeling_bert_2.py](./assets/3.png)


3. Training / Validation / Prediction (`main.py`)

![main.py](./assets/4.png)

#### 2.2 Run Complete Training Pipeline
```shell
python main.py
```


### 3. Fine-tuning with Integrated Version (Optional, using `run_classification.py`)

#### 3.1 Key Module Overview:
1. Load Data (csv or json format)
![load data](./assets/5.png)

2. Preprocess Data
![process data](./assets/6.png)

3. Load Model
![load model](./assets/7.png)

4. Training / Validation / Prediction
![train dev predict](./assets/8.png)

#### 3.2 Train Model
Run the following script to perform training, validation on development set, and prediction on test set simultaneously:

```shell
python run_classification.py \
    --model_name_or_path  bert-base-uncased \
    --train_file data/train.csv \
    --validation_file data/val.csv \
    --test_file data/test.csv \
    --shuffle_train_dataset \
    --metric_name accuracy \
    --text_column_name "text" \
    --text_column_delimiter "\n" \
    --label_column_name "target" \
    --do_train \
    --do_eval \
    --do_predict \
    --max_seq_length 512 \
    --per_device_train_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --output_dir experiments/
```

Common errors and fixes:

1. **"Network is unreachable" during model download**: Manually download the model locally from https://huggingface.co/google-bert/bert-base-uncased
2. **Process hangs after data loading**: This occurs when the `evaluate` library fails to load metrics due to network issues. Press CTRL+C to terminate.

![bug](./assets/9.png)

Solution: Download the full evaluate package from https://github.com/huggingface/evaluate/tree/main and specify the local metric path:

```shell
python run_classification.py \
    --model_name_or_path  bert-base-uncased \
    --train_file data/train.csv \
    --validation_file data/val.csv \
    --test_file data/test.csv \
    --shuffle_train_dataset \
    --metric_name evaluate/metrics/accuracy/accuracy.py \
    --text_column_name "text" \
    --text_column_delimiter "\n" \
    --label_column_name "target" \
    --do_train \
    --do_eval \
    --do_predict \
    --max_seq_length 512 \
    --per_device_train_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --output_dir experiments/
```

![reference result](./assets/10.png)


### 4. Model Deployment
After training is complete, we can host interactive demos on Gradio Spaces.

#### 4.1 Gradio Spaces Documentation
https://huggingface.co/docs/hub/en/spaces-sdks-gradio

#### 4.2 Create New Space
1. Create Space: https://huggingface.co/new-space?sdk=gradio
2. Note: VPN may be required for access in some regions

![Gradio Spaces](./assets/gradio.png)

#### 4.3 Inference Code Implementation
See `app.py` in the project package for full implementation.

![app.py](./assets/11.png)

#### 4.4 Upload Files to Gradio Spaces
1. Requirements file (`requirements.txt`)
```
transformers==4.30.2
torch==2.0.0
```

2. Repository file structure

![file overview](./assets/12.png)


3. Demo Example
Successful deployment reference: https://huggingface.co/spaces/cooelf/text-classification
Full source code is available in the "Files" tab at top right.

![Files](./assets/13.png)


4. Bonus: Browse trending demos on the Spaces platform, search and experiment with interesting LLMs and demos.
![Spaces](./assets/14.png)


### 5. Advanced Exercises
1. Try other classification/regression tasks: sentiment analysis, news classification, vulnerability detection, etc.
2. Experiment with other model architectures: T5, ELECTRA, etc.

### 6. Additional Useful Model Examples
1. Question Answering: https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering
2. Text Summarization: https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization
3. Llama 2 Inference: https://huggingface.co/docs/transformers/en/model_doc/llama2
4. Lightweight Fine-tuning with LoRA: https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/5-Fine%20Tuning/LoRA_Tuning_PEFT.ipynb


### 7. Further Reading
1. Comprehensive 43-page survey on Large Language Models (authored by Word2Vec author)
   [Survey Article](https://mp.weixin.qq.com/s?__biz=Mzk0MTYzMzMxMA==&mid=2247484539&idx=1&sn=6ee42baab4ad792e74ac6a89d7dd87d9&chksm=c2ce3e0af5b9b71c578cb6836e09cce5f60b4a1cfffadc1c4c98211fc0af621bfaaaa8fe0aff&mpshare=1&scene=24&srcid=0331FO6iqVqs5Vx3iHrtqouQ&sharer_shareinfo=45c3fb78d9cfb9627908da44dd7f5559&sharer_shareinfo_first=45c3fb78d9cfb9627908da44dd7f5559&key=a2847c972f830c4143e00e0430f657d8ab5acae4ad24ca628213273021453a3a9e17984627e2ab8506d1dcf6e1fabd9ec3123a5f71d2a65295ad6f6f56da2224d4a6e3228237c237447bbf48a6eff1f53e1971503f26c3fb5b9d99d27eca7266529a5f86d75bada7ec10bf314687bfcbf9fa3b09b8e36e73f6d6a154a5ce5ff0&ascene=14&uin=NzE3NzkyOTQx&devicetype=iMac20%2C1+OSX+OSX+14.2.1+build(23C71)&version=13080610&nettype=WIFI&lang=zh_CN&countrycode=CN&fontScale=100&exportkey=n_ChQIAhIQM7tgcovlhXp1y5J%2BRMnhfhL3AQIE97dBBAEAAAAAAFsTBwTm4FMAAAAOpnltbLcz9gKNyK89dVj0MxFZswDc%2Fk646vJPW2S3JFh8H2JhyZXiPbIl%2Bh23CsewrmIoZ4j0D2zMNylC3pLbhu9FIARUKYn%2F0r0OIdHnxesVFpw1qLo6uBJ3zmbsKBVXM05%2B0MiOBIfShfpiIfraK7THzak94U0RdS1flC%2BIDjTb5SmZs9Z4XTyTsN0QXR6NWjAXFeuxnMB4SENMJ8dUR8n08b3DGKtz9rfefn0JRlsX4mGcLvOFsFwg4nk35nl4C3Wgcs4OYociKm5UHabdhWT7%2FWbNNToZLD39eD%2FL4Xo%3D&acctmode=0&pass_ticket=8xG4QKOnJeEFUtXkScVmMqb3omdWJbPuc%2BhN5%2BA7%2FOXj7ex757M2ABNO4GVVnv2T3VtqrC9gZH%2FrU09rrUxJcA%3D%3D&wx_header=0)

    Paper Title: *Large Language Models: A Survey*
    Paper Link: https://arxiv.org/pdf/2402.06196.pdf

2. [GPT, GPT-2, GPT-3 Paper Deep Dive](https://www.bilibili.com/video/BV1AF411b7xQ?t=0.0)

3. [InstructGPT Paper Deep Dive](https://www.bilibili.com/video/BV1hd4y187CR?t=0.4)
