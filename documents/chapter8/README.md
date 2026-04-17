# Hands-On Large Models: Multimodal Large Language Models

Guide: This section introduces common architectures and construction methods for multimodal large language models.

> The emergence of large language models has shown that higher-level intelligence is strongly expressed in the language modality. As multimodal large language models can better simulate the real world, how do they achieve stronger multimodal understanding and generation? Can multimodal large language models help achieve AGI?


## Learning Objectives

1. Become familiar with the types of multimodal large language models
2. Master the general technical framework of multimodal large language models
3. Master the building, training, and inference of multimodal large language models



## 1. Theory Basics



### 1.1 Understanding MLLM Types

- Functional and modality-support classification of existing multimodal large language models


> Before building multimodal large language models (MLLMs), researchers in the field reached a shared consensus and a key assumption: because of scaling laws and emergent phenomena, current language-based LLMs already have strong semantic understanding abilities. This means language has become the key modality for carrying intelligence, so language intelligence is regarded as the hub of multimodal intelligence. Therefore, almost all MLLMs are built on language-based LLMs, which act as the core decision-making module, similar to a brain or central processor. In other words, by adding extra external non-text modality modules or encoders, the LLM is given multimodal perception and action capabilities.

We classify existing MLLMs into different types based on their modality support and functional support.

![mllm](assets/MLLM-summary.png)


- Read the full tutorial: [[Slides](https://github.com/Lordog/dive-into-llms/blob/main/documents/chapter6/mllms.pdf)] 


- More related surveys
    - [A Survey on Multimodal Large Language Models, https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models, 2023](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models)
    - [MM-LLMs: Recent Advances in MultiModal Large Language Models, 2023](https://arxiv.org/pdf/2401.13601)


### 1.2 Understanding the Common Technical Framework of MLLMs


- Architecture 1: LLM as Task Scheduler

> There are currently two common MLLM architectures in the community.
The first is the "LLM as a discrete scheduler/controller" architecture. As shown below, the LLM receives text signals and issues text commands to downstream modules. All communication inside the system happens through pure text commands produced by the LLM. There is no interaction between different functional modules.


![Architecture1](assets/Architecture1.png)

- Architecture 2: LLM as a Joint Part of the System

> The second architecture is the encoder-LLM-decoder framework.
This is also the most popular architecture today. Here, the LLM perceives multimodal information and responds or acts within an encoder-LLM-decoder structure. The key difference from the first architecture is that the LLM is a critical joint part of the system: it directly receives multimodal information from outside and delegates instructions to the decoder/generator in a more fluid way. In the encoder-LLM-decoder framework, as shown below, the encoder processes encoded signals from multiple modalities, the LLM acts as the core decision-maker, and the decoder manages multimodal outputs.

![Architecture2](assets/Architecture2.png)





## 2. Getting Started with General-Purpose Multimodal Large Language Models

> A hands-on walkthrough of building a general-purpose multimodal large language model for any-modality-to-any-modality tasks.


### 2.1 A Unified General-Purpose Any-to-Any MLLM: NExT-GPT

> Future MLLM research will continue to move toward increasingly generalist systems, so it will include as many modalities and functions as possible. NExT-GPT is one of the most pioneering works in this area; it was the first to introduce the concept of "any-to-any modality" MLLMs. This architecture enables powerful capabilities and lays the foundation for future research directions in multimodal large language models.

![NExT-GPT](assets/NExT-GPT-screen.png)


> The coding portion of this course on multimodal large language models will use the NExT-GPT codebase as the target for a detailed, accessible analysis and hands-on practice.

[NExT-GPT Project](https://next-gpt.github.io/)

[NExT-GPT GitHub Repository](https://github.com/NExT-GPT/NExT-GPT)




### 2.2 Codebase Overview


```
├── figures
├── data
│   ├── T-X_pair_data  
│   │   ├── audiocap                      # text-autio pairs data
│   │   │   ├── audios                    # audio files
│   │   │   └── audiocap.json             # the audio captions
│   │   ├── cc3m                          # text-image paris data
│   │   │   ├── images                    # image files
│   │   │   └── cc3m.json                 # the image captions
│   │   └── webvid                        # text-video pairs data
│   │   │   ├── videos                    # video files
│   │   │   └── webvid.json               # the video captions
│   ├── IT_data                           # instruction data
│   │   ├── T+X-T_data                    # text+[image/audio/video] to text instruction data
│   │   │   ├── alpaca                    # textual instruction data
│   │   │   ├── llava                     # visual instruction data
│   │   ├── T-T+X                         # synthesized text to text+[image/audio/video] instruction data
│   │   └── MosIT                         # Modality-switching Instruction Tuning instruction data
├── code
│   ├── config
│   │   ├── base.yaml                     # the model configuration 
│   │   ├── stage_1.yaml                  # enc-side alignment training configuration
│   │   ├── stage_2.yaml                  # dec-side alignment training configuration
│   │   └── stage_3.yaml                  # instruction-tuning configuration
│   ├── dsconfig
│   │   ├── stage_1.json                  # deepspeed configuration for enc-side alignment training
│   │   ├── stage_2.json                  # deepspeed configuration for dec-side alignment training
│   │   └── stage_3.json                  # deepspeed configuration for instruction-tuning training
│   ├── datast
│   │   ├── base_dataset.py
│   │   ├── catalog.py                    # the catalog information of the dataset
│   │   ├── cc3m_datast.py                # process and load text-image pair dataset
│   │   ├── audiocap_datast.py            # process and load text-audio pair dataset
│   │   ├── webvid_dataset.py             # process and load text-video pair dataset
│   │   ├── T+X-T_instruction_dataset.py  # process and load text+x-to-text instruction dataset
│   │   ├── T-T+X_instruction_dataset.py  # process and load text-to-text+x instruction dataset
│   │   └── concat_dataset.py             # process and load multiple dataset
│   ├── model                     
│   │   ├── ImageBind                     # the code from ImageBind Model
│   │   ├── common
│   │   ├── anyToImageVideoAudio.py       # the main model file
│   │   ├── agent.py
│   │   ├── modeling_llama.py
│   │   ├── custom_ad.py                  # the audio diffusion 
│   │   ├── custom_sd.py                  # the image diffusion
│   │   ├── custom_vd.py                  # the video diffusion
│   │   ├── layers.py                     # the output projection layers
│   │   └── ...  
│   ├── scripts
│   │   ├── train.sh                      # training NExT-GPT script
│   │   └── app.sh                        # deploying demo script
│   ├── header.py
│   ├── process_embeddings.py             # precompute the captions embeddings
│   ├── train.py                          # training
│   ├── inference.py                      # inference
│   ├── demo_app.py                       # deploy Gradio demonstration 
│   └── ...
├── ckpt                           
│   ├── delta_ckpt                        # tunable NExT-GPT params
│   │   ├── nextgpt         
│   │   │   ├── 7b_tiva_v0                # the directory to save the log file
│   │   │   │   ├── log                   # the logs
│   └── ...       
│   ├── pretrained_ckpt                   # frozen params of pretrained modules
│   │   ├── imagebind_ckpt
│   │   │   ├──huge                       # version
│   │   │   │   └──imagebind_huge.pth
│   │   ├── vicuna_ckpt
│   │   │   ├── 7b_v0                     # version
│   │   │   │   ├── config.json
│   │   │   │   ├── pytorch_model-00001-of-00002.bin
│   │   │   │   ├── tokenizer.model
│   │   │   │   └── ...
├── LICENCE.md
├── README.md
└── requirements.txt
```


### 2.3 Environment Setup

Clone the repository and install the required environment by running the following commands:

```
conda env create -n nextgpt python=3.8

conda activate nextgpt

# CUDA 11.6
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia

git clone https://github.com/NExT-GPT/NExT-GPT.git
cd NExT-GPT

pip install -r requirements.txt
```




### 2.4 Getting Started with Inference


#### 2.4.1 Load the Pretrained NExT-GPT Checkpoint

- **Step 1**: Load the `frozen parameters`. [NExT-GPT](https://github.com/NExT-GPT/NExT-GPT) is trained on the following existing models and modules, so please prepare the checkpoints as described below.
    - `ImageBind` is a unified image/video/audio encoder. You can download the pretrained checkpoint version `huge` from [here](https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth). Then place `imagebind_huge.pth` under [[./ckpt/pretrained_ckpt/imagebind_ckpt/huge]](ckpt/pretrained_ckpt/imagebind_ckpt/).
    - `Vicuna`: First prepare LLaMA following the instructions in [[here]](ckpt/pretrained_ckpt/prepare_vicuna.md). Then place the pretrained model under [[./ckpt/pretrained_ckpt/vicuna_ckpt/]](ckpt/pretrained_ckpt/vicuna_ckpt/).
    - `Image Diffusion` is used to generate images. NExT-GPT uses [Stable Diffusion](https://huggingface.co/runwayml/stable-diffusion-v1-5) version `v1-5`. (_The code will download it automatically_).
    - `Audio Diffusion` is used to generate audio content. NExT-GPT uses [AudioLDM](https://github.com/haoheliu/AudioLDM) version `l-full`. (_The code will download it automatically_).
    - `Video Diffusion` is used for video generation. We use [ZeroScope](https://huggingface.co/cerspense/zeroscope_v2_576w) version `v2_576w`. (_The code will download it automatically_).

- **Step 2**: Load the `trainable parameters`.

Place the NExT-GPT system under [[./ckpt/delta_ckpt/nextgpt/7b_tiva_v0]](./ckpt/delta_ckpt/nextgpt/7b_tiva_v0). You can either 1) use your own trained parameters, or 2) download the pretrained checkpoint from [Huggingface](https://huggingface.co/ChocoWu/nextgpt_7b_tiva_v0).


#### 2.4.2 Gradio Demo Deployment

After the checkpoints are loaded, you can run the demo locally with the following commands:
```angular2html
cd ./code
bash scripts/app.sh
```
The key parameter is:
- `--nextgpt_ckpt_path`: path to the pretrained NExT-GPT parameters.



#### 2.4.3 Test Examples

The current version supports arbitrary combinations of text, image, video, and audio inputs, as well as outputs across combined modalities.
It also supports multi-turn contextual interaction.

Please run the examples yourself to see the results.

- **Case 1**: Input T+I, output T+A

![case1](assets/T+I-T+A.png)


- **Case 2**: Input T+V, output T+A

![case2](assets/T+V-T+A.png)


- **Case 3**: Input T+I, output T+I+V

![case3](assets/T+I-T+I+V.png)


- **Case 4**: Input T, output T+I+V+A

![case4](assets/T-T+I+V+A.png)







### 2.5 Training Workflow


#### 2.5.1 Data Preparation

Please download the following datasets for model training:

A) T-X Pair Data
  - ***Text-image*** pair data from `CC3M`; follow the instructions [[here]](./data/T-X_pair_data/cc3m/prepare.md). Then place the data in [[./data/T-X_pair_data/cc3m]](./data/T-X_pair_data/cc3m).
  - ***Text-video*** pair data from `WebVid`; refer to the [[instructions]](./data/T-X_pair_data/webvid/prepare.md). Store the files in [[./data/T-X_pair_data/webvid]](./data/T-X_pair_data/webvid).
  - ***Text-audio*** pair data from `AudioCap`; refer to the [[instructions]](./data/T-X_pair_data/audiocap/prepare.md). Store the data in [[./data/T-X_pair_data/audiocap]](./data/T-X_pair_data/audiocap).

B) Instruction-Tuning Data
  - T+X-T
    - ***Visual instruction data*** (from `LLaVA`), download it from [here](https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md) and place it in [[./data/IT_data/T+X-T_data/llava]](./data/IT_data/T+X-T_data/llava/).
    - ***Text instruction data*** (from `Alpaca`), download it from [here](https://github.com/tatsu-lab/stanford_alpaca) and place it in [[./data/IT_data/T+X-T_data/alpaca/]](data/IT_data/T+X-T_data/alpaca/).
    - ***Video instruction data*** (from `VideoChat`), download it from [here](https://github.com/OpenGVLab/InternVideo/tree/main/Data/instruction_data) and place it in [[./data/IT_data/T+X-T_data/videochat/]](data/IT_data/T+X-T_data/videochat/).

    Note: After downloading the datasets, run `preprocess_dataset.py` to preprocess them and make the formats consistent.
  - T-X+T (T2M)
    - The `T-X+T` instruction dataset (T2M) is stored in [[./data/IT_data/T-T+X_data]](./data/IT_data/T-T+X_data).
   
  - MosIT
    - Get the download instructions from [here](./data/IT_data/MosIT_data/). Finally, place the data in [[./data/IT_data/MosIT_data/]](./data/IT_data/MosIT_data/).




#### 2.5.2 Embedding Preparation

In decoder-side alignment training for NExT-GPT, we minimize the distance between the representations of Signal Tokens and captions. To keep the system efficient and save time and memory, we precompute text embeddings for image, audio, and video captions using the text encoders from the respective diffusion models.

Before training NExT-GPT, run the following command. The generated `embedding` files will be saved in [[./data/embed]](./data/embed).
```
cd ./code/
python process_embeddings.py ../data/T-X_pair_data/cc3m/cc3m.json image ../data/embed/ runwayml/stable-diffusion-v1-5
```


Parameter descriptions:
- args[1]: path to the caption file;
- args[2]: modality, which can be `image`, `video`, or `audio`;
- args[3]: path where the embedding file is saved;
- args[4]: name of the corresponding pretrained diffusion model.




#### 2.5.3 Three-Stage Training


First refer to the base configuration file [[./code/config/base.yaml]](./code/config/base.yaml) to understand the basic system settings for the whole module.

Then run the following script to start training NExT-GPT:
```
cd ./code
bash scripts/train.sh
```

The command is:
```
deepspeed --include localhost:0 --master_addr 127.0.0.1 --master_port 28459 train.py \
    --model nextgpt \
    --stage 1\
    --save_path  ../ckpt/delta_ckpt/nextgpt/7b_tiva_v0/\
    --log_path ../ckpt/delta_ckpt/nextgpt/7b_tiva_v0/log/
```
Key parameters:
- `--include`: `localhost:0` means CUDA device `0` in DeepSpeed.
- `--stage`: training stage.
- `--save_path`: directory for storing the trained delta weights. This directory will be created automatically.
- `--log_path`: directory for storing log files.



The full NExT-GPT training process is divided into three steps:

- **Step 1**: encoder-side LLM-centered multimodal alignment. This stage trains the ***input projection layer*** while freezing ImageBind, the LLM, and the output projection layer.
  
  Just run the `train.sh` script above with `--stage 1`.
  
  Also refer to the runtime config file [[./code/config/stage_1.yaml]](./code/config/stage_1.yaml) and the DeepSpeed config file [[./code/dsconfig/stage_1.yaml]](./code/dsconfig/stage_1.yaml) for more step-by-step configuration details.

  Note that the datasets used in this step are included in `dataset_name_list`, and the dataset names must exactly match the definitions in [[./code/dataset/catalog.py]](./code/dataset/catalog.py).



- **Step 2**: decoder-side instruction-following alignment. This stage trains the ***output projection layer*** while freezing ImageBind, the LLM, and the input projection layer.

  Just run the `train.sh` script above with `--stage 2`.

  Also refer to the runtime config file [[./code/config/stage_2.yaml]](./code/config/stage_2.yaml) and the DeepSpeed config file [[./code/dsconfig/stage_2.yaml]](./code/dsconfig/stage_2.yaml) for more step-by-step configuration details.



- **Step 3**: instruction tuning. In this stage, the instruction dataset is adapted by 1) tuning the ***LLM*** through LoRA, 2) tuning the ***input projection layer***, and 3) tuning the ***output projection layer***.

  Just run the `train.sh` script above with `--stage 3`.

  Also refer to the runtime config file [[./code/config/stage_3.yaml]](./code/config/stage_3.yaml) and the DeepSpeed config file [[./code/dsconfig/stage_3.yaml]](./code/dsconfig/stage_3.yaml) for more step-by-step configuration details.
