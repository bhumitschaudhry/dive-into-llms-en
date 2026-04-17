# Hands-on Large Language Models: Model Watermarking

Guide: Introduction to watermarking for language models

> Embed undetectable "watermarks" into language model generated content that are imperceptible to humans but can be detected algorithmically.

## Tutorial Objectives

1. **Watermark Embedding**: Embed watermarks during language model text generation
2. **Watermark Detection**: Calculate watermark strength for given text
3. **Watermark Evaluation**: Assess detection performance of watermark methods
4. **Evaluate Watermark Robustness** (Optional)

---

## Preparation

### 2.1 About the X-SIR Repository

https://github.com/zwhe99/X-SIR

X-SIR repository implements:
- Three text watermarking algorithms: **X-SIR, SIR and KGW**
- Two watermark removal attack methods: **paraphrase and translation**

![X-SIR Architecture](./assets/x-sir.png)

### 2.2 Environment Setup

```shell
git clone https://github.com/zwhe99/X-SIR && cd X-SIR
conda create -n xsir python==3.10.10
conda activate xsir
pip3 install -r requirements.txt
# [optional] pip3 install flash-attn==2.3.3
```

> Versions in requirements.txt are recommended, not mandatory.

---

## Practical Walkthrough

> This example uses the KGW algorithm to embed watermarks in language model outputs

### 3.1 Data Preparation

Organize model prompts into a JSONL file:

```json
{"prompt": "Ghost of Emmett Till: Based on Real Life Events "}
{"prompt": "Antique Cambridge Glass Pink Decagon Console Bowl Engraved Gold Highlights"}
{"prompt": "2009 > Information And Communication Technology Index statistics - Countries "}
...
```

- Each line is a JSON object containing at least the `prompt` key
- Example input file: `data/dataset/mc4/mc4.en.jsonl` (contains 500 entries, you may reduce this for faster testing)

### 3.2 Watermark Embedding

- Select model and watermark algorithm. This example uses `baichuan-inc/Baichuan-7B` with the KGW method:

  ```shell
  MODEL_NAME=baichuan-inc/Baichuan-7B
  MODEL_ABBR=baichuan-7b
  WATERMARK_METHOD_FLAG="--watermark_method kgw"
  ```

- Generate watermarked content:

  ```shell
  python3 gen.py \
      --base_model $MODEL_NAME \
      --fp16 \
      --batch_size 32 \
      --input_file data/dataset/mc4/mc4.en.jsonl \
      --output_file gen/$MODEL_ABBR/kgw/mc4.en.mod.jsonl \
      $WATERMARK_METHOD_FLAG
  ```

  - Model outputs are saved to: `gen/$MODEL_ABBR/kgw/mc4.en.mod.jsonl`
  - Output format (model response is stored in the `response` field):

    ```json
    {"prompt": "Ghost of Emmett Till: Based on Real Life Events ", "response": ".In August if 1955 African American Emmett Louis Till (21)\nThe second part of The Man From Waco, about Dan Millers trial for murdering his friend Michael Capps in a Texas wiener wrastle as I believe the statute says called it then; back at that time that would have surely occurred since Dan kept his pistol in one of those watery doggy bags he keeps around to clean himself with after emptying can into a nearby lake just minutes before committing his crime. If what we read is true thats exactly where Dan left his stolen gun and later used it in the robbery gone wrong which killed two innocent boys when his own accomplice got into an argument over not being paid enough therefore wanting out. This angered Miller whos history of mental instability could be taken one way or another but this criminal act was unavoidable once they entered FBIs hands and some other very powerful law officers who were involved either directly"}
    {"prompt": "Antique Cambridge Glass Pink Decagon Console Bowl Engraved Gold Highlights", "response": "An exceptionally fine decorative antique pink decagonal glass side bowl is the perfect example of early art and innovation . Fully engraved, this beautiful English vintage tableware piece exhibits a great degree on craftsmanship! Made in England during the mid 1800's it features three sets of concentric ribbons on the exterior to elegantly highlight an intricate, deep reddish color which evokes warmth and comfort for years to come! This historically significant vase has been featured within numerous museum exhibitions including \"Glass at The Corning Museum\" ; \"The First Half Century\" & a special travelling exhibit called:\" Sight Of Glass: British Cut Glass\" by ibex limited (retailer) as well as \"SIGNALS - Celebrating History In American Silver Through The Articulated Bottle Vessel\" presented at the Corning Museum of Glass 2012 ASA national symposium! We provide our customers with quality phot"}
    ...
    ```

### 3.3 Watermark Detection

> Calculate watermark strength (z-score) for input text.

- Calculate z-scores for **watermarked** text:

  ```shell
  python3 detect.py \
      --base_model $MODEL_NAME \
      --detect_file gen/$MODEL_ABBR/kgw/mc4.en.mod.jsonl \
      --output_file gen/$MODEL_ABBR/kgw/mc4.en.mod.z_score.jsonl \
      $WATERMARK_METHOD_FLAG
  ```

- Calculate z-scores for **unwatermarked** human text:

  ```shell
  python3 detect.py \
      --base_model $MODEL_NAME \
      --detect_file data/dataset/mc4/mc4.en.jsonl \
      --output_file gen/$MODEL_ABBR/kgw/mc4.en.hum.z_score.jsonl \
      $WATERMARK_METHOD_FLAG
  ```

- Output file format:

  ```json
  {"z_score": 12.105422509165574, "prompt": "Ghost of Emmett Till: Based on Real Life Events ", "response": ".In August if 1955 African American Emmett Louis Till (21)\nThe second part of The Man From Waco, about Dan Millers trial for murdering his friend Michael Capps in a Texas wiener wrastle as I believe the statute says called it then; back at that time that would have surely occurred since Dan kept his pistol in one of those watery doggy bags he keeps around to clean himself with after emptying can into a nearby lake just minutes before committing his crime. If what we read is true thats exactly where Dan left his stolen gun and later used it in the robbery gone wrong which killed two innocent boys when his own accomplice got into an argument over not being paid enough therefore wanting out. This angered Miller whos history of mental instability could be taken one way or another but this criminal act was unavoidable once they entered FBIs hands and some other very powerful law officers who were involved either directly", "biases": null}
  {"z_score": 12.990684249887122, "prompt": "Antique Cambridge Glass Pink Decagon Console Bowl Engraved Gold Highlights", "response": "An exceptionally fine decorative antique pink decagonal glass side bowl is the perfect example of early art and innovation . Fully engraved, this beautiful English vintage tableware piece exhibits a great degree on craftsmanship! Made in England during the mid 1800's it features three sets of concentric ribbons on the exterior to elegantly highlight an intricate, deep reddish color which evokes warmth and comfort for years to come! This historically significant vase has been featured within numerous museum exhibitions including \"Glass at The Corning Museum\" ; \"The First Half Century\" & a special travelling exhibit called:\" Sight Of Glass: British Cut Glass\" by ibex limited (retailer) as well as \"SIGNALS - Celebrating History In American Silver Through The Articulated Bottle Vessel\" presented at the Corning Museum of Glass 2012 ASA national symposium! We provide our customers with quality phot", "biases": null}
  ...
  ```

- Visually compare z-score distributions between the two output files.

### 3.4 Watermark Evaluation
<a name="eval"></a>

- Calculate detection accuracy and generate ROC curve from z-score files:

  ```shell
  python3 eval_detection.py \
          --hm_zscore gen/$MODEL_ABBR/kgw/mc4.en.hum.z_score.jsonl \
          --wm_zscore gen/$MODEL_ABBR/kgw/mc4.en.mod.z_score.jsonl \
          --roc_curve roc
  ```

  Example output:
  ```
  AUC: 1.000

  TPR@FPR=0.1: 0.998
  TPR@FPR=0.01: 0.998

  F1@FPR=0.1: 0.999
  F1@FPR=0.01: 0.999
  ```

![ROC Curve](./assets/curve.png)

---

## Robustness Evaluation (Optional)

> Test watermark detection performance after applying paraphrase or translation attacks.

### 4.1 Preparation

This example uses `gpt-3.5-turbo-1106` for attack operations.

- Set OpenAI API key:

  ```shell
  export OPENAI_API_KEY=xxxx
  ```

- Adjust rate limits (RPM/TPM) in `attack/const.py`

### 4.2 Perform Attack (Translation example)

- Translate watermarked text to Chinese:

  ```shell
  python3 attack/translate.py \
      --input_file gen/$MODEL_ABBR/kgw/mc4.en.mod.jsonl \
      --output_file gen/$MODEL_ABBR/kgw/mc4.en-zh.mod.jsonl \
      --model gpt-3.5-turbo-1106 \
      --src_lang en \
      --tgt_lang zh
  ```

- Re-evaluate detection performance
  - Use the same evaluation procedure as [3.4](#eval)
- Compare performance before and after attack.

---

## Advanced Exercises

- Review [X-SIR documentation](https://github.com/zwhe99/X-SIR) and implement the X-SIR and SIR algorithms.
- Evaluate performance of all three algorithms against different attack methods.
