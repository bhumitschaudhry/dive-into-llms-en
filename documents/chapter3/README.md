# Hands-on Large Language Models: Knowledge Editing for Large Models
Guide: Language model editing methods and tools
> Want to control how language models remember specific knowledge? Let's choose appropriate editing methods, edit specific knowledge, and verify the edited model!

## 1. Tutorial Objectives:

- Get familiar with using the EasyEdit toolkit
- Master (simplified) language model editing methods
- Understand selection and application scenarios for different editing approaches

## 2. Preparation:
### 2.1 Introduction to EasyEdit

https://github.com/zjunlp/EasyEdit

EasyEdit is a Python package for editing language models such as GPT-J, Llama, GPT-NEO, GPT2, T5, etc. Its goal is to effectively modify a language model's behavior for specific knowledge without negatively affecting performance on other inputs, while remaining easy to use and extend.

EasyEdit integrates existing popular editing methods:
![](./assets/1.png)

### 2.2 Core Framework

![](./assets/2.png)
EasyEdit includes a unified Editor, Method and Evaluation framework, representing editing scenarios, editing techniques and evaluation methods respectively:
- **Editor**: Describes the working scenario, includes the model to be edited, knowledge to edit, and other necessary hyperparameters
- **Method**: The specific knowledge editing technique used (e.g. ROME, MEND etc.)
- **Evaluate**: Metrics for assessing knowledge editing performance, including reliability, generality, locality, and portability
- **Trainer**: Some editing methods require training processes, implemented by the Trainer module

## 3. Environment Setup:
```
git clone https://github.com/zjunlp/EasyEdit.git
(Optional) conda create -n EasyEdit python=3.9.7
cd EasyEdit
pip install -r requirements.txt
```

## 4. Editing Example
> Goal: Modify GPT-2-XL's knowledge memory, change Lionel Messi's profession from football to basketball.

Steps:
- Select editing method and prepare parameters
- Prepare knowledge editing data
- Instantiate Editor
- Run!

Below is a detailed introduction using the ROME method:
### 4.1 ROME
Jupyter Notebook: [https://colab.research.google.com/drive/1KkyWqyV3BjXCWfdrrgbR-QS3AAokVZbr?usp=sharing#scrollTo=zWfGkNb9FBJQ]

- **Select editing method and prepare parameters**
  - Choose ROME as the editing method, prepare required parameters for ROME and GPT2-XL
  - Example: `alg_name: "ROME"`, `model_name: "./hugging_cache/gpt2-xl"` (or local path to model), `"device": GPU index to use
  - Remaining parameters can be kept as default
![](./assets/3.png)

- **Prepare knowledge editing data**
    ```python
    prompts = ['Question:What sport does Lionel Messi play? Answer:'] # x_e
    ground_truth = ['football'] # y
    target_new = ['basketball'] # y_e
    subject = ['Lionel Messi']
    ```

- **Instantiate Editor**: Pass prepared parameters to `BaseEditor` class to get customized Editor instance
    ```python
    hparams = ROMEHyperParams.from_hparams('./hparams/ROME/gpt2-xl.yaml')
    editor = BaseEditor.from_hparams(hparams)
    ```

- **Run!** Call the editor's edit method:
    ```python
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        subject=subject,
        keep_original_weight=False
    )
    ```
![](./assets/4.png)

> Note: When editing a model for the first time, Wiki corpus will be downloaded and layer statistics for the model will be calculated (saved to `stats_dir: "./data/stats"`) for reuse in subsequent edits. This may take some time initially - please be patient with a stable internet connection.

### 4.2 Validation and Evaluation
`editor.edit()` returns metrics calculated by EasyEdit's Evaluation module:
![](./assets/5.png)

To obtain numerical values for generality, locality and portability, evaluation data must be passed into the edit method.

For locality example: this will cause the edit method to calculate locality metrics (model answer accuracy on `locality_inputs`):
```python
locality_inputs = {
    'neighborhood': {
        'prompt': ['Joseph Fischhof, the', 'Larry Bird is a professional', 'In Forssa, they understand'],
        'ground_truth': ['piano', 'basketball', 'Finnish']
    }
}
metrics, edited_model, _ = editor.edit(
    prompts=prompts,
    ground_truth=ground_truth,
    target_new=target_new,
    locality_inputs=locality_inputs,
    keep_original_weight=False
)
```

Or directly compare model generation behavior before and after editing:
```python
generation_prompts = [
    "Lionel Messi, the",
    "The law in Ikaalinen declares the language"
]

model = GPT2LMHeadModel.from_pretrained('./hugging_cache/gpt2').to('cuda')
batch = tokenizer(generation_prompts, return_tensors='pt', padding=True, max_length=30)

pre_edit_outputs = model.generate(
    input_ids=batch['input_ids'].to('cuda'),
    attention_mask=batch['attention_mask'].to('cuda'),
    max_new_tokens=3
)
post_edit_outputs = edited_model.generate(
    input_ids=batch['input_ids'].to('cuda'),
    attention_mask=batch['attention_mask'].to('cuda'),
    max_new_tokens=3
)
```

## 5. Large-Scale Editing (Optional)
### 5.1 Batch Edit
Multiple data entries can be passed as parallel lists into the edit method for batch editing. MEMIT is the optimal method for this case.
(https://colab.research.google.com/drive/1P1lVklP8bTyh8uxxSuHnHwB91i-1LW6Z)

```python
prompts = [
    'Question:What sport does Lionel Messi play? Answer:',
    'The law in Ikaalinen declares the language'
]
ground_truth = ['football', 'Finnish']
target_new = ['basketball', 'Swedish']
subject = ['Lionel Messi', 'Ikaalinen']
```

### 5.2 Benchmark Testing
- Counterfact
- ZsRE

```json
{
    "case_id": 4402,
    "pararel_idx": 11185,
    "requested_rewrite": {
      "prompt": "{} debuted on",
      "relation_id": "P449",
      "target_new": {
        "str": "CBS",
        "id": "Q43380"
      },
      "target_true": {
        "str": "MTV",
        "id": "Q43359"
      },
      "subject": "Singled Out"
    },
    "paraphrase_prompts": [
      "No one on the ground was injured.  v",
      "The sex ratio was 1063. Singled Out is to debut on"
    ],
    "neighborhood_prompts": [
      "Daria premieres on",
      "Teen Wolf was originally aired on",
      "Spider-Man: The New Animated Series was originally aired on",
      "Celebrity Deathmatch premiered on",
      "Æon Flux premiered on",
      "My Super Psycho Sweet 16 premieres on",
      "Daria was released on",
      "Jersey Shore premiered on",
      "Skins was originally aired on",
      "All You've Got premiered on"
    ]
}
```

https://github.com/zjunlp/EasyEdit/blob/main/examples/run_zsre_llama2.py
