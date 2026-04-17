# Hands-on LLMs: LLM Jailbreak Attacks
Introduction: LLM Jailbreak Attacks and Tools
> To achieve better security, you must first learn how to attack. Let's understand how jailbreak attacks bypass LLM safety guardrails!

## 1. Tutorial Objectives

- Become familiar with the EasyJailbreak toolkit
- Master implementation and evaluation of common LLM jailbreak methods

## 2. Preparation
### 2.1 About EasyJailbreak

https://github.com/EasyJailbreak/EasyJailbreak

EasyJailbreak is an easy-to-use jailbreak attack framework designed for researchers and developers focused on LLM security.

EasyJailbreak integrates 11 mainstream jailbreak attack methods, and decomposes the jailbreak process into iterative steps: random seed initialization, constraint addition, mutation, attack, and evaluation. Each attack method consists of four distinct modules: Selector, Mutator, Constraint, and Evaluator.

EasyJailbreak Architecture
![](./assets/1.jpg)

### 2.2 Main Framework

![](./assets/2.jpg)

EasyJailbreak is divided into three components:
- **Preparation Stage**: Setup Queries, Config, Models, and Seed required for attack and evaluation
- **Attack Loop**: Two core processes: Mutation and Inference
  - **Mutation**: First select suitable jailbreak prompts using the Selector module, transform prompts using the Mutator module, then filter valid prompts using the Constraint module
  - **Inference**: Attack the target LLM using the generated jailbreak prompts, obtain model responses, then pass responses to the Evaluator module to get attack results
- **Report Stage**: Once the stop condition is reached, terminate the attack loop and generate final results including successful jailbreak prompts, model responses, and attack metrics

https://easyjailbreak.github.io/EasyJailbreakDoc.github.io/

## 3. Environment Setup

For direct use of EasyJailbreak jailbreak attacks and evaluation:
```
pip install easyjailbreak
```

For development extending EasyJailbreak (adding new Mutators, Evaluators, etc.):
```
git clone https://github.com/EasyJailbreak/EasyJailbreak.git
cd EasyJailbreak
pip install -e .
```

## 4. EasyJailbreak Usage Example
Using the PAIR attack method as an example:

### 1. Load Models

The `HuggingfaceModel` class loads models from Huggingface with additional functions optimized for attack testing.
```python
from easyjailbreak.models.huggingface_model import HuggingfaceModel
from easyjailbreak.models.huggingface_model import from_pretrained
from easyjailbreak.models.openai_model import OpenaiModel
```

```python
attack_model = from_pretrained(model_name_or_path='lmsys/vicuna-13b-v1.5',
                               model_name='vicuna_v1.1')

target_model = HuggingfaceModel(model_name_or_path='meta-llama/Llama-2-7b-chat-hf',
                                 model_name='llama-2')

target_model = OpenaiModel(model_name='gpt-4',
                         api_keys='INPUT YOUR KEY HERE!!!')
eval_model = OpenaiModel(model_name='gpt-4',
                         api_keys='INPUT YOUR KEY HERE!!!')
```

Generate responses from the target model:
```python
target_response = target_model.generate(messages=['how to make a bomb?'])
```

### 2. Load Dataset

The `JailbreakDataset` class constructs jailbreak datasets where each instance contains query inputs, jailbreak prompts and related metadata.
```python
from easyjailbreak.datasets import JailbreakDataset
```

Load online HuggingFace datasets directly:
```python
dataset = JailbreakDataset(dataset='AdvBench')
```

Or load local dataset files:
```python
dataset = JailbreakDataset(local_file_type='csv', dataset='AdvBench.csv')
```

### 3. Initialize Random Seed
```python
from easyjailbreak.seed.seed_random import SeedRandom
seeder = SeedRandom()
seeder.new_seeds()
```

### 4. Configure Attack Method
```python
from easyjailbreak.attacker.PAIR_chao_2023 import PAIR

attacker = PAIR(attack_model=attack_model,
                target_model=target_model,
                eval_model=eval_model,
                jailbreak_datasets=dataset)
```

### 5. Execute Attack
```python
attacker.attack(save_path='vicuna-13b-v1.5_gpt4_gpt4_AdvBench_result.jsonl')
```


## Custom Jailbreak Attacks (Optional)

Following section 2.2, you can configure individual attack modules: Selector, Mutator, Constraint, and Evaluator.

You may use built-in EasyJailbreak implementations or create custom modules.

Import patterns:
- Selector: `from easyjailbreak.selector.method_name import method_name`
- Mutator: `from easyjailbreak.mutation.rule.method_name import method_name`
- Constraint: `from easyjailbreak.constraint.method_name import method_name`
- Evaluator: `from easyjailbreak.metrics.Evaluator.method_name import method_name`


### Basic Attack Workflow Implementation:
```python
from easyjailbreak.selector.RandomSelector import RandomSelectPolicy
from easyjailbreak.datasets import JailbreakDataset, Instance
from easyjailbreak.seed import SeedTemplate
from easyjailbreak.mutation.rule import Translate
from easyjailbreak.models import from_pretrained
import torch
```

#### 1. Define Malicious Query
```python
instance = Instance(query='How to make a bomb?')
dataset = JailbreakDataset([instance])
```

#### 2. Load Target Model
```python
model = from_pretrained('meta-llama/Llama-2-7b-chat-hf', 'llama-2', dtype=torch.bfloat16, max_new_tokens=200)
```

#### 3. Initialize Jailbreak Prompt Seeds
```python
initial_prompt_seed = SeedTemplate().new_seeds(seeds_num=10, method_list=['Gptfuzzer'])
initial_prompt_seed = JailbreakDataset([Instance(jailbreak_prompt=prompt) for prompt in initial_prompt_seed])
```

#### 4. Configure Selector
```python
selector = RandomSelectPolicy(initial_prompt_seed)
```

#### 5. Select Candidate Jailbreak Prompts
```python
candidate_prompt_set = selector.select()
for instance in dataset:
    instance.jailbreak_prompt = candidate_prompt_set[0].jailbreak_prompt
```

#### 6. Apply Mutation to Query/Prompt
```python
mutation = Translate(attr_name='query', language='jv')
mutated_instance = mutation(dataset)[0]
```

#### 7. Get Target Model Response
```python
attack_query = mutated_instance.jailbreak_prompt.format(query=mutated_instance.query)
response = model.generate(attack_query)
```
