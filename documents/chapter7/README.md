# LLM Text Steganography

This project implements text steganography based on large language models (LLMs). It uses a GPT-2 model to hide information inside generated text. The project includes two encoding methods: Huffman Coding and Fixed Length Coding (FLC).

## Features

- Implements steganography in GPT-2 text generation
- Supports two encoding methods:
  - Huffman Coding
  - Fixed Length Coding (FLC)
- Generates both steganographic text and normal text for comparison

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- CUDA (optional, for GPU acceleration)

## Installation

Install the dependencies:

```bash
pip install torch # If you are using a GPU, make sure to install the GPU-enabled torch package
pip install transformers
pip install jupyter
```

## Usage

1. Open Jupyter Notebook:
```bash
jupyter notebook llm_stega.ipynb
```

2. The notebook contains three main parts:
   - Huffman Coding implementation
   - Fixed Length Coding (FLC) implementation
   - Text generation and steganography

3. Generate steganographic text:
   - Choose an encoding method (Huffman Coding or FLC)
   - Set the parameter value for the encoding method
   - Run the main function with a custom prompt

Example code:
```python
# Initialize the steganography handler
k = 2
handle = Huffman(k=2**k, bits=bits)
# Or use FLC
# handle = FLC(k=k, bits=bits)

# Generate text with steganography
prompt = "Hello, I'm"
stega_output, normal_output = generate_text_with_steganography(
    model, tokenizer, prompt, handle
)
```

## Output Files

The program generates two text files:
- `outputs-gpt2-stega.txt`: steganographic text
- `outputs-gpt2-normal.txt`: normally generated text without steganographic constraints

## Notes

- The GPT-2 model will be downloaded automatically the first time you run the notebook. If needed, you can also download it in advance from the https://hf-mirror.com mirror or use another model.
- Make sure you have enough disk space available. The model is about 500 MB.
- A CUDA-capable GPU is recommended for better performance.
- The length of the hidden information affects the length of the generated text.

## How It Works

1. Steganography process:
   - Convert secret information into a binary sequence
   - Use the GPT-2 model to generate text
   - Embed the information bits into the generated text through the encoding method

2. Decoding process:
   - Use the same GPT-2 model and context
   - Extract hidden information bits from the text through the decoding method
   - Convert the binary sequence back to the original information
