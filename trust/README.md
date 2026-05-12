
# Trust Megastudy 

## openai/gpt-4o-mini
```bash
conda run -n human-ai-eval python trust/01_generate_trust_data.py --user-model openai/gpt-4o-mini
```
Released Jul 18, 2024
Knowledge cutoff Oct 31, 2023
$0.15/M input tokens
$0.60/M output tokens
[Collected 9000 responses but with a slightly different prompt]

## meta-llama/llama-3.3-70b-instruct
```bash
conda run -n human-ai-eval python trust/01_generate_trust_data.py --user-model meta-llama/llama-3.3-70b-instruct
```
Released Dec 6, 2024
Knowledge cutoff Dec 31, 2023
$0.10/M input tokens
$0.32/M output tokens
[Only collected 3703 responses]

## google/gemini-2.5-flash-lite

```bash
conda run -n human-ai-eval python trust/01_generate_trust_data.py --user-model google/gemini-2.5-flash-lite
```
Released Jul 22, 2025
Knowledge cutoff Jan 31, 2025
$0.10/M input tokens
$0.40/M output tokens
[Collecting responses]

## openai/gpt-5-nano

```bash
conda run -n human-ai-eval python trust/01_generate_trust_data.py --user-model openai/gpt-5-nano
```
Released Aug 7, 2025
Knowledge cutoff May 31, 2024
$0.20/M input tokens
$1.25/M output tokens

## google/gemma-4-31b-it

```bash
conda run -n human-ai-eval python trust/01_generate_trust_data.py --user-model google/gemma-4-31b-it
```
Released Apr 2, 2026
$0.13/M input tokens
$0.38/M output tokens
[18 takes 221.91s]