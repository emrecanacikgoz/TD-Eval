# TD-EVAL: Revisiting Task-Oriented Dialogue Evaluation by Combining Turn-Level Precision with Dialogue-Level Comparisons
[![Paper](https://img.shields.io/badge/arXiv-Paper-red.svg)](https://arxiv.org/abs/2504.19982)

TD-Eval is a framework for evaluating conversational agents and their ability to assess dialogue quality. This README provides a step-by-step guide to set up the environment, configure API credentials, run evaluations, and use the Qualtrics integration.

[![Paper](https://img.shields.io/badge/arXiv-Paper-red.svg)](https://arxiv.org/abs/2502.08820)

## Table of Contents
1. [Project Structure](#project-structure)
2. [Environment Setup](#environment-setup)
3. [API Credential Configuration](#api-credential-configuration)
4. [Running Evaluations](#running-evaluations)
5. [Supported Clients](#supported-clients)
6. [Qualtrics Survey Integration](#qualtrics-survey-integration)

---

## Project Structure
The project is organized into the following modules:

- **generate/**: Contains code for generating agent responses
  - `generate.py`: Main entry point for generating agent responses
  - `llm_agents.py`: Client interfaces for different LLM providers
  - `mw_database.py`: MultiWOZ database utilities
  - `prompts/`: Agent prompts for different domains

- **judge/**: Contains code for evaluating agent responses
  - `judge.py`: Main entry point for judging agent responses
  - `evaluator.py`: Evaluation logic and metrics
  - `prompts/`: Judge prompts for different evaluation dimensions

- **postprocess/**: Contains code for postprocessing evaluation results
  - `postprocess.py`: Functions for generating visualizations and statistics

- **qualtrics/**: Contains code for Qualtrics integration
  - `qualtrics.py`: Main entry point for generating Qualtrics surveys
  - `qualtrics_utils.py`: Utility functions for Qualtrics

- **main.py**: Main entry point for the entire framework

---

## Environment Setup
Follow these steps to set up the environment:

1. Clone the TDEval repository:
   ```bash
   git clone git@github.com:emrecanacikgoz/TD-Eval.git
   cd TD-Eval
   ```

2. Create a new Conda environment and install the required dependencies:
   ```bash
   conda create --name td-eval python=3.12 -y
   conda activate td-eval
   pip install -r requirements.txt
   ```

3. Clone the MultiWOZ_Evaluation repo
   ```bash
   rm -rf MultiWOZ_Evaluation
   git clone https://github.com/Tomiinek/MultiWOZ_Evaluation
   ```

4. Install MultiWOZ Evaluation pip requirements
   ```bash
   cd MultiWOZ_Evaluation
   pip install -r requirements.txt
   ```

---

## API Credential Configuration
TDEval requires API keys for supported LLM clients. Obtain the API keys by contacting the respective providers or authors.

Set the API credentials as environment variables:
```bash
export TOGETHER_API_KEY="<your_together_api_key>"
export ANTHROPIC_API_KEY="<your_anthropic_api_key>"
export OPENAI_API_KEY="<your_openai_api_key>"
export MISTRAL_API_KEY="<your_mistral_api_key>"
```
Replace `<your_xxx_api_key>` with the appropriate key.

---

## Running Evaluations
You can evaluate dialogue agents using the following commands:

1. **Standard evaluation with OpenAI's GPT-4o-mini as the agent (to generate multiwoz responses) and judge:**
   ```bash
   python main.py --dataset_path path/to/dataset.json \
                  --agent_client openai \
                  --agent_model gpt-4o-mini \
                  --judge_client openai \
                  --judge_model gpt-4o-mini
   ```

2. **Evaluation with TogetherAI's meta-llama as judge. The agent responses for judgement are provided as a json file through (--agent_result_path)**
   ```bash
   python main.py --agent_result_path path/to/agent/results.json \
                  --judge_client togetherai \
                  --judge_model meta-llama/Llama-3.1-405B-Instruct
   ```

3. **Skip generation and use pre-generated agent responses:**
   ```bash
   python main.py --agent_result_path path/to/agent/results.json \
                  --skip_generation \
                  --judge_client openai \
                  --judge_model gpt-4o
   ```

4. **Generate responses without judging them:**
   ```bash
   python main.py --dataset_path path/to/dataset.json \
                  --agent_client openai \
                  --agent_model gpt-4o-mini \
                  --skip_judging
   ```

5. **Evaluate tau-bench results with TogetherAI's meta-llama as judge (tau-bench dialogue provided as json file in --dataset_path)**
   ```bash
   python main.py --dataset_path path/to/dataset.json \
                  --judge_client togetherai \
                  --judge_model meta-llama/Llama-3.1-405B-Instruct \
                  --tau_tool
   ```

Customize the `dataset_path`, `agent_client`, `agent_model`, `agent_result_path`, `use_gt_state`, `judge_client`, `judge_model`, and `tau_tool`/`tau_react` parameters as needed.

---

## Supported Clients
TDEval supports the following LLM clients:
- **TogetherAI**
- **OpenAI**
- **Anthropic**
- **MistralAI**

Ensure you have the required API keys for the clients you plan to use.

---

## Qualtrics Survey Integration
TDEval provides a module, `qualtrics/qualtrics.py`, to convert evaluation results into a format compatible to import as Qualtrics surveys.

### Usage
Run the following command to generate a Qualtrics-compatible file:
```bash
python -m qualtrics.qualtrics --batch_path batches.json \
                              --mwoz_path results/<your_results_file>.json \
                              --output_name qualtrics_output.txt
```

### Parameters
- `--batch_path`: Path to the JSON file specifying batches of dialogues.
- `--mwoz_path`: Path to the model output JSON file for MultiWOZ datasets.
- `--is_autotod`: Flag to indicate AutoTOD format.
- `--tau_retail_path`: Path to retail TAU benchmark results.
- `--tau_airline_path`: Path to airline TAU benchmark results.
- `--tau_react`: Flag to indicate TAU React format.
- `--output_name`: Name of the output file.

### Import into Qualtrics
1. Create a new survey in Qualtrics.
2. Navigate to `Tools > Import/Export > Import Survey`.
3. Upload the generated `.txt` file.

### Notes
- Processing more than 50 dialogue questions may take over 10 minutes.

---

### Agreement Calculation Scripts

TDEval provides python notebook scripts for calculating human agreement based on the evaluation scores collected in `dataset/main_human_eval`. The script that calculates the first-step inter-annotator agreement is `calculate_irr.ipynb`. The script that calculates the second-step of human evaluation is `calculate_annotator_agreement.ipynb`.

---

For your inqueries, please contact with authors `acikgoz2@illinous.edu`, `carlguo2@illinois.edu`, `sdey@illinois.edu`.
