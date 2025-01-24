# TD-Eval
From Turns to Dialogues: Rethinking TOD Evaluation by Combining Turn-Level Precision with Dialogue-Level Comparisons

TD-Eval is a framework for evaluating conversational agents and their ability to assess dialogue quality. This README provides a step-by-step guide to set up the environment, configure API credentials, run evaluations, and use the Qualtrics integration.

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [API Credential Configuration](#api-credential-configuration)
3. [Running Evaluations](#running-evaluations)
4. [Supported Clients](#supported-clients)
5. [Qualtrics Survey Integration](#qualtrics-survey-integration)

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

1. **Standard evaluation with OpenAI's GPT-4o-mini as the agent and judge:**
   ```bash
   python main.py --result_path path/to/results.json \
                  --agent_client openai \
                  --llm_agent gpt-4o \
                  --agent_model openai \
                  --agent_result_path path/to/agent/results.json (optional) \
                  --judge_client openai \
                  --judge_model gpt-4o \
   ```

2. **Evaluation with TogetherAI's meta-llama as judge:**
   ```bash
   python main.py --result_path path/to/results.json \
                  --llm_agent_client openai \
                  --llm_agent gpt-4o-mini \
                  --agent_result_path path/to/agent/results.json (optional) \
                  --llm_judge_agent_client togetherai \
                  --llm_judge_agent meta-llama/Llama-3.1-405B-Instruct
   ```

3. **Judge GPT-4o-mini with TogetherAI's meta-llama:**
   ```bash
   python main.py --result_path path/to/results.json \
                  --llm_agent_client openai \
                  --llm_agent gpt-4o-mini \
                  --agent_result_path path/to/agent/results.json (optional) \
                  --llm_judge_agent_client togetherai \
                  --llm_judge_agent meta-llama/Llama-3.1-405B-Instruct-4o
   ```

Customize the `result_path`, `llm_agent_client`, `llm_agent`, `agent_result_path`, `llm_judge_agent_client`, and `llm_judge_agent` parameters as needed.

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
TDEval provides a script, `convert_qualtrics.py`, to convert evaluation results into a format compatible with Qualtrics surveys.

### Usage
Run the following command to generate a Qualtrics-compatible file:
```bash
python3 convert_qualtrics.py --result_path results/<your_results_file>.json \
                             --batch_path batches.json \
                             --output_path qualtrics/<output_file>.txt
```

### Parameters
- `--result_path`: Path to the model output JSON file.
- `--batch_path`: Path to the JSON file specifying batches of dialogues.
- `--output_path`: Path to save the Qualtrics-compatible text file.

### Import into Qualtrics
1. Create a new survey in Qualtrics.
2. Navigate to `Tools > Import/Export > Import Survey`.
3. Upload the generated `.txt` file.

### Example Commands
```bash
python3 convert_qualtrics.py --result_path results/openai/gpt4omini-c_gpt4omini-j.json \
                             --batch_path batches.json \
                             --output_path qualtrics/qualtrics_gpt4omini.txt

python3 convert_qualtrics.py --result_path results/20241121_015210/gpt4o_c-gpt4o_j.json \
                             --batch_path batches.json \
                             --output_path qualtrics/qualtrics_gpt4o_j.txt
```

### Notes
- Processing more than 50 dialogue questions may take over 10 minutes.

---

For more information, please contact with authors Emre Can Acikgoz, Carl Guo, and Akul Datta.