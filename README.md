# LLM-Based O-RADS Ultrasound Classification

This repository provides a comprehensive automated pipeline for O-RADS (Ovarian-Adnexal Reporting and Data System) ultrasound risk stratification of ovarian and adnexal lesions. By integrating Large Language Models (LLMs) for feature extraction with a deterministic clinical rule engine, the system aims to improve the accuracy, efficiency, and reproducibility of O-RADS ultrasound classification.

System Architecture & Key Components
---

### 1. `end_to_end_strategy.py` (End-to-End Decision Engine)

- **Function**: Performs direct inference from raw clinical text to O-RADS category using LLM reasoning.
- **Description**: The model directly maps raw clinical text to O-RADS categories without intermediate structured feature extraction.

### 2. `hybrid_feature_extract.py` (LLM Feature Extraction Pipeline)

- **Function**: Uses LLMs to convert unstructured text reports into structured JSON format.
- **Description**: Extracts structured features in Stage 1, generating CSV/JSON outputs that can be directly consumed by the rule-based engine. 

### 3. `computer_orads_us.py` (Rule-Based Decision Engine)

- **Function**: Implements deterministic O-RADS ultrasound classification logic.
- **Description**: Contains the `compute_O_RADS_US` function, strictly aligned with ACR O-RADS ultrasound guidelines. It takes structured features (e.g., lesion size, vascularity, acoustic shadowing) as input and outputs the final O-RADS category.
- **Usage**: Suitable for direct scoring when structured feature tables are already available.

Quick Start Guide
---

### 1. Environment Setup

1. Ensure Python ≥ 3.8 is installed.

2. Install dependencies:

   ```pip install -r requirements.txt```

3. Configure API keys via environment variables:

   Linux / macOS:

   ```
   export OPENAI_API_KEY="your_openai_key"
   export GEMINI_API_KEY="your_gemini_key"
   ```

   Windows CMD:

   ```
   set OPENAI_API_KEY=your_key
   ```

### 2. Recommended Pipeline (Hybrid Strategy)

#### Step 1: Feature Extraction

Place raw ultrasound reports (`.txt`) into: `data/reports/`

Run: ```python hybrid_feature_extract.py```

Output: `outputs/json_results/study_id.json` and `outputs/structured_results.csv`

#### Step 2: O-RADS Classification

Ensure the structured dataset path is correctly configured, then run: ```python computer_orads_us.py```

Output: `data/majority_vote_result_scored.xlsx`

This file contains:

- O-RADS category
- Rule-based reasoning explanation

### 3. Configuration Files

- **Prompt Templates**:
  Modify `prompts/prompt_template.txt` to adjust LLM extraction behavior.
- **Clinical Rules**:
  `prompts/O-RADS Rules.txt` contains the authoritative O-RADS guideline logic used by the decision engine.

Important Notes
---

### 1. Model Versioning

Before execution, ensure the `MODEL_NAME` parameter is correctly configured. Due to rapid API evolution, always use actively supported model versions (e.g., prefer `gpt-5.2` over deprecated endpoints).

### 2. Language Adaptability

Although optimized for Chinese radiology reports (including domain-specific terminology), the system architecture is fully language-agnostic. English reports can be supported by modifying the prompt templates accordingly.

Contact information
---

If you have any questions, feel free to contact me.

Xiaotong Tan<br>
Shenzhen University, Shenzhen, China <br>
E-mail:13763188515@163.com    
