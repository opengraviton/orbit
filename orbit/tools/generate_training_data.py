"""Generate training data using the current model — self-improvement foundation."""

from __future__ import annotations

import json
from pathlib import Path

from orbit.tools.base import Tool, ToolResult


class GenerateTrainingDataTool(Tool):
    """
    Generate synthetic training data using the current LLM.
    Creates JSONL with instruction/response pairs for self-training.
    """

    name = "generate_training_data"
    description = "Generate training examples with current model. args: prompt_template (str), num_examples (int), output_path (str). Needs _llm from agent."

    def run(
        self,
        prompt_template: str,
        num_examples: int = 20,
        output_path: str = "./orbit_data/generated.jsonl",
        _llm=None,
    ) -> ToolResult:
        if _llm is None:
            return ToolResult(
                success=False,
                output="",
                error="This tool requires the agent's LLM. Use via orbit run.",
            )
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            examples = []
            for i in range(num_examples):
                prompt = prompt_template.format(i=i, n=num_examples)
                response = _llm.generate(prompt, max_tokens=256, temperature=0.8)
                examples.append({"prompt": prompt, "response": response.strip()})
            with open(output_path, "w") as f:
                for ex in examples:
                    f.write(json.dumps({"content": f"{ex['prompt']}\n{ex['response']}"}) + "\n")
            return ToolResult(
                success=True,
                output=f"Generated {num_examples} examples to {output_path}",
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
