"""
Prompting Techniques Module using Ollama + mistral:latest
---------------------------------------------------------

This module allows a programmer to easily call different prompting techniques
with a user prompt and (optionally) a system prompt.

If the system prompt is not provided, a generalized default will be used
based on the selected prompting technique.

Author: Suraj (Data Science Intern)
"""

import ollama
from typing import Optional


class PromptingTechniques:
    def __init__(self, model: str = "mistral:latest"):
        self.model = model

    def _chat(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Internal helper to call Ollama safely with exception handling.
        """
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})

            response = ollama.chat(model=self.model, messages=messages)
            return response.get("message", {}).get("content", "").strip()

        except Exception as e:
            return f"[Error] Failed to generate response: {e}"

    # -----------------------
    # Prompting Techniques
    # -----------------------

    def zero_shot(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        default_system = "You are a helpful AI assistant. Answer concisely."
        return self._chat(user_prompt, system_prompt or default_system)

    def few_shot(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        default_system = ("You are a sentiment analysis assistant. "
                          "Follow the style of given examples.")
        return self._chat(user_prompt, system_prompt or default_system)

    def chain_of_thought(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        default_system = ("You are an AI that solves problems step by step. "
                          "Always explain your reasoning before the final answer.")
        return self._chat(user_prompt, system_prompt or default_system)

    def generated_knowledge(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        default_system = ("You are an AI that first recalls relevant background knowledge "
                          "before answering.")
        return self._chat(user_prompt, system_prompt or default_system)

    def least_to_most(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        default_system = ("You are an AI that breaks tasks into smaller sub-problems "
                          "and solves from easy to hard.")
        return self._chat(user_prompt, system_prompt or default_system)

    def self_refine(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        default_system = ("You are an AI that first provides an initial answer, "
                          "then critiques and refines it for improvement.")
        return self._chat(user_prompt, system_prompt or default_system)

    def maieutic(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        default_system = ("You are an AI that reasons via multiple possible paths, "
                          "then reconciles them to give a final answer.")
        return self._chat(user_prompt, system_prompt or default_system)