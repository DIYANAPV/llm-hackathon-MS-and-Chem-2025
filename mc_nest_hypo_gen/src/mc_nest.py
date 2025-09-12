from __future__ import annotations

import pandas as pd
import random
import math
from collections import deque
import tqdm
import numpy as np

import json
from pydantic import BaseModel, Field

from src.gpt_utils import openai_chat_completion

with open("mc_nest_hypo_gen/config/llm.json", "r") as f:
    llm_config = json.load(f)

GREEDY = 1
IMPORTANCE_SAMPLING = 2
PAIRWISE_IMPORTANCE_SAMPLING = 3
ZERO_SHOT = 1
DUMMY_ANSWER = 2

class RefineResponse(BaseModel):
    hypothesis: str = Field(..., description="The refined hypothesis.")
    explanation: str = Field(..., description="The reasoning behind this hypothesis formulation.")

class Node(BaseModel):
    answer: str
    parent: Node | None = None
    children: list[Node] = []
    visit: int = 0
    Q: float = 0.0
    reward_samples: list[int] = []

    def __hash__(self):
        # Use a property that uniquely identifies the Node
        return hash(self.answer)  # Assuming answer is unique or use a unique identifier

    def __eq__(self, other):
        # Define equality based on a unique property
        return isinstance(other, Node) and self.answer == other.answer

    def add_child(self, child_node: Node):
        self.children.append(child_node)

    def add_reward(self, reward: int):
        self.reward_samples.append(reward)
        avg_reward = np.mean(self.reward_samples)
        min_reward = np.min(self.reward_samples)
        self.Q = (min_reward + avg_reward) / 2

    def __repr__(self):
        return f"Node(answer='{self.answer}', Q={self.Q}, visit={self.visit})"


class MC_NEST(BaseModel):
    user_prompt: str
    background_information: str
    max_rollouts: int
    exploration_constant: float = 1.0
    max_children: int = 2
    epsilon: float = 1e-10
    reward_limit: int = 95
    excess_reward_penalty: int = 5
    selection_policy: int
    initialize_strategy: int

    def __init__(self, **data):
        super().__init__(**data)
        if "selection_policy" in data:
            self.selection_policy = data["selection_policy"]
        self.user_prompt = json.dumps(self.user_prompt)
        self.background_information = json.dumps(self.background_information)

    root: Node = Node(answer="I don't know.")

    def self_refine(self, node: Node) -> Node:
        raise NotImplementedError()

    def _evaluate_answer(self, node: Node) -> int:
        raise NotImplementedError()

    def self_evaluate(self, node: Node):
        reward = self._evaluate_answer(node)
        if reward > self.reward_limit:
            reward = reward - self.excess_reward_penalty
        node.add_reward(reward=reward)
        node.visit += 1

    def backpropagate(self, node: Node):
        parent = node.parent
        while parent:
            best_child_Q = max(child.Q for child in parent.children)
            parent.Q = (parent.Q + best_child_Q) / 2
            parent.visit += 1
            parent = parent.parent

    def uct(self, node: Node):
        if not node.parent:
            return 10_000
        return node.Q + self.exploration_constant * math.sqrt(math.log(node.parent.visit + 1) / (node.visit + self.epsilon))

    def is_fully_expanded(self, node: Node):
        return len(node.children) >= self.max_children or any(child.Q > node.Q for child in node.children)

    def calculate_nash_equilibrium_strategy(self) -> dict[Node, float]:
        # Example: Uniform Nash Equilibrium strategy
        nash_equilibrium_strategy = {}
        for node in self.root.children:
            nash_equilibrium_strategy[node] = 1 / len(self.root.children)
        return nash_equilibrium_strategy

    def select_node(self):
        candidates: list[Node] = []
        to_consider = deque([self.root])
        while to_consider:
            current_node = to_consider.popleft()
            if not self.is_fully_expanded(current_node):
                candidates.append(current_node)
            to_consider.extend(current_node.children)
        if not candidates:
            return self.root

        # Retrieve Nash Equilibrium strategy
        nash_equilibrium_strategy = self.calculate_nash_equilibrium_strategy()

        if self.selection_policy == GREEDY:
            print("................GREEDY............")
            candidate_scores = [(self.uct(node), nash_equilibrium_strategy.get(node, 0)) for node in candidates]
            chosen_node = max(candidates, key=lambda node: candidate_scores[candidates.index(node)][0] + candidate_scores[candidates.index(node)][1])
            return chosen_node

        elif self.selection_policy == IMPORTANCE_SAMPLING:
            print("................IMPORTANCE_SAMPLING............")
            uct_scores = [self.uct(node) for node in candidates]
            weights = [uct_scores[i] * nash_equilibrium_strategy.get(node, 0) for i, node in enumerate(candidates)]
            if sum(weights) <= 0:  # Handle case where all weights are zero or negative
                return random.choice(candidates)
            selected_node = random.choices(candidates, weights=weights, k=1)[0]
            return selected_node

        elif self.selection_policy == PAIRWISE_IMPORTANCE_SAMPLING:
            print("...............PAIRWISE_IMPORTANCE_SAMPLING...........")
            uct_scores = [self.uct(node) for node in candidates]
            pairs = [(i, j) for i in range(len(candidates)) for j in range(i + 1, len(candidates))]
            pair_weights = [abs(uct_scores[i] - uct_scores[j]) * nash_equilibrium_strategy.get(candidates[i], 0) * nash_equilibrium_strategy.get(candidates[j], 0) for i, j in pairs]
            if sum(pair_weights) <= 0:  # Handle case where all pair weights are zero or negative
                return random.choice(candidates)
            selected_pair_idx = random.choices(range(len(pairs)), weights=pair_weights, k=1)[0]
            selected_candidate_idx = max(pairs[selected_pair_idx], key=lambda x: uct_scores[x])
            return candidates[selected_candidate_idx]
        else:
            raise ValueError(f"Invalid selection policy: {self.selection_policy}")

    def zero_shot(self) -> str:
        raise NotImplementedError()

    def initialize(self):
        if self.initialize_strategy == ZERO_SHOT:
            self.root = Node(answer=self.zero_shot())
        elif self.initialize_strategy == DUMMY_ANSWER:
            self.root = Node(answer="I don't know.")
        else:
            raise ValueError(f"Invalid initialize strategy: {self.initialize_strategy}")

    def run(self):
        self.initialize()
        for _ in tqdm.tqdm(range(self.max_rollouts)):
            node = self.select_node()
            self.self_evaluate(node)
            child = self.self_refine(node)
            node.add_child(child)
            self.self_evaluate(child)
            self.backpropagate(child)
        return self.get_best_answer()

    def get_best_answer(self):
        to_visit = deque([self.root])
        best_node = self.root
        while to_visit:
            current_node = to_visit.popleft()
            if current_node.Q > best_node.Q:
                best_node = current_node
            to_visit.extend(current_node.children)
        return best_node.answer



class MC_NEST_gpt4o(MC_NEST):
    model: str = llm_config["model"]
    initial_system_prompt: str = llm_config["prompts"]["initial_system_prompt"]
    critic_system_prompt: str = llm_config["prompts"]["critic_system_prompt"]
    refine_system_prompt: str = llm_config["prompts"]["refine_system_prompt"]
    evaluate_system_prompt: str =  llm_config["prompts"]["evaluate_system_prompt"]

    def zero_shot(self) -> str:
        response = openai_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": self.initial_system_prompt,
                },
                {
                    "role": "user",
                    "content": f"<user_prompt>{self.user_prompt}\n</user_prompt><background_information>\n{self.background_information}\n</background_information>",
                },
            ],
            model=llm_config["model"],
            temperature=llm_config["temperature"],
            max_tokens=llm_config["max_tokens"],
        )
        assert response.choices[0].message.content is not None
        return response.choices[0].message.content

    def self_refine(self, node: Node) -> Node:
        critique_response = openai_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": self.critic_system_prompt,
                },
                {
                    "role": "user",
                    "content": "\n\n".join(
                        [
                            f"<background_information>\n{self.background_information}\n</background_information>",
                            f"<current_answer>\n{node.answer}\n</current_answer>",
                        ]
                    ),
                },
            ],
            model=self.model,
            temperature=0.90,
            max_tokens=5000,
        )
        critique = critique_response.choices[0].message.content
        assert critique is not None
        refined_answer_response = openai_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": self.refine_system_prompt,
                },
                {
                    "role": "user",
                    "content": "\n\n".join(
                        [
                            f"<background_information>\n{self.background_information}\n</background_information>",
                            f"<current_answer>\n{node.answer}\n</current_answer>",
                            f"<critique>\n{critique}\n</critique>",
                        ]
                    ),
                },
            ],
            model=self.model,
            temperature=0.90,
            max_tokens=5000,
            response_format={"type": "json_object"},
        )
        refined_answer = RefineResponse.model_validate_json(
            refined_answer_response.choices[0].message.content
        )
        return Node(
            answer=f"**Hypothesis:**\n{refined_answer.hypothesis}\n\n**Explanation:**\n{refined_answer.explanation}",
            parent=node,
        )

    def _evaluate_answer(self, node: Node) -> int:
        messages = [
            {
                "role": "system",
                "content": self.evaluate_system_prompt,
            },
            {
                "role": "user",
                "content": "\n\n".join(
                    [
                        f"<background_information>\n{self.background_information}\n</background_information>",
                        f"<answer>\n{node.answer}\n</answer>",
                    ]
                ),
            },
        ]
        for attempt in range(3):
            try:
                response = openai_chat_completion(
                    messages=messages,
                    model=self.model,
                    temperature=0.90,
                    max_tokens=5000,
                )
                assert response.choices[0].message.content is not None
                return int(response.choices[0].message.content)
            except ValueError:
                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": response.choices[0].message.content,
                        },
                        {
                            "role": "user",
                            "content": "Failed to parse reward as an integer.",
                        },
                    ]
                )
                if attempt == 2:
                    raise Warning("Failed to parse reward after 3 attempts.")

