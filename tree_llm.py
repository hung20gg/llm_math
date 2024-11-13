from llm.llm_utils import get_code_from_text_response, get_json_from_text_response
from llm.llm.abstract import LLM
import utils
from typing import List
import numpy as np
import re
import copy
from collections import deque
import random
from tqdm import tqdm

SEARCH_POLICY = 'sampling'
MAX_CHILD = 2

# How the algorithm work
# First, we will generate a weak answer
# Then, we will use a Reward Model to generate reward based on the solution
# Then, we will generate a new answer based on the given answer
# Add the new solution to the tree and update Q value
# Use mcts algo to find the next selected weak answer
# Do it until end of search limit or no other node found

class ReasoningNode:
    def __init__(self, content: str, critique: str = None):
        self.content: str = content 
        self.critique: str = critique
        self.parent: ReasoningNode = None 
        self.children: List[ReasoningNode] = []
        self.rank: int = 0
        self.reward: List[float] = []
        self.Q: float = 0
        self.N: int = 0
        
    def add_child(self, node) -> None:
        node.rank = self.rank + 1
        self.children.append(node)
        node.parent = self
        
    def add_critique(self, critique:str) -> None:
        self.critique = critique
        
    def add_reward(self, reward: float):
        self.reward.append(reward)
        
        # New Q value score
        avg_reward = np.mean(self.reward)
        min_reward = np.min(self.reward)
        
        self.Q = (min_reward + avg_reward)/2
        
    def get_solution(self, parent_solution = False) -> str:
        if parent_solution:
            reasoning_chain = []
            node = copy.deepcopy(self)
            while node.parent:
                reasoning_chain.append((node.content, node.critique))
                node = node.parent
            reasoning_chain = reasoning_chain[::-1]
            
            answer = ""
            for i, (content, critique) in enumerate(reasoning_chain):
                if i == 0:
                    answer += "## First solution"
                    answer += "\n\n".join([
                        f"<previous_answer>\n{content}\n</previous_answer>",
                        f"<critique>\n{critique}\n</critique>"
                    ])  
                    
                elif i == len(reasoning_chain) - 1:
                    answer + "\n\n## Current solution"
                    answer += f"\n<current_answer>\n{content}\n</current_answer>"
                else:
                    answer += f"\n\n## Previous solution number {i+1} "
                    answer += "\n\n".join([
                        f"<previous_answer>\n{content}\n</previous_answer>",
                        f"<critique>\n{critique}\n</critique>"
                    ])  
            
            return answer    
            
        else:
            return f"<current_answer>{self.content}<current_answer>"


def reward_llm(llm: LLM, task: str, node: ReasoningNode, verbose: bool = False) -> int:
    system_prompt = utils.read_file_without_comments('prompt/tree/system_prompt_reward.txt')
        
    prompt = f"""
This is the problem that someone have been working on:
<problem>
{task}    
</problem>

<current_answer>
{node.get_solution(False)}
</current_answer>

<task>
Thinking step-by-step first and then provide an integer score from -100 to 100 to evaluate the quality of the given solution. Using strict standards.
Do not give a full score.
</task>

Finally, return the score in JSON format.
    ```
        {{
            "score": 5
        }}
    ```
"""
    
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    response = llm(messages)
    if verbose:
        print(f"Reward:\n{response}\n====================")
    return int(get_json_from_text_response(response, new_method=True)['score'])



def self_refine(llm: LLM, node: ReasoningNode, task: str, use_parent_solution: bool = False, verbose: bool = False, critic: LLM = None) -> ReasoningNode:
    
    # Critic
    answer = node.get_solution(use_parent_solution)
    
    messages = [
        {
            "role":"system",
            "content":utils.read_file_without_comments('prompt/tree/system_prompt_critic.txt')
        },
        {
            "role":"user",
            "content": '\n\n'.join([
                f"<problem>{task}</problem>",
                answer,
            ])
        }
    ]
    
    if critic is not None:
        critique = critic(messages)
    else:
        critique = llm(messages)
        
    if verbose:
        print(f"Critique:\n{critique}\n====================")
    
    # Refine
    messages = [
        {
            "role":"system",
            "content": utils.read_file_without_comments('prompt/tree/system_prompt_refine.txt')
        },
        {
            "role":"user",
            "content": '\n\n'.join([
                f"<problem>{task}</problem>",
                answer,
                f"<critique>{critique}</critique>"
            ])
        }
    ]
    
    refine = llm(messages)
    if verbose:
        print(f"Refine:\n{refine}\n====================")
    
    node.add_critique(critique)
    return ReasoningNode(content=refine)
    
    
# Check the UCB value
def UCB(node: ReasoningNode, c = 1, epsilon = 1e-6) -> float: # AlphaGo UCB
    if node.parent is not None:
        ucb = node.Q + c * np.sqrt(2 * np.log(node.parent.N +1) / (node.N + epsilon))
        return ucb
    else:
        return 10000
    

def backward_Q_value(node: ReasoningNode) -> None:
    # Backward update the Q value
    parent = node.parent
    while parent is not None:
        parent.N += 1
        maxQ = max([child.Q for child in parent.children])
        
        # Check the formula 
        parent.Q = (parent.Q + maxQ)/2
        parent = parent.parent

def fully_expanded(node: ReasoningNode, max_child: int = MAX_CHILD) -> bool:
    
    flag = False
    # If the children of the node is greater than max child
    if len(node.children) >= max_child:
        flag |= True
        
    # If Q value of child > current node
    for child in node.children:
        if child.Q > node.Q:
            flag |= True 
            break
        
    return flag
    

def select_node(node: ReasoningNode, search_policy: str = 'sampling') -> ReasoningNode:
    
    # Get the root node
    while node.parent is not None:
        node = node.parent
        
    # Select the candidate note for search
    candidate = []
    bfs_node = deque([node])
    
    while bfs_node:
        current_node = bfs_node.popleft()
        if not fully_expanded(current_node):
            candidate.append(current_node)
        bfs_node.extend(current_node.children)
        
    if not candidate:
        return node 
    
    if search_policy.lower() == 'greedy': # Get node with highest UCB score
        return max(candidate, key= lambda x: UCB(x))
    
    elif search_policy.lower() == 'sampling':
        ucbs = [UCB(x) for x in candidate]
        choice = random.choices(
            range(len(candidate)), weights=ucbs, k=1
        )[0]
        return candidate[choice]
    else:
        raise ValueError("Invalid Search Policy")
    
    
def travel(llm, task, begin = "zero-shot", max_search = 20, verbose: bool = False) -> ReasoningNode:
    if begin != "zero-shot":
        begin = ReasoningNode("I dont know")
        
    else:
        messages = [
            {
                "role":"system",
                "content":utils.read_file_without_comments('prompt/system_prompt.txt')
            },
            {
                "role":"user",
                "content":f"<problem>{task}</problem>"
            }
        ]
        
        weak_answer = llm(messages)
        begin = ReasoningNode(weak_answer)

    for _ in tqdm(range(max_search)):
        node = select_node(begin)
        reward = reward_llm(llm, task, node, verbose=verbose)
        node.add_reward(reward)
        child = self_refine(llm, node, task, verbose=verbose)
        node.add_child(child)
        reward = reward_llm(llm, task, child, verbose=verbose)
        child.add_reward(reward)
        backward_Q_value(child)
        
    return begin
    