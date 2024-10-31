from unsloth import FastLanguageModel
from transformers import StoppingCriteria, StoppingCriteriaList
import torch
from tools import add_numbers, square_number, square_root_number   
import ast
import re

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "akshayballal/phi-3.5-mini-xlam-function-calling",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

FastLanguageModel.for_inference(model);


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_ids:list):
        self.keywords = keywords_ids

    def __call__(self, input_ids: torch.LongTensor, _: torch.FloatTensor, **kwargs) -> bool:
        if input_ids[0][-1] in self.keywords:
            return True
        return False


stop_ids = [17171]
stop_criteria = KeywordsStoppingCriteria(stop_ids)


tools = [add_numbers, square_number, square_root_number] # add the tools to a list


tool_descriptions = []
for tool in tools:
    spec = {
        "name": tool.__name__,
        "description": tool.__doc__.strip(),
        "parameters": [
            {
                "name": param,
                "type": arg.__name__ if hasattr(arg, '__name__') else str(arg),
            } for param, arg in tool.__annotations__.items() if param != 'return'
        ]
    }
    tool_descriptions.append(spec)
tool_descriptions



class Agent:
    def __init__(
        self, system: str = "", function_calling_prompt: str = "", tools=[]
    ) -> None:
        self.system = system
        self.tools = tools
        self.function_calling_prompt = function_calling_prompt
        self.messages: list = []
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message=""):
        if message:
            self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        with model.disable_adapter():  # disable the adapter for thinking and reasoning
            inputs = tokenizer.apply_chat_template(
                self.messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            output = model.generate(
                input_ids=inputs,
                max_new_tokens=128,
                stopping_criteria=StoppingCriteriaList([stop_criteria]),
            )
            return tokenizer.decode(
                output[0][inputs.shape[-1] :], skip_special_tokens=True
            )

    def function_call(self, message):
        inputs = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": self.function_calling_prompt.format(
                        tool_descriptions=tool_descriptions, query=message
                    ),
                }
            ],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        output = model.generate(input_ids=inputs, max_new_tokens=128, temperature=0.0)
        prompt_length = inputs.shape[-1]

        answer = ast.literal_eval(
            tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)
        )[
            0
        ]  # get the output of the function call model as a dictionary
        print(answer)
        tool_output = self.run_tool(answer["name"], **answer["arguments"])
        return tool_output

    def run_tool(self, name, *args, **kwargs):
        for tool in self.tools:
            if tool.__name__ == name:
                return tool(*args, **kwargs)
            

system_prompt = f"""
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions. Stop when you have the Answer.
Your available actions are:

{tools}
Example session:

Question: What is the mass of Earth times 2?
Thought: I need to find the mass of Earth
Action: get_planet_mass: Earth
PAUSE 

Observation: 5.972e24

Thought: I need to multiply this by 2
Action: calculate: 5.972e24 * 2
PAUSE

Observation: 1,1944×10e25

If you have the answer, output it as the Answer.

Answer: \\\\{{1,1944×10e25\\\\}}.
PAUSE
Now it's your turn:

""".strip()

function_calling_prompt = """
You are a helpful assistant. Below are the tools that you have access to.  \n\n### Tools: \n{tool_descriptions} \n\n### Query: \n{query} \n
"""


def loop_agent(agent: Agent, question, max_iterations=5):

    next_prompt = question
    i = 0
    while i < max_iterations:
        result = agent(next_prompt)
        print(result)
        if "Answer:" in result:
            return result

        action = re.findall(r"Action: (.*)", result)
        if action:
            tool_output= agent.function_call(action)
            next_prompt = f"Observation: {tool_output}"
            print(next_prompt)
        else:
            next_prompt = "Observation: tool not found"
        i += 1
    return result


agent = Agent( system=system_prompt, function_calling_prompt=function_calling_prompt, tools=tools)

loop_agent(agent, "what is the square root of the difference between 32^2 and 54");
