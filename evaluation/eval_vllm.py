import json
from vllm import LLM, SamplingParams
from tqdm import tqdm
import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Cell Type Annotation Evaluation')
    parser.add_argument('--model_name', default = "Qwen2.5-72B-Instruct", help='Name of the model to use')
    parser.add_argument('--output_path', default= "./results/response.json", help='Path to save full output results')
    parser.add_argument('--dataset_path', default= "./data/cta_scrna_full.json", help='Path to test dataset JSON file')
    return parser.parse_args()

args = parse_arguments()

model_name = args.model_name
metric_path = args.metric_path
output_path = args.output_path
dataset_path = args.dataset_path


llm = LLM(
    model=model_name,
    tensor_parallel_size=4,         
    dtype="half",                   
    gpu_memory_utilization=0.95,    
    enforce_eager=True,                
    trust_remote_code=True
)


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)


sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=4096,
    repetition_penalty=1.1,
    stop_token_ids=[tokenizer.eos_token_id]
)

def messages_to_prompt(messages):
    system_prompt = ""
    chat_history = []
    
    for msg in messages:
        if msg["role"] == "system":
            system_prompt = msg["content"]
        else:
            chat_history.append({"role": msg["role"], "content": msg["content"]})
    

    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=True
        )
    else:

        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        prompt += "\nassistant: "
    
  
    if system_prompt:
        prompt = f"System: {system_prompt}\n\n{prompt}"
    
    return prompt


with open(dataset_path, 'r') as f:
    dataset = json.load(f)


print("Preparing prompts...")
prompts = [messages_to_prompt(item["messages"][:-1]) for item in dataset]


batch_size = 1  
results = []
all_outputs = [] 

for i in tqdm(range(0, len(prompts), batch_size), desc="Processing"):
    batch_prompts = prompts[i:i+batch_size]
    
    try:
        outputs = llm.generate(
            batch_prompts,
            sampling_params,
            use_tqdm=False
        )
        
        all_outputs.extend(outputs)  
        for out in outputs:
            results.append(out.outputs[0].text)  
            
    except Exception as e:
        print(f"batch process {i}-{i+batch_size} fail: {str(e)}")
        results.extend([""] * len(batch_prompts))
        all_outputs.extend([None] * len(batch_prompts)) 


output_data = []
for item, pred in zip(dataset, results):
    output_item = {
        "messages": item["messages"],
        "model_response": pred,
        "ground_truth": item["messages"][-1]["content"]
    }
    output_data.append(output_item)


with open(output_path, 'w') as f:
    json.dump(output_data, f, indent=2)

print("Done")