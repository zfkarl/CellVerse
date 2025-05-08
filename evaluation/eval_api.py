import json
import openai
from openai import OpenAI
import os
from tqdm import tqdm
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Cell Type Annotation Evaluation')
    parser.add_argument('--model_name', default = "gpt-4.1-mini", help='Name of the GPT model to use') 
    parser.add_argument('--output_path', default= "./results/response.json", help='Path to save full output results')
    parser.add_argument('--dataset_path', default= "./data/cta_scrna_full.json", help='Path to test dataset JSON file')
    parser.add_argument('--openai_api_key', default= "", help='OpenAI API key')
    parser.add_argument('--base_url', default= "", help='Base URL for OpenAI API')
    return parser.parse_args()

args = parse_arguments()

# 使用命令行参数替换原来的硬编码值
model_name = args.model_name
output_path = args.output_path
dataset_path = args.dataset_path

os.environ["OPENAI_API_KEY"] = args.openai_api_key
os.environ["BASE_URL"] = args.base_url

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("BASE_URL"))

def predict_cell_type(messages):
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0
        )
        return response.choices[0].message.content

    except Exception as e:
        print(f"API Error: {e}")
        return ""



with open(dataset_path, 'r') as f:
    dataset = json.load(f)


output_data = []
for item in tqdm(dataset, desc="Processing Answers", unit="sample"):  
    messages = item['messages'][:-1] 
    ground_truth = item['messages'][-1]['content']
    
    prediction = predict_cell_type(messages)

    output_item = {
        "messages": item['messages'],
        "model_response": prediction,
        "ground_truth": ground_truth
    }
    output_data.append(output_item)


with open(output_path, 'w') as f:
    json.dump(output_data, f, indent=2)

print("Done")