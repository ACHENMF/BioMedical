import os
from tqdm import tqdm
from openai import OpenAI
import json



OPENAI_KEY = ""


client = OpenAI(api_key=OPENAI_KEY)




'''Prompt3'''
def step1_prompt(question, answer):
    context = f"Given a pair of question and answer: \n'question': {question} and 'answer': {answer}, generate a concise external knowledge prompt by thinking step by step in the following steps.\n"
    prompt = '''Step 1: Focus on the core aspect of the question. What is the main attribute or issue being asked?\
        Step 2: Analyze the answer. What key reasoning supports it? Identify any possible ambiguities or areas needing further clarification to align it with the given question.
        Step 3: Provide focused external knowledge that directly supports the answer, without deviating into unnecessary detail. Ensure this knowledge fills in any gaps in understanding while remaining tightly related to the question and answer. Importantly, ensure that the knowledge you generate does not include words or phrases present in the answer. 
        Step 4: Establish a direct causal connection between the knowledge and the answer. Explain how the answer follows logically from the question, based on the background provided.
        Generate a concise external knowledge prompt that connects the question and answer logically, helping to clarify reasoning. The prompt should avoid excessive elaboration, focusing instead on supporting the core reasoning. Please output in the format as follow and do not output other words:
              Prompt: the external knowledge prompt you generated
              Reason: the reason why you generated this information'''
    prompt_1 = context + prompt
    return prompt_1


# 定义生成 判断结果 的函数
def generation(conversation, prompt_text):
    conversation.append(
        {'role': 'user', "content": prompt_text}
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=conversation
    )
    conversation.append(
        {"role": "assistant",
         "content": response.choices[0].message.content.strip()}
    )
    result = response.choices[0].message.content.replace('\n', ' ').strip()
    return conversation, result


# 定义处理链式推理的函数
def process_chain_of_thought(data):
    system_role = {'role': 'system', "content": "You are an expert in medical field."}
    
    question, answer = data['question'], data['answer']
    
    conversation = [system_role]

    # 第一步
    prompt1 = step1_prompt(question, answer)
    conversation, r1 = generation(conversation, prompt1)
    
 # 返回生成的 prompt 结果
    final_prompt = r1
    
    return final_prompt

def process_json_files_line_by_line(input_path, output_path):
    for filename in ['dev.jsonl', 'test.jsonl', 'train.jsonl']:
        input_file = os.path.join(input_path, filename)
        output_file = os.path.join(output_path, filename)
        name_prefix = filename.split('.')[0]  # 取文件名的前缀部分
        counter_file_path = os.path.join(output_path, f'{name_prefix}_counter.txt')

        # 计算文件总行数
        with open(input_file, 'r', encoding='utf-8') as infile:
            total_lines = sum(1 for _ in infile)

        print(f"Processing {filename} with {total_lines} lines...")

        if not os.path.exists(counter_file_path):
            with open(counter_file_path, 'w', encoding='utf-8') as f:
                f.write('0')

        # 打开输入文件和输出文件
        with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'a', encoding='utf-8') as outfile:
            i = 0
            for line in tqdm(infile, total=total_lines, dynamic_ncols=True, leave=True, desc=f"Processing {filename}"):
                i += 1
                with open(counter_file_path, 'r', encoding='utf8') as f:
                    counter = f.read().strip()
                if i <= int(counter):
                    continue
                data = json.loads(line.strip())  # 逐行读取并解析 JSON 对象

                # 生成 prompt
                final_prompt = process_chain_of_thought(data)

                # 提取 'Prompt' 部分
                prompt_start_index = final_prompt.find("Prompt:") + len("Prompt:")
                prompt_end_index = final_prompt.find("Reason:")  # Find where 'Reason:' starts
                prompt_only = final_prompt[prompt_start_index:prompt_end_index].strip()  # Extract the prompt text

                # 更新 JSON 数据
                data['prompt'] = prompt_only

                # 写入输出文件
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                outfile.flush()  # 确保实时写入
                with open(counter_file_path, 'w', encoding='utf8') as f:
                    f.write(str(i))

dataset = 'medqa'

# 基础目录设置
base_input_dir = f"D:/postgraduate/projects/czc/task3/{dataset}"
base_output_dir = f"D:/postgraduate/projects/czc/task3/output/{dataset}"


input_dir = base_input_dir
output_dir = base_output_dir

# 处理 JSON 文件
process_json_files_line_by_line(input_dir, output_dir)