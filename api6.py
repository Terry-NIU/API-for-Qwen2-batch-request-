from fastapi import FastAPI, Request, HTTPException
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn
import json
import datetime
import torch

# 设置设备参数
DEVICE = "cuda"  # 使用CUDA
DEVICE_ID = "0"  # CUDA设备ID，如果未设置则为空
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE  # 组合CUDA设备信息

# 清理GPU内存函数
def torch_gc():
    if torch.cuda.is_available():  # 检查是否可用CUDA
        with torch.cuda.device(CUDA_DEVICE):  # 指定CUDA设备
            torch.cuda.empty_cache()  # 清空CUDA缓存
            torch.cuda.ipc_collect()  # 收集CUDA内存碎片

# 创建FastAPI应用
app = FastAPI()

# 处理POST请求的端点
@app.post("/")
async def create_item(request: Request):
    global model, tokenizer  # 声明全局变量以便在函数内部使用模型和分词器
    try:
        json_post_raw = await request.json()  # 获取POST请求的JSON数据
        if not isinstance(json_post_raw, list):
            raise HTTPException(status_code=400, detail="JSON data must be a list of objects")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")

    responses = []
    for item in json_post_raw:
        instruction = item.get('instruction')
        prompt = item.get('input')
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt input is missing in one of the provided list items")

        messages = [
            {"role": "system",
             "content": f"请根据以下评分标准为这篇英语作文评分：1. 语言准确性（grammar and spelling），2. 内容相关性（relevance to the topic），3. 组织结构（cohesion and coherence）。本题满分为20分，评分等级为：0-19分（0分最低，19分最高）。请直接给出分数，无需分析。本题满分为20分，成绩分为六个档次。17-19分：切题。表达思想清楚，文字通顺、连贯，基本上无语言错误，仅有个别小错。15-17分：切题。表达思想清楚，文字较连贯，但有少量语言错误。13-15分：基本切题。有些地方表达思想不够清楚，文字勉强连贯;语言错误相当多，其中有一些是严重错误。11-13分：基本切题。表达思想不清楚，连贯性差。有较多的严重语言错误。8-11分：条理不清，思路紊乱，语言支离破碎或大部分句子均有错误，且多数为严重错误。0-8分：未作答，或只有几个孤立的词，或作文与主题毫不相关。作文题目: {instruction}"},
            {"role": "user", "content": prompt}
        ]

        # 调用模型进行对话生成
        input_ids = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([input_ids], return_tensors="pt").to('cuda')
        generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # 构建响应项，只保留input和response
        responses.append({
            "input": prompt,
            "response": response
        })

    torch_gc()  # 执行GPU内存清理
    return {"responses": responses, "status": 200}

# 主函数入口
if __name__ == '__main__':
    # 加载预训练的分词器和模型
    model_name_or_path = '/home/nbw/LLaMA-Factory/export/train7'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", torch_dtype=torch.bfloat16)

    # 启动FastAPI应用
    # 用6006端口可以将autodl的端口映射到本地，从而在本地使用api
    uvicorn.run(app, host='0.0.0.0', port=6006, workers=1)
