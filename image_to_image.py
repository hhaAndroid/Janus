import os
import PIL.Image
import torch
import numpy as np
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
import random
import json

# specify the path to the model
model_path = "/cpfs01/shared/llm_razor/huanghaian/code/xpuyu_vlm/xpuyu/vlm_tools/work_dirs/janus_sft7/hf-1292-of-1292"
repeat_time = 3
seed = 42
use_jsonl = True
media_root = '/cpfs01/shared/llm_razor/huanghaian/data/SEED-Data/SEED-Data-Edit-Part1-Unsplash/auto_editing/unsplash/images/'
annotation = '/cpfs01/shared/llm_razor/huanghaian/data/SEED-Data/SEED-Data-Edit-Part1-Unsplash/seed_edit_part1_1210.jsonl'
random_choice_num = 10

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# 可以设置 conversation 进行单样本测试，也可以指定 jsonl，然后随机选择进行测试
all_conversation = []
if use_jsonl:
    with open(annotation) as f:
        data = f.readlines()
    data = [json.loads(d) for d in data]
    data = random.choices(data, k=random_choice_num)

    for d in data:
        image_path_in = d['conversations'][0]['images'][0]
        image_path = os.path.join(media_root, image_path_in)
        input_ = d['conversations'][0]['value']
        input_ = input_.replace('<image>', '<image_placeholder>')
        conversation = [
            {
                "role": "User",
                "content": input_,
                "images": [image_path]
            },
            {"role": "Assistant", "content": ""},
        ]
        all_conversation.append(conversation)

else:
    conversation = [
        {
            "role": "User",
            "content": "<image_placeholder>\nPlease generate the target image based the source image and editing "
                       "instructions, and provide a description of the target image. Editing instructions: replace the man with a woman",
            "images": ["/cpfs01/shared/llm_razor/huanghaian/code/photo-1579661804513-ca6142cf6c37_0.jpg"]
        },
        {"role": "Assistant", "content": ""},
    ]
    all_conversation.append(conversation)

vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
vl_chat_processor.system_prompt = ''
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()


for k, conversation in enumerate(all_conversation):
    print(f'====================={k}============================')
    print('input:', conversation[0]['content'], '====')
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(vl_gpt.device)

    is_image_placehold = False

    for j in range(repeat_time):
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        is_image_gen = False
        text_tokens = []
        image_tokens = []
        for i in range(576 + 100):
            with torch.no_grad():
                outputs = vl_gpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True,
                                                      past_key_values=outputs.past_key_values if i != 0 else None)
            hidden_states = outputs.last_hidden_state
            if is_image_gen:
                if is_image_placehold:
                    next_token = torch.tensor([0]).reshape(1, 1).cuda()
                    is_image_placehold = False
                else:
                    logits = vl_gpt.gen_head(hidden_states[:, -1, :])
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
            else:
                logits = vl_gpt.language_model.lm_head(hidden_states[:, -1, :])
                probs = torch.softmax(logits, dim=-1)

                # next_token1 = torch.argmax(probs, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                next_token = next_token[0]

            if is_image_gen:
                inputs_embeds = vl_gpt.prepare_gen_img_embeds(next_token)
                image_tokens.append(next_token)
            else:
                inputs_embeds = vl_gpt.language_model.get_input_embeddings()(next_token)
                inputs_embeds = inputs_embeds.unsqueeze(dim=1)
                text_tokens.append(next_token.item())

            if next_token == vl_chat_processor.image_start_id:
                is_image_gen = True
                # 如果训练没有占位符，则需要把这个强制设置为 False
                is_image_placehold = True

            if len(image_tokens) == 576:
                break

        out_text = tokenizer.decode(text_tokens, skip_special_tokens=True)
        print('output:', out_text, '=====================')

        with torch.no_grad():
            image_tokens = torch.cat(image_tokens, dim=1)
            dec = vl_gpt.gen_vision_model.decode_code(image_tokens.to(dtype=torch.int), shape=[1, 8, 384 // 16, 384 // 16])
            dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
            dec = np.clip((dec + 1) / 2 * 255, 0, 255)
            visual_img = np.zeros((1, 384, 384, 3), dtype=np.uint8)
            visual_img[:, :, :] = dec

        os.makedirs('generated_samples', exist_ok=True)
        save_path = os.path.join('generated_samples', "{}_img_{}.jpg".format(k, j))
        PIL.Image.fromarray(visual_img[0]).save(save_path)
