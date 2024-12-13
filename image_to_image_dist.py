import os
import PIL.Image
import torch
import numpy as np
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
import random
import json
from mmengine.dist import infer_launcher, init_dist, get_world_size, get_rank, barrier, all_gather_object
from mmengine.runner import set_random_seed
import math
import torch.distributed as dist

if __name__ == '__main__':
    # specify the path to the model
    model_path = "/cpfs01/shared/llm_razor/huanghaian/code/xpuyu_vlm/xpuyu/vlm_tools/work_dirs/janus_sft7/hf-1292-of-1292"
    repeat_time = 3
    seed = 1024
    random_choice_num_pre_gpu = 10

    media_root = '/cpfs01/shared/llm_razor/huanghaian/data/SEED-Data/SEED-Data-Edit-Part1-Unsplash/auto_editing/unsplash/images/'
    annotation = '/cpfs01/shared/llm_razor/huanghaian/data/SEED-Data/SEED-Data-Edit-Part1-Unsplash/seed_edit_part1_1210_val.jsonl' # 500

    dist_launcher = infer_launcher()
    init_dist(dist_launcher)
    set_random_seed(seed)
    rank, world_size = get_rank(), get_world_size()

    all_conversation = []
    with open(annotation) as f:
        data = f.readlines()
    data = [json.loads(d) for d in data]
    data = random.choices(data, k=random_choice_num_pre_gpu * world_size)

    if rank == 0:
        os.makedirs('generated_samples', exist_ok=True)
    barrier()

    n_samples = len(data)
    per_rank_samples = math.ceil(n_samples / world_size)
    per_rank_ids = range(per_rank_samples * rank,
                         min(n_samples, per_rank_samples * (rank + 1)))
    print(f'rank: {rank}, per_rank_ids: {per_rank_ids}')

    for ids in per_rank_ids:
        d = data[ids]
        image_path_in = d['conversations'][0]['images'][0]
        image_path = os.path.join(media_root, image_path_in)
        image_path_out = d['conversations'][1]['images'][0]
        image_path_out = os.path.join(media_root, image_path_out)
        input_ = d['conversations'][0]['value']
        input_ = input_.replace('<image>', '<image_placeholder>')
        conversation = [
            {
                "role": "User",
                "content": input_,
                "images": [image_path]
            },
            {"role": "Assistant", "content":  d['conversations'][1]['value'], "images": [image_path_out]},
        ]
        all_conversation.append(conversation)

    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    vl_chat_processor.system_prompt = ''

    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    pred_datas = []

    for k, conversation in enumerate(all_conversation):
        target_text = conversation[1]['content']
        target_image = conversation[1].pop('images')[0]
        target_image = PIL.Image.open(target_image).convert('RGB').resize((384, 384))
        target_image = np.array(target_image)
        conversation[1]['content'] = ''

        print(f' {rank} ====================={k}============================')
        print('input:', conversation[0]['content'])
        pil_images = load_pil_images(conversation)
        prepare_inputs = vl_chat_processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(vl_gpt.device)

        is_image_placehold = False

        for j in range(repeat_time):
            sft_format = prepare_inputs.sft_format[0]
            pred_data = {"inputs": sft_format, 'target': target_text}

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
            print(f' {rank} output:', out_text, '\ntarget', target_text, '=====================')
            pred_data[f'output'] = out_text

            with torch.no_grad():
                image_tokens = torch.cat(image_tokens, dim=1)
                dec = vl_gpt.gen_vision_model.decode_code(image_tokens.to(dtype=torch.int),
                                                          shape=[1, 8, 384 // 16, 384 // 16])
                dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
                dec = np.clip((dec + 1) / 2 * 255, 0, 255)
                visual_img = np.zeros((1, 384, 384, 3), dtype=np.uint8)
                visual_img[:, :, :] = dec

            save_path = os.path.join('generated_samples', "rank{}_{}_img_{}.jpg".format(rank, k, j))
            pred_img = visual_img[0]
            input_image = np.array(pil_images[0].resize((384, 384)))
            save_image = np.concatenate([input_image, target_image, pred_img], axis=1)
            PIL.Image.fromarray(save_image).save(save_path)
            pred_data['file'] = "rank{}_{}_img_{}.jpg".format(rank, k, j)
            pred_datas.append(pred_data)

    pred_datas = all_gather_object(pred_datas)
    pred_datas = [item for sublist in pred_datas for item in sublist]
    if rank == 0:
        with open('generated_samples/pred_datas.jsonl', 'w') as f:
            for item in pred_datas:
                f.write(json.dumps(item) + '\n')
    barrier()
    dist.destroy_process_group()
