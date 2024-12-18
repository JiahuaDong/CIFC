import torch
from diffusers import DPMSolverMultistepScheduler
from lib.pipelines.pipeline_edlora import EDLoRAPipeline, StableDiffusionPipeline
from lib.utils.convert_edlora_to_diffusers import convert_edlora, load_new_concept, merge_lora_into_weight
import argparse
import os
import json
from lib.models.edlora import EDLoRA_FusionAttnProcessor, MultiLoRALinearLayer

def parse_new_concepts(concept_cfg):
    with open(concept_cfg, 'r') as f:
        concept_list = json.load(f)

    model_paths = [concept['lora_path'] for concept in concept_list]

    embedding_1_list = []
    embedding_2_list = []
    unet_list = []
    for model_path in model_paths:
        model = torch.load(model_path)['params']

        if 'new_concept_embedding' in model and len(
                model['new_concept_embedding']) != 0:
            embedding_1_list.append(model['new_concept_embedding'])
        else:
            embedding_1_list.append(None)
        if 'new_concept_embedding_2' in model and len(
                model['new_concept_embedding_2']) != 0:
            embedding_2_list.append(model['new_concept_embedding_2'])
        else:
            embedding_2_list.append(None)
        if 'unet' in model and len(model['unet']) != 0:
            unet_list.append(model['unet'])
        else:
            unet_list.append(None)

    return embedding_1_list, concept_list, unet_list

def merge_new_concepts_(embedding_list, concept_list, tokenizer, text_encoder):
    def add_new_concept(concept_name, embedding):
        new_token_names = [
            f'<new{start_idx + layer_id}>'
            for layer_id in range(NUM_CROSS_ATTENTION_LAYERS)
        ]
        num_added_tokens = tokenizer.add_tokens(new_token_names)
        assert num_added_tokens == NUM_CROSS_ATTENTION_LAYERS
        new_token_ids = [
            tokenizer.convert_tokens_to_ids(token_name)
            for token_name in new_token_names
        ]

        text_encoder.resize_token_embeddings(len(tokenizer))
        token_embeds = text_encoder.get_input_embeddings().weight.data

        token_embeds[new_token_ids] = token_embeds[new_token_ids].copy_(
            embedding[concept_name])

        embedding_features.update({concept_name: embedding[concept_name]})

        return start_idx + NUM_CROSS_ATTENTION_LAYERS, new_token_ids, new_token_names

    embedding_features = {}
    new_concept_cfg = {}

    start_idx = 0

    NUM_CROSS_ATTENTION_LAYERS = 16

    for idx, (embedding,
              concept) in enumerate(zip(embedding_list, concept_list)):
        concept_names = concept['replace_mapping'].split(' ')

        for concept_name in concept_names:
            if not concept_name.startswith('<'):
                continue
            else:
                assert concept_name in embedding, 'check the config, the provide concept name is not in the lora model'
            start_idx, new_token_ids, new_token_names = add_new_concept(
                concept_name, embedding)
            new_concept_cfg.update({
                concept_name: {
                    'concept_token_names': new_token_names
                }
            })
    return embedding_features, new_concept_cfg

def bind_concept_prompt(prompts, new_concept_cfg):
    if isinstance(prompts, str):
        prompts = [prompts]
    new_prompts = []
    for prompt in prompts:
        prompt = [prompt] * 16
        for concept_name, new_token_cfg in new_concept_cfg.items():
            prompt = [
                p.replace(concept_name, new_name) for p, new_name in zip(prompt, new_token_cfg['concept_token_names'])
            ]
        new_prompts.extend(prompt)
    return new_prompts

def encode_prompt(prompts, new_concept_cfg, tokenizer, text_encoder):
    
    prompt_embeds_list = []

    prompt = bind_concept_prompt(prompts, new_concept_cfg)
    
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids

    prompt_embeds = text_encoder(text_input_ids.to(text_encoder.device))[0]

    return prompt_embeds

def revise_edlora_unet_fusionattention_forward(unet, token_embedding, task_indexes):
    def change_forward(unet, count, token_embedding, task_indexes):
        for name, layer in unet.named_children():
            if layer.__class__.__name__ == 'Attention' and 'attn2' in name:
                layer.set_processor(EDLoRA_FusionAttnProcessor(count, token_embedding[count], task_indexes))
                count += 1
            else:
                count = change_forward(layer, count, token_embedding, task_indexes)
        return count

    # use this to ensure the order
    cross_attention_idx = change_forward(unet.down_blocks, 0, token_embedding, task_indexes)
    cross_attention_idx = change_forward(unet.mid_block, cross_attention_idx, token_embedding, task_indexes)
    cross_attention_idx = change_forward(unet.up_blocks, cross_attention_idx, token_embedding, task_indexes)
    print(f'Number of attention layer registered {cross_attention_idx}')
    
    return cross_attention_idx


def main(args):
    enable_edlora = True  # True for edlora, False for lora
    alpha=args.alpha

    embedding_1_list, concept_list, unet_list = parse_new_concepts(args.concept_cfg)
    with open(args.concept_cfg, 'r') as f:
        concept_list = json.load(f)

    
    pipeclass = EDLoRAPipeline if enable_edlora else StableDiffusionPipeline
    pipe = pipeclass.from_pretrained(args.pretrained_path, 
                                    scheduler=DPMSolverMultistepScheduler.from_pretrained(args.pretrained_path, 
                                    torch_dtype=torch.float16,
                                    subfolder='scheduler')).to('cuda')


    _, new_concept_cfg = merge_new_concepts_(embedding_1_list, concept_list, pipe.tokenizer, pipe.text_encoder)

    for name, module in pipe.unet.named_modules():
        if name.find('attn2')!=-1:
            if module.__class__.__name__ == 'Attention':
                for child_name, child_module in module.named_modules():
                    if child_module.__class__.__name__ == 'Linear' or (child_module.__class__.__name__ == 'Conv2d' and child_module.kernel_size == (1, 1)):
                        lora_module = MultiLoRALinearLayer(name + '.' + child_name, child_module, unet_list, alpha=alpha, method=args.method).to('cuda')

    prompt = []
    task_indexes= [1]
    match_dict = {}
    with torch.no_grad():
        for concept in concept_list:
            match_dict.update({concept["replace_mapping"]: concept["concept_name"]})
            prompt += concept["replace_mapping"].split()
            task_indexes.append(task_indexes[-1] + len(concept["replace_mapping"].split()))
        prompt=' '.join(prompt)
        token_embedding =encode_prompt(prompt, new_concept_cfg, pipe.tokenizer, pipe.text_encoder).cuda()

    revise_edlora_unet_fusionattention_forward(pipe.unet, token_embedding, task_indexes)

    pipe.set_new_concept_cfg(new_concept_cfg)

    if args.method == 'composer':
        lora_num = len(unet_list)
    else:
        lora_num = 1

    # pipe.load_lora_weights(ckpt)
    torch.cuda.empty_cache()
    text_path = args.text_path
    output_path= os.path.join(args.output_path, args.method, args.replace_prompt)
    negative_prompt = 'longbody, lowres, bad anatomy, bad hands, extra digit, fewer digits, cropped, worst quality, low quality'
    count = 0
    image_info_list = []
    with open(text_path, "r") as f:
        data = f.readlines()

        data = [[prompt.strip().replace('<TOK>', args.replace_prompt)]*args.batch_size for prompt in data if prompt.strip()] 

    for prompt in data:
        evluation_text = prompt[0].replace('\n', '')
        for replace_prompt in match_dict.keys():
            evluation_text = evluation_text.replace(replace_prompt, match_dict[replace_prompt])
        generator=torch.Generator('cuda').manual_seed(2024)
        name = prompt[0][:50]
        os.makedirs(f'{output_path}/samples', exist_ok=True)
        all_images=[]
        for i in range(args.iter):
            images = pipe(prompt, negative_prompt=[negative_prompt]*args.batch_size, num_inference_steps=50, guidance_scale=7.5, generator=generator, lora_num = lora_num).images
            all_images += images

        for i, im in enumerate(all_images):
            im.save(f'{output_path}/samples/{count}.jpg')
            image_info_list.append({

                f'{count}': evluation_text,
            })            
            count+=1

    with open(os.path.join(output_path, 'prompts.json'), 'w') as f:

        json.dump(image_info_list, f)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--concept_cfg', type=str, default='./datasets/data_cfgs/task10.json')
    parser.add_argument('--text_path', type=str, default='./datasets/evaluation_prompts/test_pet.txt')
    parser.add_argument('--pretrained_path', type=str, default='your model path of SD1.5')
    parser.add_argument('--output_path', type=str, default='./results_08')
    parser.add_argument('--replace_prompt', type=str, default='<dog1> <dog2>')
    parser.add_argument('--method', type=str, default='ours')
    parser.add_argument('--alpha', type=float, default=0.8)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--iter', type=int, default=10)

    args = parser.parse_args()

    main(args)
