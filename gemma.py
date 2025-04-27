import torch
from transformers import pipeline
import json
from tqdm import tqdm
import os

def extract_sentence(sentence):
    # 检查原始句子是否包含所需的部分
    if "Original image:" in sentence and "Edited image:" in sentence:
        # 分割句子以提取内容
        original_part = sentence.split("Original image:")[1].split("Edited image:")[0].strip()
        edited_part = sentence.split("Edited image:")[1].strip()
        
        # 打印结果
        return original_part, edited_part
    else:
        print("句子格式错误")

# model
# model_id = "princeton-nlp/gemma-2-9b-it-SimPO"
model_id = "/network_space/server128/shared/zhuoying/cache/hub/models--princeton-nlp--gemma-2-9b-it-SimPO/snapshots/8c87091f412e3aa6f74f66bd86c57fb81cbc3fde"

generator = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)

# prefix
prefix = """
Given an instruction for image editing, output the parts of the original image that should be modified and the modified parts in the edited image compared to the original image. If there are no parts that should be modified, output None. Don't output the reasons.
Here are some examples:
1. Instruction: "Replace the bike with bear." Output: Original image: bike. Edited image: bear.
2. Instruction: "Add a bear." Output: Original image: None. Edited image: bear.
3. Instruction: "Remove the bike." Output: Original image: bike. Edited image: None.
4. Instruction: "Change the image to a cartoon style." Output: Original image: all. Edited image: all.
Here is the instruction: """
# prefix = """
# Given a source caption that describes the original image and a target caption for the modified image, determine whether the type of modification is a change in the style of the image. Answer "yes" or "no".
# Here are some examples:
# 1. Original image caption: a christmas tree is reflected in a mirror on the wall. Edited image caption: a christmas tree is reflected in a mirror on the wall, Kyoto animation. Output: yes.
# 2. Original image caption: the sun is shining through the trees. Edited image caption: the sun is shining through the palm trees. Output: no.
# 3. Original image caption: a colorful tent on the shore of a body of water. Edited image caption: a colorful tent on the shore of a body of water, Manga style. Output: yes.
# Here are the captions to be analyzed:  """
# prefix = """
# Given an instruction for image editing, please determine whether it is of the type "adding object", and output 'yes' or 'no'. Here are some examples:
# Instruction: put a beauty queen seated on top of bus. Output: yes.
# Instruction: replace the cat with a cat with better resolution. Output: no.
# Instruction: turn the wine into absinthe. Output: no.
# Instruction: let the tie be yellow. Output: no.
# Here is the instruction to be determined: """
# prefix = """
# Given an editing instruction, please determine its editing type. The available editing types are: add, remove, replace, change attribute, other.  Here are some examples:
# 1. Instruction: put a beauty queen seated on top of bus. Output: add.
# 2. Instruction: let the tie be yellow. Output: change attribute.
# 3. Instruction: turn the wine into absinthe. Output: replace.
# 4. Instruction: remove the glasses of juice. Output: remove.
# Here is the instruction to be determined: """
# prefix = """
# Given an editing instruction of adding an object, please provide the object that is being added to. Here are some examples:
# 1. Instruction: put a car on the screen of the laptop. Output: screen of the laptop.
# 2. Instruction: let a pizza be inside the microwave. Output: microwave.
# 3. Instruction: What if the cat has glasses. Output: cat.
# 4. Instruction: Add a gas pump. Output: None.

# Please output only the objects; do not output anything else and do not use bold.
# Here is the instruction to be analyzed: """

# prefix = """
# Given an editing instruction of adding an object, please provide the added object. Here are some examples:
# 1. Instruction: put a car on the screen of the laptop. Output: car.
# 2. Instruction: let a pizza be inside the microwave. Output: pizza.
# 3. Instruction: What if the cat has glasses. Output: glasses.
# 4. Instruction: Add a gas pump. Output: gas pump.

# Please output only the objects; do not output anything else and do not use bold.
# Here is the instruction to be analyzed:"""
# prefix = """
# Given an instruction for image editing, output what size changes of the objects are involved, such as bigger, taller. If there are no size changes involved, output None. Don't output the reasons.
# Here are some examples:
# 1. Instruction: "give the bear bigger claws." Output: bigger.
# 2. Instruction: "Make the vase thinner." Output: thinner.
# 3. Instruction: "Remove the bike." Output: None.
# 4. Instruction: "Let's add a cowboy hat to the giraffe." Output: None.
# Here is the instruction: 
# """
# prefix = """
# Given an instruction for image editing and edited area in the original and edited image, output the positional relationship between the edited area in the edited image and the edited area in the original image. The optional positional relationships are: left, right, above, below, inside, unchanged, None.
# Here are some examples:
# 1. Instruction: "put a car on the screen of the laptop". Edited area in original image: screen of the laptop. Edited area in edited image: car. Output: inside.
# 2. Instruction: "Let a horse stand on the grass". Edited area in original image: grass. Edited area in edited image: horse. Output: None.
# 3. Instruction: "put a bird on top of the tower". Edited area in original image: tower. Edited area in edited image: bird. Output: above.
# 4. Instruction: "replace the cat with a dog". Edited area in original image: cat. Edited area in edited image: dog. Output: unchanged.
# Here is the case to be analyzed: 
# """


json_files = ['mgie_metadata.json']

for json_file in json_files:
    # load data
    # json_file = os.path.join('/network_space/server128/shared/zhuoying/data/MyData/metadata/', json_file)
    json_file = '/network_space/server128/shared/zhuoying/data/MyData/user_study/sample_test/style_meta.json'
    with open(json_file, 'r') as f:
        metadata = json.load(f)

    cnt = 0

    for i in tqdm(range(len(metadata))):
        # print(i)
        data = metadata[i]
        if 'LLM' not in data:
            data['LLM'] = {}
        # if 'add' in data['adding']:
        #     cnt += 1
            # if cnt > 30:
            #     break

        # instruction = f"Original image caption: {data['source_caption']} Edited image caption: {data['target_caption']}"
        instruction = f"{data['instruction']}"
        # instruction = f"Instruction: {data['instruction']} Edited area in original image: {data['LLM']['gemma_origin']} Edited area in edited image: {data['LLM']['gemma_edited']}"
        # Instruction: "replace the cat with a dog". Edited area in original image: cat. Edited area in edited image: dog.
        content = prefix + instruction
        # print(f"content: {content}")

        outputs = generator([{"role": "user", "content": content}],
                            do_sample=False,
                            eos_token_id=[generator.tokenizer.convert_tokens_to_ids("<end_of_turn>"), generator.tokenizer.eos_token_id],
                            max_new_tokens=200)
        # print(f"out: {outputs[0]['generated_text'][1]['content']}\n")
        # data["added_object"] = outputs[0]['generated_text'][1]['content'].lower()
        original_part, edited_part = extract_sentence(outputs[0]['generated_text'][1]['content'])
        
        data['LLM']["gemma_origin"] = original_part
        data['LLM']["gemma_edited"] = edited_part

    with open(json_file, 'w') as f:
        json.dump(metadata, f, indent=4)