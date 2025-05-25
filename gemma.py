import torch
from transformers import pipeline
import json
from tqdm import tqdm
from bpm_utils.prompt import (
    PARSE_INSTRUCTION_PROMPT,
    EDIT_TYPE_PROMPT,
    BEING_ADDED_TO_OBJECT_PROMPT,
    ADDED_OBJECT_PROMPT,
    SIZE_PROMPT,
    POSITION_PROMPT
)

def extract_sentence(sentence):
    # Check if the original sentence contains the required parts
    if "Original image:" in sentence and "Edited image:" in sentence:
        # Split the sentence to extract content
        original_part = sentence.split("Original image:")[1].split("Edited image:")[0].strip()
        edited_part = sentence.split("Edited image:")[1].strip()
        
        return original_part.lower(), edited_part.lower()
    else:
        print("Incorrect sentence format")

def parse_instruction(instruction, generator):
    # parse instruction
    content = PARSE_INSTRUCTION_PROMPT + instruction
    outputs = generator([{"role": "user", "content": content}],
                        do_sample=False,
                        eos_token_id=[generator.tokenizer.convert_tokens_to_ids("<end_of_turn>"), generator.tokenizer.eos_token_id],
                        max_new_tokens=300)
    original_part, edited_part = extract_sentence(outputs[0]['generated_text'][1]['content'])
    
    # judge the type of instruction
    content = EDIT_TYPE_PROMPT + instruction
    outputs = generator([{"role": "user", "content": content}],
                        do_sample=False,
                        eos_token_id=[generator.tokenizer.convert_tokens_to_ids("<end_of_turn>"), generator.tokenizer.eos_token_id],
                        max_new_tokens=300)
    edit_type = outputs[0]['generated_text'][1]['content'].lower()

    # judge the object being added to
    being_added = ""
    if "none" in original_part:
        content = BEING_ADDED_TO_OBJECT_PROMPT + instruction
        outputs = generator([{"role": "user", "content": content}],
                            do_sample=False,
                            eos_token_id=[generator.tokenizer.convert_tokens_to_ids("<end_of_turn>"), generator.tokenizer.eos_token_id],
                            max_new_tokens=300)
        being_added = outputs[0]['generated_text'][1]['content'].lower()
    
    # judge the added object
    added_object = ""
    if "none" in original_part:
        content = ADDED_OBJECT_PROMPT + instruction
        outputs = generator([{"role": "user", "content": content}],
                            do_sample=False,
                            eos_token_id=[generator.tokenizer.convert_tokens_to_ids("<end_of_turn>"), generator.tokenizer.eos_token_id],
                            max_new_tokens=300)
        added_object = outputs[0]['generated_text'][1]['content'].lower()

    # judge the size
    content = SIZE_PROMPT + instruction
    outputs = generator([{"role": "user", "content": content}],
                        do_sample=False,
                        eos_token_id=[generator.tokenizer.convert_tokens_to_ids("<end_of_turn>"), generator.tokenizer.eos_token_id],
                        max_new_tokens=300)
    size = outputs[0]['generated_text'][1]['content'].lower()

    # judge the position
    content = POSITION_PROMPT + instruction
    outputs = generator([{"role": "user", "content": content}],
                        do_sample=False,
                        eos_token_id=[generator.tokenizer.convert_tokens_to_ids("<end_of_turn>"), generator.tokenizer.eos_token_id],
                        max_new_tokens=300)
    position = outputs[0]['generated_text'][1]['content'].lower()
    return original_part, edited_part, edit_type, being_added, added_object, size, position



if __name__ == "__main__":
    model_id = "princeton-nlp/gemma-2-9b-it-SimPO"

    generator = pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device="cuda",
    )

    json_file = './data/sample_test/style_meta.json'
    with open(json_file, 'r') as f:
        metadata = json.load(f)
    
    for entry in tqdm(metadata):
        instruction = entry['instruction']
        original_part, edited_part, edit_type, being_added, added_object, size, position = parse_instruction(instruction, generator)
        
        # Update the entry with the parsed information
        entry['original_part'] = original_part
        entry['edited_part'] = edited_part
        entry['edit_type'] = edit_type
        entry['being_added'] = being_added
        entry['added_object'] = added_object
        entry['size'] = size
        entry['position'] = position
    
    # Save the updated metadata back to the JSON file
    with open(json_file, 'w') as f:
        json.dump(metadata, f, indent=4)
