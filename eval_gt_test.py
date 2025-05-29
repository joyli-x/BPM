import json

def compare_score(score1, score2, score3):
    # 对于三个分数，谁最高就返回哪个model的value
    idx2model = {'1': 'sd_xl', '2': 'noise_jitter', '3': 'dalle2'}
    if score1 > score2 and score1 > score3:
        return idx2model['1']
    elif score2 > score1 and score2 > score3:
        return idx2model['2']
    elif score3 > score1 and score3 > score2:
        return idx2model['3']
    else:
        raise ValueError("All scores are equal or invalid scores provided.")

json_file = './data/gt_test_metadata_new.json'
with open(json_file, 'r') as f:
    metadata = json.load(f)

LLM = 'gemma'
scale = 0.7

# compute min and max
my_preservation = []
my_edit_quality = []

for entry in metadata:
    for i in range(1,4):
        my_preservation.append(entry[f'edited_image{i}'][f'preservation'])
        # if entry[f'edited_image{i}']['my_edit_quality3'] > 0:
        my_edit_quality.append(entry[f'edited_image{i}'][f'edit_quality'])

my_preservation_new = [x for x in my_preservation if (x!=0 and x!=-1)]
preservation_min = min(my_preservation_new)
preservation_max = max(my_preservation_new)
edit_quality_min = min(my_edit_quality)
edit_quality_max = max(my_edit_quality)
print(f"preservation_min: {preservation_min}, preservation_max: {preservation_max}")
print(f"edit_quality_min: {edit_quality_min}, edit_quality_max: {edit_quality_max}")

for entry in metadata:
    for i in range(1,4):
        entry[f'edited_image{i}']['preservation_norm'] = (entry[f'edited_image{i}'][f'preservation'] - preservation_min) / (preservation_max-preservation_min)
        entry[f'edited_image{i}']['edit_quality_norm'] = (entry[f'edited_image{i}'][f'edit_quality'] - edit_quality_min) / (edit_quality_max-edit_quality_min)

with open(json_file, 'w') as f:
    json.dump(metadata, f, indent=4)

score = {'sd_xl': 0, 'noise_jitter': 0, 'dalle2': 0}

for entry in metadata:
    
    semantic1 = (entry[f'edited_image1'][f'preservation'] - preservation_min) / (preservation_max-preservation_min) + (entry[f'edited_image1'][f'edit_quality'] - edit_quality_min) / (edit_quality_max-edit_quality_min)
    region1 = entry[f'edited_image1']['position'] + entry[f'edited_image1']['size']
    my_overall1 = scale * semantic1 + (1-scale) * region1

    semantic2 = (entry[f'edited_image2'][f'preservation'] - preservation_min) / (preservation_max-preservation_min) + (entry[f'edited_image2'][f'edit_quality'] - edit_quality_min) / (edit_quality_max-edit_quality_min)
    region2 = entry[f'edited_image2']['position'] + entry[f'edited_image2']['size']
    my_overall2 = scale * semantic2 + (1-scale) * region2
    
    semantic3 = (entry[f'edited_image3'][f'preservation'] - preservation_min) / (preservation_max-preservation_min) + (entry[f'edited_image3'][f'edit_quality'] - edit_quality_min) / (edit_quality_max-edit_quality_min)
    region3 = entry[f'edited_image3']['position'] + entry[f'edited_image3']['size']
    my_overall3 = scale * semantic3 + (1-scale) * region3

    # Compare the overall scores
    best_method = compare_score(my_overall1, my_overall2, my_overall3)
    score[best_method] += 1

print(f"{score}")
