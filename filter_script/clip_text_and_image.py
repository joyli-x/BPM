import json
import os

json_file = '/network_space/server128/shared/zhuoying/data/MyData/user_study/sample_test/metadata.json'
with open(json_file, 'r') as f:
    metadata = json.load(f)

clip_image_score = []
clip_text_score = []

for entry in metadata:
    for i in range(1,5):
        clip_image_score.append(entry[f'edited_image{i}']['clip_image'])
        clip_text_score.append(entry[f'edited_image{i}']['clip_text'])

# min and max
clip_image_min = min(clip_image_score)
clip_image_max = max(clip_image_score)
clip_text_min = min(clip_text_score)
clip_text_max = max(clip_text_score)

# normalize
for entry in metadata:
    for i in range(1,5):
        clip_i_norm = (entry[f'edited_image{i}']['clip_image'] - clip_image_min) / (clip_image_max-clip_image_min)
        clip_t_norm = (entry[f'edited_image{i}']['clip_text'] - clip_text_min) / (clip_text_max-clip_text_min)
        entry[f'edited_image{i}']['clip_image_text'] = clip_i_norm + clip_t_norm

# Save the updated metadata
with open(json_file, 'w') as f:
    json.dump(metadata, f, indent=4)


