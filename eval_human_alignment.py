import json

def compare_relations(a, b, c, d, idx, type):
    # Returns 1 if relations are the same, otherwise returns 0
    # Check the relation between a and b
    relation_ab = (a > b) - (a < b)  # 1 means a > b, -1 means a < b, 0 means equal
    # Check the relation between c and d
    relation_cd = (c > d) - (c < d)

    # Exclude cases where values are equal
    if relation_ab == 0:
        return -1

    # Compare relations
    if relation_ab == relation_cd:
        return 1
    else:
        return 0

json_file = './data/local_metadata.json'
with open(json_file, 'r') as f:
    metadata = json.load(f)

LLM = 'gemma'
scale = 0.7
if 'global' in json_file:
    scale = 1.0  # exclude region judge
compare_groups = [(1,2), (1,3), (2,3), (4,3), (4,2), (4,1)]

# compute min and max
my_preservation = []
my_edit_quality = []

for entry in metadata:
    for i in range(1,5):
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
    for i in range(1,5):
        entry[f'edited_image{i}']['preservation_norm'] = (entry[f'edited_image{i}'][f'preservation'] - preservation_min) / (preservation_max-preservation_min)
        entry[f'edited_image{i}']['edit_quality_norm'] = (entry[f'edited_image{i}'][f'edit_quality'] - edit_quality_min) / (edit_quality_max-edit_quality_min)

with open(json_file, 'w') as f:
    json.dump(metadata, f, indent=4)

over_all = []

for compare_index1, compare_index2 in compare_groups:
    print(f"{compare_index1} vs {compare_index2}")
    cnt_mgie_ip2p = 0
    cnt_preservation = 0
    cnt_modify = 0

    pre_same_cnt = 0
    mod_same_cnt = 0
    all_same_cnt = 0

    all_cnt = 0
    for idx, entry in enumerate(metadata):
        all_cnt+=1
        idx = entry['index']

        semantic1 = (entry[f'edited_image{compare_index1}'][f'preservation'] - preservation_min) / (preservation_max-preservation_min) + (entry[f'edited_image{compare_index1}'][f'edit_quality'] - edit_quality_min) / (edit_quality_max-edit_quality_min)
        region1 = entry[f'edited_image{compare_index1}']['position'] + entry[f'edited_image{compare_index1}']['size']
        my_overall1 = scale * semantic1 + (1-scale) * region1

        semantic2 = (entry[f'edited_image{compare_index2}'][f'preservation'] - preservation_min) / (preservation_max-preservation_min) + (entry[f'edited_image{compare_index2}'][f'edit_quality'] - edit_quality_min) / (edit_quality_max-edit_quality_min)
        region2 = entry[f'edited_image{compare_index2}']['position'] + entry[f'edited_image{compare_index2}']['size']
        my_overall2 = scale * semantic2 + (1-scale) * region2
        all_quality = compare_relations(entry[f'edited_image{compare_index1}']['user_overall_quality'], entry[f'edited_image{compare_index2}']['user_overall_quality'], my_overall1, my_overall2, idx, 'overall')

        if all_quality == -1:
            all_same_cnt += 1
        else:
            cnt_mgie_ip2p += all_quality

    # print(f"same: {pre_same_cnt}, {mod_same_cnt}, {all_same_cnt}")
    # print(f'all_cnt: {all_cnt}')
    print(f"Overall: {round(cnt_mgie_ip2p / (all_cnt-all_same_cnt), 3)}")
    print(f'\n')
    over_all.append(round(cnt_mgie_ip2p / (all_cnt-all_same_cnt), 3))

print("all result:")
print(sum(over_all)/len(over_all))
