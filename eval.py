# 检测user study和我们的metric的一致性
import json
import argparse

def compare_relations(a, b, c, d, idx, type):
    # 关系相同返回1，否则返回0
    # 检查 a 和 b 的关系
    relation_ab = (a > b) - (a < b)  # 1 表示 a > b, -1 表示 a < b, 0 表示相等
    # 检查 c 和 d 的关系
    relation_cd = (c > d) - (c < d)  # 同样的计算方式

    # 排除相等情况
    if relation_ab == 0:
        # print(type)
        # print(idx)
        # print("a == b")
        return -1

    # 比较关系
    if relation_ab == relation_cd:
        return 1
    else:
        if type == 'preservation':
            print(f"{type}, {idx}, 关系不等")
        return 0

parser = argparse.ArgumentParser()
parser.add_argument('--metrics', type=str, default='my')
args = parser.parse_args()
# clip_image clip_text clip_image_text lpips dino mse gpt my
metrics = args.metrics

json_file = '/network_space/server128/shared/zhuoying/data/MyData/user_study/sample_test/metadata.json'
with open(json_file, 'r') as f:
    metadata = json.load(f)

LLM = 'gemma'
scale = 0.5
compare_groups = [(1,2), (1,3), (2,3), (4,3), (4,2), (4,1)]

if metrics == 'my':
    # compute min and max
    my_preservation = []
    my_edit_quality = []

    for entry in metadata:
        for i in range(1,5):
            my_preservation.append(entry[f'edited_image{i}'][f'my_preservation_{LLM}'])
            # if entry[f'edited_image{i}']['my_edit_quality3'] > 0:
            my_edit_quality.append(entry[f'edited_image{i}'][f'my_edit_quality_{LLM}'])

    my_preservation_new = [x for x in my_preservation if x!=0]
    preservation_min = min(my_preservation)
    preservation_max = max(my_preservation_new)
    edit_quality_min = min(my_edit_quality)
    edit_quality_max = max(my_edit_quality)
    print(f"preservation_min: {preservation_min}, preservation_max: {preservation_max}")
    print(f"edit_quality_min: {edit_quality_min}, edit_quality_max: {edit_quality_max}")

    for entry in metadata:
        for i in range(1,5):
            entry[f'edited_image{i}']['my_preservation_norm'] = (entry[f'edited_image{i}'][f'my_preservation_{LLM}'] - preservation_min) / (preservation_max-preservation_min)
            entry[f'edited_image{i}']['my_edit_quality_norm'] = (entry[f'edited_image{i}'][f'my_edit_quality_{LLM}'] - edit_quality_min) / (edit_quality_max-edit_quality_min)

    # with open(json_file, 'w') as f:
    #     json.dump(metadata, f, indent=4)


idx2model = {'1': 'mgie', '2': 'ft_ip2p', '3': 'ip2p', '4': 'dalle2'}
# ignore_index = [1,2,3,7,8,9,11,12,13,17,19,20,21,27,28]
# idx2model = {'1': 'masactrl', '2': 'infedit'}
# 12 13 23 43 42 41
# cnt_mgie_ip2p = 0
# cnt_ft_ip2p_ip2p = 0
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
        # if entry['idx'] in ignore_index:
        #     continue
        all_cnt+=1
        # idx = entry['idx']
        idx = entry['index']
        # print(f"idx: {entry['idx']}")
        if metrics=='my': 
            # preservation = compare_relations(entry[f'edited_image{compare_index1}']['user_preservation'], entry[f'edited_image{compare_index2}']['user_preservation'], entry[f'edited_image{compare_index1}'][f'my_preservation_{LLM}'], entry[f'edited_image{compare_index2}'][f'my_preservation_{LLM}'], idx, 'preservation')
            # # preservation = compare_relations(entry[f'edited_image{compare_index1}']['user_preservation'], entry[f'edited_image{compare_index2}']['user_preservation'], entry[f'edited_image{compare_index1}']['mse'], entry[f'edited_image{compare_index2}']['mse'], idx, 'preservation')
            # if preservation == -1:
            #     pre_same_cnt += 1
            # else:
            #     cnt_preservation += preservation
            
            # modify = compare_relations(entry[f'edited_image{compare_index1}']['user_edit_quality'], entry[f'edited_image{compare_index2}']['user_edit_quality'], entry[f'edited_image{compare_index1}'][f'my_edit_quality_{LLM}'], entry[f'edited_image{compare_index2}'][f'my_edit_quality_{LLM}'], idx, 'modify')
            # # modify = compare_relations(entry[f'edited_image{compare_index1}']['user_edit_quality'], entry[f'edited_image{compare_index2}']['user_edit_quality'], entry[f'edited_image{compare_index1}']['clip_text'], entry[f'edited_image{compare_index2}']['clip_text'], idx, 'modify')
            # if modify == -1:
            #     mod_same_cnt += 1
            # else:
            #     cnt_modify += modify
            
            # normalize
            # my_overall1 = scale * (entry[f'edited_image{compare_index1}'][f'my_preservation_{LLM}'] - preservation_min) / (preservation_max-preservation_min)
            my_overall1 = scale * (entry[f'edited_image{compare_index1}'][f'my_preservation_{LLM}'] - preservation_min) / (preservation_max-preservation_min) + (1-scale)*(entry[f'edited_image{compare_index1}'][f'my_edit_quality_{LLM}'] - edit_quality_min) / (edit_quality_max-edit_quality_min)
            my_overall2 = scale * (entry[f'edited_image{compare_index2}'][f'my_preservation_{LLM}'] - preservation_min) / (preservation_max-preservation_min) + (1-scale)*(entry[f'edited_image{compare_index2}'][f'my_edit_quality_{LLM}'] - edit_quality_min) / (edit_quality_max-edit_quality_min)
            # my_overall2 = scale * (entry[f'edited_image{compare_index2}'][f'my_preservation_{LLM}'] - preservation_min) / (preservation_max-preservation_min)
            # my_overall1 = entry[f'edited_image{compare_index1}']['mse'] / 100 + entry[f'edited_image{compare_index1}']['mse']
            # my_overall2 = entry[f'edited_image{compare_index2}']['mse'] / 100 + entry[f'edited_image{compare_index2}']['mse']
            all_quality = compare_relations(entry[f'edited_image{compare_index1}']['user_overall_quality'], entry[f'edited_image{compare_index2}']['user_overall_quality'], my_overall1, my_overall2, idx, 'overall')
            # if all_quality == 1:
            #     if "all" not in entry['LLM']['gemma_origin']:
            #         print(f"idx: {idx}, {compare_index1}, {compare_index2}, {entry[f'edited_image{compare_index1}']['user_overall_quality']}, {entry[f'edited_image{compare_index2}']['user_overall_quality']}, {my_overall1}, {my_overall2}")
            if all_quality == -1:
                all_same_cnt += 1
            else:
                cnt_mgie_ip2p += all_quality
        elif metrics=='gpt':
            score1 = entry[f'edited_image{compare_index1}']['alignment_score'] + entry[f'edited_image{compare_index1}']['coherence_score']
            score2 = entry[f'edited_image{compare_index2}']['alignment_score'] + entry[f'edited_image{compare_index2}']['coherence_score']
            all_quality = all_quality = compare_relations(entry[f'edited_image{compare_index1}']['user_overall_quality'], entry[f'edited_image{compare_index2}']['user_overall_quality'], score1, score2, idx, 'overall')
            if all_quality == -1:
                all_same_cnt += 1
            else:
                cnt_mgie_ip2p += all_quality
        else:
            all_quality = all_quality = compare_relations(entry[f'edited_image{compare_index1}']['user_overall_quality'], entry[f'edited_image{compare_index2}']['user_overall_quality'], entry[f'edited_image{compare_index1}'][metrics], entry[f'edited_image{compare_index2}'][metrics], idx, 'overall')
            if all_quality == -1:
                all_same_cnt += 1
            else:
                cnt_mgie_ip2p += all_quality

    print(f"same: {pre_same_cnt}, {mod_same_cnt}, {all_same_cnt}")
    print(f'all_cnt: {all_cnt}')
    # print(f"Preservation: {round(cnt_preservation / (100-pre_same_cnt), 3)}")
    # print(f"Modify: {round(cnt_modify / (100-mod_same_cnt), 3)}")
    print(f"Overall: {round(cnt_mgie_ip2p / (all_cnt-all_same_cnt), 3)}")
    print(f'\n')
    over_all.append(round(cnt_mgie_ip2p / (all_cnt-all_same_cnt), 3))

print("all result:")
print(sum(over_all)/len(over_all))
