import json

json_file = '/network_space/server128/shared/zhuoying/data/MyData/gt_test/gt_test_metadata.json'
with open(json_file, 'r') as f:
    data = json.load(f)

preservation_min = -0.0410771773847966
preservation_max = -0.003097
edit_quality_min = 0.0
edit_quality_max = 0.3245394825935364

metric_list = ['clip_text', 'clip_image', 'lpips', 'mse', 'dino', 'gpt', 'clip_image_text', 'my']
# metric_list = ['clip_score']

for metric in metric_list:
    score = {'noise_jitter': 0, 'sd_xl': 0, 'dalle2': 0}
    for entry in data:
        if metric == 'my':
            metric_scores = {}
            for edit_method in entry['my_preservation_score'].keys():
                # normalization
                metric_scores[edit_method] = (entry['my_preservation_score'][edit_method] - preservation_min) / (preservation_max - preservation_min) + (entry['my_modification_score'][edit_method] - edit_quality_min) / (edit_quality_max - edit_quality_min)
            # if metric_scores['sd_xl'] > metric_scores['dalle2']:
            #     print(f'entry: {entry['idx']}, sd_xl > dalle2')
        else:
            metric_scores = entry[metric] # 这是一个字典，key是edit_method，value是score
        # 将分数最高的在score里面+1
        best_method = max(metric_scores, key=metric_scores.get)
        score[best_method] += 1
            

    print(f"metric: {metric}, score: {score}")