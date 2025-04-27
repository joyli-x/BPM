import json

json_file = '/network_space/server128/shared/zhuoying/data/MyData/user_study/sample_test/metadata.json'
with open(json_file, 'r') as f:
    data = json.load(f)

correct = [0, 0, 0, 0, 0]
size_is_one = [0, 0, 0, 0, 0]

# size
for i in range(100):
    for j in range(1, 5):
        if data[i][f"edited_image{j}"]['user_size'] == data[i][f"edited_image{j}"]['my_size']:
            correct[j] += 1

print(f"size: ")
print(f"mgie: {correct[1]}, ft_ip2p: {correct[2]}, ip2p: {correct[3]}, dalle2: {correct[4]}")
# print(size_is_one)

# positioin
correct = [0, 0, 0, 0, 0]
for i in range(100):
    for j in range(1, 5):
        if data[i][f"edited_image{j}"]['user_position'] == data[i][f"edited_image{j}"]['my_position']:
            correct[j] += 1
print(f"position: ")
print(f"mgie: {correct[1]}, ft_ip2p: {correct[2]}, ip2p: {correct[3]}, dalle2: {correct[4]}")
