import json, IPython

fd = open('./new_test_images_list.txt', 'w')
test_dir = '/datadrive/models/im2txt/data/mscoco/raw-data/'

counter = 0
data = json.load(open('data/cocotalk.json', 'r'))
for img in data['images']:
    if img['split'] == 'test':
        counter += 1
        fd.write(test_dir + img['file_path'] + '\n')
        if counter == 100:
            break
fd.close()
