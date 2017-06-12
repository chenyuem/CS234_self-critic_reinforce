import json
import IPython

fd = open('new_test_images_list.txt', 'r')
captions = {}
while True:
    line = fd.readline()
    if line:
        key = line.split('/')[-1][:-1]
        captions[key] = []
    else:
        break
fd.close()

fd = open('data/dataset_coco.json', 'r')
data = json.load(fd)
fd.close()
for img in data['images']:
    if img['filename'] in captions:
        for sent in img['sentences']:
            captions[img['filename']].append(sent['raw'])

with open('new_test_gt_captions.txt', 'w') as outfile:
    json.dump(captions, outfile)

