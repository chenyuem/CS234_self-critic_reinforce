from os import listdir

test_dir = '/datadrive/models/im2txt/data/mscoco/raw-data/test2014/'
filenames = listdir(test_dir)

fd = open('./test_images_list.txt', 'w')
for filename in filenames:
    fd.write(test_dir + filename + '\n')
fd.close()

