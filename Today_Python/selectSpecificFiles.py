import os
dirs_list = ['dir_1', 'dir_2']

for root, dirs, files in os.walk('labels'):
    dirs[:] = [d for d in dirs if d in dirs_list]
    for file in files:
        if file.endswith(".txt"):
             print(os.path.join(root, file))