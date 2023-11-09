import os

def print_folder_structure(startpath, indent=0):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}+ {}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        #for f in files:
        #    print('{}{}'.format(subindent, f))

# Replace 'your_directory' with the path to the directory you want to print
your_directory = '../MLBenchmarks'
print_folder_structure(your_directory)