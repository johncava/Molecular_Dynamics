import glob

dir_ = "./outputs_backbone/"

files = glob.glob(dir_ + '*.pdb')

files = [(int(f.split('rand-orient-')[-1].split('.pdb')[0]), f) for f in files]

files.sort(key=lambda tup: tup[0])

files = files[::10]

files = files[5000:10001]
print(files)

print(len(files))

