import os

files = os.listdir("Flowers/images")
files = sorted(files)
files = files[:-1]
print(files)
print(len(files))

os.system("mkdir Flowers_converted")
for i in range(17):
    os.system(f"mkdir Flowers_converted/{i}")

for i, elem in enumerate(files):
    path = "Flowers/images/" + elem
    cl = i // 80
    os.system(f"cp {path} Flowers_converted/{cl}")

with open("Flowers_converted/classes.txt", "w+") as f:
    for i in range(17):
        print(f"{i} {i}", file=f)

with open("Flowers_converted/image_class_labels.txt", "w+") as f:
    for i in range(1360):
        cl = i//80
        print(f"{i} {cl}", file=f)

with open("Flowers_converted/images.txt", "w+") as f:
    for i in range(1360):
        cl = i//80
        path = f"Flowers_converted/{cl}/{files[i]}"
        print(f"{i} {path}", file=f)

with open("Flowers_converted/train_test_split.txt", "w+") as f:
    for i in range(1360):
        pos = i % 80
        if pos > 70:
            flag = 0
        else:
            flag = 1
        print(f"{i} {flag}", file=f)

