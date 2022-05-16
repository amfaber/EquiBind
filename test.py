import os
path = "data/results/"
name = "4aa5"
dirs = ["unit_test", "unit_test_base"]
data = []
for dir in dirs:
    full_path = os.path.join(path, dir, name, "lig_equibind_corrected.sdf")
    with open(full_path) as file:
        data.append(file.read())
print(data[0] == data[1])