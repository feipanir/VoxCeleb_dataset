import os

source_root = "./voxceleb2_proc_segments/"
label_files = ["label_part1.txt", "label_part2.txt", "label_part3.txt"]


with open(os.path.join(source_root, "merged_labels.txt"), "w") as outfile:
    for label_file in label_files:
        part_path = os.path.join(source_root, label_file)
        with open(part_path, "r") as infile:
            for line in infile:
                outfile.write(line)