import sys
import os

def extract_cater_list(root):
    with open(root + "trainlist_multiview.txt", "w") as f:
        for folder in os.listdir(root):
            if folder.endswith('.txt'):
                continue
            dir = root + folder + "/"
            f.write(dir + " 181\n")

            #dir = root + folder + "/images_view1/"
            #f.write(dir + " 181\n")
            #dir = root + folder + "/images_view2/"
            #f.write(dir + " 181\n")
            #dir = root + folder + "/images_view3/"
            #f.write(dir + " 181\n")
    f.close()


if __name__ == "__main__":
    directory = "/proj/vondrick/datasets/CATER/GREATER_multi_fbb/train/"
    extract_cater_list(directory)