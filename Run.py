import os

# ResNet50 task
print(os.system("cd ResNet50"))
print(os.system("python main.py"))

# GraphSAGE task
print(os.system("cd .."))
print(os.system("cd GraphSAGE"))
print(os.system("python main.py"))

# # Transformer task
# print(os.system("cd .."))
# print(os.system("cd transformer-pytorch-master"))
# print(os.system("python train.py --problem wmt32k --output_dir ./output --data_dir ./wmt32k_data"))
