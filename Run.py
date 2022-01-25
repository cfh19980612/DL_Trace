import os
import wandb

wandb.init(project="Switching", entity="fahao", name="ResNet-Transformer")

# Task 1
for i in range (100):
print(os.system("cd GraphSAGE; python main.py --num-epochs=1"))
print(os.system("cd ResNet50; python main.py"))

# Task 2
# print(os.system("cd ResNet50; python main.py"))
# print(os.system("cd GraphSAGE; python main.py --num-epochs=20"))
 
# Task 3 
# print(os.system("cd GraphSAGE; python main.py --num-epochs=1")) 
# print(os.system("cd transformer-pytorch-master; python train.py --problem wmt32k --output_dir ./output --data_dir ./wmt32k_data")) 
 
# Task 4 
# print(os.system("cd transformer-pytorch-master; python train.py --problem wmt32k --output_dir ./output --data_dir ./wmt32k_data")) 
# print(os.system("cd GraphSAGE; python main.py --num-epochs=20")) 
 
# Task 5 
# print(os.system("cd ResNet50; python main.py")) 
# print(os.system("cd transformer-pytorch-master; python train.py --problem wmt32k --output_dir ./output --data_dir ./wmt32k_data")) 
 
# Task 6 
# print(os.system("cd transformer-pytorch-master; python train.py --problem wmt32k --output_dir ./output --data_dir ./wmt32k_data")) 
# print(os.system("cd ResNet50; python main.py")) 
