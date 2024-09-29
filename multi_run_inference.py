import os

# 分两次跑,太多并行cpu吃不消
cuda = [2,3] #[5,6,7]
exp_ids = list(range(0, 24)) # (12,24)
# 注意卡数要能被24整除 （2，3，4，6）
# python main_inference.py --evaluate /home/huguohong/raf-db-model_best.pth --data_json {} --gpu {}

cmd = ""

for i, cuda_id in enumerate(cuda):
    split_num = len(exp_ids) // len(cuda)
    for exp_id in exp_ids[i*split_num:(i+1)*split_num]:
        cmd += f"python main_inference.py --evaluate /home/huguohong/raf-db-model_best.pth --data_json faceCaption5M_refine_216k_split/{exp_id}.json --gpu {cuda_id} && "
    cmd = cmd[:-4] + " & "

print(cmd)
os.system(cmd)
print("done")