
import torch
import networkx as nx


from pcfgan import PCFGAN
from pcfgan.PCFGAN import gen_data_nonlinear
from decaf.data import DataModule
import pytorch_lightning as pl

# 使用示例
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
latent_dim = 100
num_features = 10

# 因果图的父节点关系定义，例如：{0: [], 1: [0], 2: [0, 1], ...}
parents_dict = {
    0: [],
    1: [0],
    2: [0, 1],
    3: [2],
    4: [1],
    5: [3, 4],
    6: [2, 5],
    7: [6],
    8: [4],
    9: [7, 8]
}

# 创建一个空列表来存储结果
result_list = []
# 遍历字典中的每个键值对
for key, parents in parents_dict.items():
    # 遍历每个父节点
    for parent in parents:
        # 将父节点和子节点的组合添加到结果列表中
        result_list.append([parent, key])

# 输出结果
print(result_list)


# edge removal dictionary
bias_dict = {6: [2]}  # This removes the edge into 6 from 3.
# 使用next和iter获取第一个键
first_key = next(iter(bias_dict))

print(bias_dict[first_key][0])

cfgan = PCFGAN(latent_dim, num_features, result_list,bias_dict)

# DATA SETUP according to dag_seed
G = nx.DiGraph(result_list)
data = gen_data_nonlinear(G, SIZE=2000)
dm = DataModule(data.values)
dataset = dm.dataset.x

trainer = pl.Trainer(
    gpus=0,
    max_epochs=50,
    progress_bar_refresh_rate=1,
    profiler=False,
    callbacks=[],
)
trainer.fit(cfgan, dm)
synth_data = (
    cfgan.gen_synthetic(dataset, biased_edges=bias_dict).detach().cpu().numpy()
)
print("Data generated successfully!")


# 假设 'dataset' 是一个包含真实数据的 DataLoader 对象
# for epoch in range(50):
#     for real_data in dataset:
#         rl=torch.FloatTensor(real_data).unsqueeze(0)
#         losses = cfgan.train_step(rl)
#         print(losses)

