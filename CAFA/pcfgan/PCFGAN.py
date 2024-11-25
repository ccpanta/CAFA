import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
from typing import Any, List, Optional, Union
import pytorch_lightning as pl
import numpy as np
import pandas as pd

from pcfgan.data import DataModule
# 定义生成器
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_nonlin(name: str) -> nn.Module:
    if name == "none":
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    elif name == "relu":
        return nn.ReLU()
    elif name == "leaky_relu":
        return nn.LeakyReLU()
    elif name == "selu":
        return nn.SELU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "softmax":
        return nn.Softmax(dim=-1)
    else:
        raise ValueError(f"Unknown nonlinearity {name}")

class Generator_causal(nn.Module):
    def __init__(
        self,
        z_dim: int,
        x_dim: int,
        h_dim: int,
        f_scale: float = 0.1,
        dag_seed: list = [],
        nonlin_out: Optional[List] = None,
    ) -> None:
        super().__init__()

        if nonlin_out is not None:
            out_dim = 0
            for act, length in nonlin_out:
                out_dim += length
            if out_dim != x_dim:
                raise RuntimeError("Invalid nonlin_out")

        self.x_dim = x_dim
        self.nonlin_out = nonlin_out

        def block(in_feat: int, out_feat: int, normalize: bool = False) -> list:
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.ReLU(inplace=True))
            return layers

        self.shared = nn.Sequential(*block(h_dim, h_dim), *block(h_dim, h_dim)).to(
            DEVICE
        )

        if len(dag_seed) > 0:
            M_init = torch.rand(x_dim, x_dim) * 0.0
            M_init[torch.eye(x_dim, dtype=bool)] = 0
            M_init = torch.rand(x_dim, x_dim) * 0.0
            for pair in dag_seed:
                M_init[pair[0], pair[1]] = 1

            M_init = M_init.to(DEVICE)
            self.M = torch.nn.parameter.Parameter(M_init, requires_grad=False).to(
                DEVICE
            )
        else:
            M_init = torch.rand(x_dim, x_dim) * 0.2
            M_init[torch.eye(x_dim, dtype=bool)] = 0
            M_init = M_init.to(DEVICE)
            self.M = torch.nn.parameter.Parameter(M_init).to(DEVICE)

        self.fc_i = nn.ModuleList(
            [nn.Linear(x_dim + 1, h_dim) for i in range(self.x_dim)]
        )
        self.fc_f = nn.ModuleList([nn.Linear(h_dim, 1) for i in range(self.x_dim)])

        for layer in self.shared.parameters():
            if type(layer) == nn.Linear:
                torch.nn.init.xavier_normal_(layer.weight)
                layer.weight.data *= f_scale

        for i, layer in enumerate(self.fc_i):
            torch.nn.init.xavier_normal_(layer.weight)
            layer.weight.data *= f_scale
            layer.weight.data[:, i] = 1e-16

        for i, layer in enumerate(self.fc_f):
            torch.nn.init.xavier_normal_(layer.weight)
            layer.weight.data *= f_scale

    def sequential(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        gen_order: Union[list, dict, None] = None,
        biased_edges: dict = {},
    ) -> torch.Tensor:
        out = x.clone().detach()

        if gen_order is None:
            gen_order = list(range(self.x_dim))

        for i in gen_order:
            x_masked = out.clone() * self.M[:, i]
            x_masked[:, i] = 0.0
            if i in biased_edges:
                for j in biased_edges[i]:
                    x_j = x_masked[:, j]
                    perm = torch.randperm(len(x_j))
                    x_masked[:, j] = x_j[perm]
            out_i = self.fc_i[i](torch.cat([x_masked, z[:, i].unsqueeze(1)], axis=1))
            out_i = nn.ReLU()(out_i)
            out_i = self.shared(out_i)
            out_i = self.fc_f[i](out_i).squeeze()
            out[:, i] = out_i

        if self.nonlin_out is not None:
            split = 0
            for act_name, step in self.nonlin_out:
                activation = get_nonlin(act_name)
                out[..., split : split + step] = activation(
                    out[..., split : split + step]
                )

                split += step

            if split != out.shape[-1]:
                raise ValueError("Invalid activations")

        return out
# 定义生成器
class Generator(nn.Module):
    def __init__(self, latent_dim, num_features, parents_dict):
        super(Generator, self).__init__()
        self.num_features = num_features
        self.parents_dict = {str(k): v for k, v in parents_dict.items()}  # 确保字典键为字符串类型

        self.input_layers = nn.ModuleDict({
            str(i): nn.Linear(latent_dim, 128) for i in range(num_features)
        })
        self.hidden_layers = nn.ModuleDict({
            str(i): nn.Sequential(
                nn.ReLU(True),
                nn.Linear(128 + len(self.parents_dict[str(i)]), 256),
                nn.ReLU(True),
                nn.Linear(256, 1),
                nn.Sigmoid()
            ) for i in range(num_features)
        })

    def forward(self, z):
        outputs = {}
        for i in range(self.num_features):
            input_layer = self.input_layers[str(i)]
            hidden_layer = self.hidden_layers[str(i)]

            parent_outputs = torch.cat([outputs[str(p)] for p in self.parents_dict[str(i)]], dim=1) if \
            self.parents_dict[str(i)] else torch.zeros(z.size(0), 0).to(z.device)
            combined_input = torch.cat([input_layer(z), parent_outputs], dim=1)
            outputs[str(i)] = hidden_layer(combined_input)


        return torch.cat([outputs[str(i)] for i in range(self.num_features)], dim=1)

# 定义判别器
class Discriminator(nn.Module):
    def __init__(self, num_features):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# 定义分类器
class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# 定义CFGAN类
class PCFGAN(pl.LightningModule):
    def __init__(self, latent_dim, num_features, parents_dict, biased_dict, lr=0.0002):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_features = num_features
        self.lr=lr
        self.z_dim=num_features
        # 构建生成器G1
        # self.generator_g1 = Generator(latent_dim, num_features, parents_dict).to(device)

        self.biased_edges = biased_dict
        self.generator_g1 = Generator_causal(z_dim=self.num_features,x_dim=self.num_features,h_dim=latent_dim,dag_seed=parents_dict).to(DEVICE)


        # 构建判别器D1和D2
        self.discriminator_d1 = Discriminator(num_features).to(DEVICE)
        self.discriminator_d2 = Discriminator(num_features-1).to(DEVICE)
        self.discriminator_d3 = Discriminator(1).to(DEVICE)

        # 构建分类器
        self.classifier = Classifier(self.num_features-2, 1)

        # 损失函数
        self.loss_fn = nn.BCELoss()

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.generator_g1(x, z)

    def get_W(self) -> torch.Tensor:
        return self.generator_g1.M

    def get_dag(self) -> np.ndarray:
        return np.round(self.get_W().detach().cpu().numpy(), 3)

    def get_gen_order(self) -> list:
        dense_dag = np.array(self.get_dag())
        dense_dag[dense_dag > 0.5] = 1
        dense_dag[dense_dag <= 0.5] = 0
        G = nx.from_numpy_matrix(dense_dag, create_using=nx.DiGraph)
        gen_order = list(nx.algorithms.dag.topological_sort(G))
        return gen_order

    def sample_z(self, n: int) -> torch.Tensor:
        return torch.randn(n, self.z_dim, device=DEVICE)

    def training_step(self, real_data: torch.Tensor,batch_idx: int,optimizer_idx: int):
        batch_size = real_data.size(0)
        real_labels = torch.ones(batch_size, 1).to(DEVICE)
        fake_labels = torch.zeros(batch_size, 1).to(DEVICE)
        z = torch.randn(batch_size, self.latent_dim).to(DEVICE)


        # 生成假数据
        generated_data_g1 = self.generator_g1.sequential(real_data,z,self.get_gen_order())
        generated_data_g2 = self.generator_g1.sequential(real_data,z,self.get_gen_order(), biased_edges=self.biased_edges)
        # generated_data_g2 = self.generator_g2(z)

        # 训练判别器D1
        if optimizer_idx == 0:

            real_preds_d1 = self.discriminator_d1(real_data)
            fake_preds_d1 = self.discriminator_d1(generated_data_g1.detach())
            d1_loss_real = self.loss_fn(real_preds_d1, real_labels)
            d1_loss_fake = self.loss_fn(fake_preds_d1, fake_labels)
            d1_loss = d1_loss_real + d1_loss_fake
            return d1_loss


        # 训练判别器D2
        if optimizer_idx == 1:
            # self.d2_optimizer.zero_grad()
            # 构建列索引张量，排除要删除的列
            num_cols = self.num_features
            first_key = next(iter(self.biased_edges))
            indices = [i for i in range(num_cols) if i != self.biased_edges[first_key][0]]
            indices_tensor= torch.tensor(indices)
            rl_data=real_data[:,indices_tensor]
            rl_data_s=real_data[:,self.biased_edges[first_key][0]].unsqueeze(1)
            fk_data=generated_data_g2.detach()[:,indices_tensor]
            fk_data_s=generated_data_g2.detach()[:,self.biased_edges[first_key][0]].unsqueeze(1)

            real_preds_d2 = self.discriminator_d2(rl_data)
            fake_preds_d2 = self.discriminator_d2(fk_data)
            d2_loss_real = self.loss_fn(real_preds_d2, rl_data_s)
            d2_loss_fake = self.loss_fn(fake_preds_d2, fk_data_s)
            d2_loss = d2_loss_real + d2_loss_fake
            return d2_loss

        # 训练判别器D2
        if optimizer_idx == 2:
            # self.d2_optimizer.zero_grad()
            # 构建列索引张量，排除要删除的列
            num_cols = self.num_features
            first_key = next(iter(self.biased_edges))
            indices = [i for i in range(num_cols) if i != self.biased_edges[first_key][0] & i !=num_cols-1]
            indices_tensor= torch.tensor(indices)

            rl_data_x=real_data[:,indices_tensor]
            rl_data_y=real_data[:,num_cols-1]
            rl_data_s=real_data[:,self.biased_edges[first_key][0]].unsqueeze(1)

            fk_data_x=generated_data_g2.detach()[:,indices_tensor]
            fk_data_y=generated_data_g2.detach()[:,num_cols-1]
            fk_data_s=generated_data_g2.detach()[:,self.biased_edges[first_key][0]].unsqueeze(1)

            y_pred = self.classifier(rl_data_x)
            real_preds_d3 = self.discriminator_d3(y_pred)
            fake_preds_d3 = self.discriminator_d3(self.classifier(fk_data_x))
            d3_loss_real = self.loss_fn(real_preds_d3, rl_data_s)
            d3_loss_fake = self.loss_fn(fake_preds_d3, fk_data_s)
            d3_loss = d3_loss_real + d3_loss_fake
            return d3_loss


        # 训练生成器G3
        if optimizer_idx == 3:
            # self.g1_optimizer.zero_grad()
            fake_preds_g1 = self.discriminator_d1(generated_data_g1)
            g1_loss = self.loss_fn(fake_preds_g1, real_labels)
            return g1_loss
            # g1_loss.backward()
            # self.g1_optimizer.step()
        #训练分类器
        if optimizer_idx == 4:
            num_cols = self.num_features
            first_key = next(iter(self.biased_edges))
            indices = [i for i in range(num_cols) if i != self.biased_edges[first_key][0] & i !=num_cols-1]
            indices_tensor= torch.tensor(indices)

            rl_data_x=real_data[:,indices_tensor]
            rl_data_y=real_data[:,num_cols-1].unsqueeze(1)

            rl_pre_y = self.classifier(rl_data_x)
            c_loss = self.loss_fn(rl_pre_y, rl_data_y)
            return c_loss

        # 训练生成器G2
        # if optimizer_idx == 3:
        #     # self.g2_optimizer.zero_grad()
        #     fake_preds_g2 = self.discriminator_d2(generated_data_g2)
        #     g2_loss = self.loss_fn(fake_preds_g2, real_labels)
        #     return g2_loss
            # g2_loss.backward()
            # self.g2_optimizer.step()
      # return {"d1_loss": d1_loss.item(), "d2_loss": d2_loss.item(), "g1_loss": g1_loss.item(),"g2_loss": g2_loss.item()}

    def gen_synthetic(self, x: torch.Tensor, biased_edges: dict = {}) -> torch.Tensor:
        self.generator_g1 = self.generator_g1.to(DEVICE)
        x = x.to(DEVICE)
        gen_order = self.get_gen_order()
        return self.generator_g1.sequential(
            x,
            self.sample_z(x.shape[0]).type_as(x),
            gen_order=gen_order,
            biased_edges=biased_edges,
        )


    def configure_optimizers(self) -> tuple:
        lr = self.lr


        opt_g1 = torch.optim.AdamW(
            self.generator_g1.parameters(),
            lr=lr,
        )

        # opt_g2 = torch.optim.AdamW(
        #     self.generator_g2.parameters(),
        #     lr=lr,
        # )

        opt_d1 = torch.optim.AdamW(
            self.discriminator_d1.parameters(),
            lr=lr,
        )
        opt_d2 = torch.optim.AdamW(
            self.discriminator_d1.parameters(),
            lr=lr,
        )

        opt_d3 = torch.optim.AdamW(
            self.discriminator_d3.parameters(),
            lr=lr,
        )
        opt_c = torch.optim.AdamW(
            self.classifier.parameters(),
            lr=lr,
        )
        return [opt_d1, opt_d2,opt_d3, opt_g1,opt_c], []
# It will apply a perturbation at each node provided in perturb.
def gen_data_nonlinear(
    G: Any,
    base_mean: float = 0,
    base_var: float = 0.3,
    mean: float = 0,
    var: float = 1,
    SIZE: int = 10000,
    err_type: str = "normal",
    perturb: list = [],
    sigmoid: bool = True,
    expon: float = 1.1,
) -> pd.DataFrame:
    list_edges = G.edges()
    list_vertex = G.nodes()

    order = []
    for ts in nx.algorithms.dag.topological_sort(G):
        order.append(ts)

    g = []
    for v in list_vertex:
        if v in perturb:
            g.append(np.random.normal(mean, var, SIZE))
            print("perturbing ", v, "with mean var = ", mean, var)
        else:
            if err_type == "gumbel":
                g.append(np.random.gumbel(base_mean, base_var, SIZE))
            else:
                g.append(np.random.normal(base_mean, base_var, SIZE))

    for o in order:
        for edge in list_edges:
            if o == edge[1]:  # if there is an edge into this node
                if sigmoid:
                    g[edge[1]] += 1 / 1 + np.exp(-g[edge[0]])
                else:
                    g[edge[1]] += g[edge[0]] ** 2
    g = np.swapaxes(g, 0, 1)

    return pd.DataFrame(g, columns=list(map(str, list_vertex)))

