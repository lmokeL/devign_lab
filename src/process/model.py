from http.client import UnimplementedFileMode
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import GatedGraphConv

torch.manual_seed(2020)


def get_conv_mp_out_size(in_size, last_layer, mps):
    size = in_size

    for mp in mps:
        size = round((size - mp["kernel_size"]) / mp["stride"] + 1)

    size = size + 1 if size % 2 != 0 else size

    return int(size * last_layer["out_channels"])


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)


import torch
import torch.nn as nn
import torch.nn.functional as F


class ReadoutInvariantPool(nn.Module):
    def __init__(self, h_shape1, x_shape1, max_nodes):
        super(ReadoutInvariantPool, self).__init__()
        self.max_nodes = max_nodes

        # 节点处理器
        self.node_processor = nn.Sequential(
            nn.Linear(h_shape1 + x_shape1, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # 图分类器
        self.graph_classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, h, x):
        # 批处理大小（图数量）
        batch_size = h.shape[0] // self.max_nodes

        # 拼接节点特征
        h_reshaped = h.view(batch_size, self.max_nodes, -1)
        x_reshaped = x.view(batch_size, self.max_nodes, -1)
        mask = (x_reshaped != 0).any(dim=2)
        combined = torch.cat([h_reshaped, x_reshaped], dim=2)

        # 对每个节点应用相同的MLP
        nodes = self.node_processor(combined)
        # 处理掩码
        nodes = nodes * mask.unsqueeze(-1)
        # 平均池化与最大池化，是置换不变的
        avg_pool = nodes.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        max_pool = nodes.max(dim=1)[0]
        rep = torch.cat([avg_pool, max_pool], dim=1)

        # 图分类
        output = self.graph_classifier(rep)
        return torch.sigmoid(torch.flatten(output))


class ReadoutConv1D(nn.Module):
    def __init__(self, h_shape1, x_shape1, max_nodes):
        super(ReadoutConv1D, self).__init__()
        self.max_nodes = max_nodes

        # 节点处理器
        self.node_processor = nn.Sequential(
            nn.Linear(h_shape1 + x_shape1, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # 1D卷积模块
        self.conv_layers = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )

        # 图分类器
        self.graph_classifier = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, h, x):
        # 批处理大小（图数量）
        batch_size = h.shape[0] // self.max_nodes

        # 拼接节点特征
        h_reshaped = h.view(batch_size, self.max_nodes, -1)
        x_reshaped = x.view(batch_size, self.max_nodes, -1)
        mask = (x_reshaped != 0).any(dim=2)
        combined = torch.cat([h_reshaped, x_reshaped], dim=2)

        # 节点处理
        nodes = self.node_processor(combined)
        # 处理掩码
        nodes = nodes * mask.unsqueeze(-1)
        # 应用1D卷积
        conv_output = self.conv_layers(nodes.transpose(1, 2))
        rep = conv_output.squeeze(-1)

        # 图分类
        output = self.graph_classifier(rep)
        return torch.sigmoid(torch.flatten(output))


class ReadoutPaperDirect(nn.Module):
    def __init__(self, h_shape1, x_shape1, max_nodes):
        super(ReadoutPaperDirect, self).__init__()
        self.max_nodes = max_nodes

        # 处理 [最终特征, 初始特征]
        self.conv_path1 = nn.Sequential(
            nn.Conv1d(
                in_channels=h_shape1 + x_shape1,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Conv1d(64, 16, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )

        # 处理最终特征
        self.conv_path2 = nn.Sequential(
            nn.Conv1d(
                in_channels=h_shape1,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Conv1d(64, 16, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )

        # 全局池化
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # 图分类器
        self.mlp1 = nn.Sequential(
            nn.Linear(16 * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(16 * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, h, x):
        # 批处理大小（图数量）
        batch_size = h.shape[0] // self.max_nodes

        # 重塑
        h_reshaped = h.view(batch_size, self.max_nodes, -1)
        x_reshaped = x.view(batch_size, self.max_nodes, -1)
        mask = (x_reshaped != 0).any(dim=2)

        # 处理掩码
        h_reshaped = h_reshaped * mask.unsqueeze(-1)
        x_reshaped = x_reshaped * mask.unsqueeze(-1)

        # 处理 [最终特征, 初始特征]
        path1_input = torch.cat([h_reshaped, x_reshaped], dim=2)
        path1_conv = self.conv_path1(path1_input.transpose(1, 2))
        path1_avg = self.global_avg_pool(path1_conv).squeeze(-1)
        path1_max = self.global_max_pool(path1_conv).squeeze(-1)
        path1_output = self.mlp1(torch.cat([path1_avg, path1_max], dim=1))

        # 处理最终特征
        path2_conv = self.conv_path2(h_reshaped.transpose(1, 2))
        path2_avg = self.global_avg_pool(path2_conv).squeeze(-1)
        path2_max = self.global_max_pool(path2_conv).squeeze(-1)
        path2_output = self.mlp2(torch.cat([path2_avg, path2_max], dim=1))

        # 相乘
        output = path1_output * path2_output
        return torch.sigmoid(torch.flatten(output))


class ReadoutPaperWeighted(nn.Module):
    def __init__(self, h_shape1, x_shape1, max_nodes):
        super(ReadoutPaperWeighted, self).__init__()
        self.max_nodes = max_nodes

        # 处理 [最终特征, 初始特征]
        self.conv_path1 = nn.Sequential(
            nn.Conv1d(
                in_channels=h_shape1 + x_shape1,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Conv1d(64, 16, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )

        # 处理最终特征
        self.conv_path2 = nn.Sequential(
            nn.Conv1d(
                in_channels=h_shape1,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Conv1d(64, 16, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )

        # 全局池化
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # 图分类器
        self.mlp1 = nn.Sequential(
            nn.Linear(16 * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(16 * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, h, x, rho=0.75):
        # 批处理大小（图数量）
        batch_size = h.shape[0] // self.max_nodes

        # 重塑
        h_reshaped = h.view(batch_size, self.max_nodes, -1)
        x_reshaped = x.view(batch_size, self.max_nodes, -1)
        mask = (x_reshaped != 0).any(dim=2)

        # 处理掩码
        h_reshaped = h_reshaped * mask.unsqueeze(-1)
        x_reshaped = x_reshaped * mask.unsqueeze(-1)

        # 处理 [最终特征, 初始特征]
        path1_input = torch.cat([h_reshaped, x_reshaped], dim=2)
        path1_conv = self.conv_path1(path1_input.transpose(1, 2))
        path1_avg = self.global_avg_pool(path1_conv).squeeze(-1)
        path1_max = self.global_max_pool(path1_conv).squeeze(-1)
        path1_output = self.mlp1(torch.cat([path1_avg, path1_max], dim=1))

        # 处理最终特征
        path2_conv = self.conv_path2(h_reshaped.transpose(1, 2))
        path2_avg = self.global_avg_pool(path2_conv).squeeze(-1)
        path2_max = self.global_max_pool(path2_conv).squeeze(-1)
        path2_output = self.mlp2(torch.cat([path2_avg, path2_max], dim=1))

        # 加权
        output = rho * path1_output + (1 - rho) * path2_output
        return torch.sigmoid(torch.flatten(output))


class Net(nn.Module):
    def __init__(self, gated_graph_conv_args, emb_size, max_nodes, device, readout_type=0):
        super(Net, self).__init__()
        self.ggc = GatedGraphConv(**gated_graph_conv_args).to(device)
        self.emb_size = emb_size

        if readout_type == 0:
            self.readout = ReadoutInvariantPool(gated_graph_conv_args["out_channels"], emb_size, max_nodes)
        elif readout_type == 1:
            self.readout = ReadoutConv1D(gated_graph_conv_args["out_channels"], emb_size, max_nodes)
        elif readout_type == 2:
            self.readout = ReadoutPaperDirect(gated_graph_conv_args["out_channels"], emb_size, max_nodes)
        else:
            self.readout = ReadoutPaperWeighted(gated_graph_conv_args["out_channels"], emb_size, max_nodes)

    def forward(self, data):

        x, edge_index = data.x, data.edge_index
        x = self.ggc(x, edge_index)

        x = self.readout(x, data.x)

        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
