import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import pywt



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class SelfAttention(nn.Module):
    def __init__(self, in_dims=2, d_model=64, num_heads=4):
        super(SelfAttention, self).__init__()
        self.embedding = nn.Linear(in_dims, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)

        self.edge_query = nn.Linear(d_model//num_heads, d_model//num_heads)
        self.edge_key = nn.Linear(d_model//num_heads, d_model//num_heads)

        self.scaled_factor = torch.sqrt(torch.Tensor([d_model])).cuda()
        self.softmax = nn.Softmax(dim=-1)
        self.num_heads = num_heads
    def split_heads(self, x):
        # x [batch_size seq_len d_model]
        x = x.reshape(x.shape[0], -1, self.num_heads, x.shape[-1] // self.num_heads).contiguous()
        return x.permute(0, 2, 1, 3)  # [batch_size nun_heads seq_len depth]
    def forward(self, x, edge_inital, G):
        # batch_size seq_len 2
        assert len(x.shape) == 3
        embeddings = self.embedding(x)  # batch_size seq_len d_model
        query = self.query(embeddings)  # batch_size seq_len d_model
        key = self.key(embeddings)      # batch_size seq_len d_model

        query = self.split_heads(query)  # B num_heads seq_len d_model
        key = self.split_heads(key)  # B num_heads seq_len d_model

        edge_query = self.edge_query(edge_inital)  # batch_size 4 seq_len d_model
        edge_key = self.edge_key(edge_inital)      # batch_size 4 seq_len d_model
        div = torch.sum(G, dim=1)[:, None, :, None]

        Gquery = query + torch.einsum('bmn,bhnc->bhmc', G.transpose(-2, -1), edge_query) / div  # q [batch, num_agent, heads, 64/heads]
        Gkey = key + torch.einsum('bmn,bhnc->bhmc', G.transpose(-2, -1), edge_key) / div
        g_attention = torch.matmul(Gquery, Gkey.permute(0, 1, 3, 2))  # (batch_size, num_heads, seq_len, seq_len)
        g_attention = self.softmax(g_attention / self.scaled_factor)

        return g_attention



class Edge_inital(nn.Module):
    def __init__(self, in_dims=2, d_model=64):
        super(Edge_inital, self).__init__()
        self.x_embedding = nn.Linear(in_dims, d_model//4)
        self.edge_embedding = nn.Linear(d_model//4, d_model//4)
    def forward(self, x, G):
        assert len(x.shape) == 3
        embeddings = self.x_embedding(x)  # batch_size seq_len d_model
        div = torch.sum(G, dim=-1)[:, :, None]
        edge_init = self.edge_embedding(torch.matmul(G, embeddings) / div)  # T N d_model
        edge_init = edge_init.unsqueeze(1).repeat(1, 4, 1, 1)
        return edge_init



class AsymmetricConvolution(nn.Module):
    def __init__(self, in_cha, out_cha):
        super(AsymmetricConvolution, self).__init__()
        self.conv1 = nn.Conv2d(in_cha, out_cha, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.conv2 = nn.Conv2d(in_cha, out_cha, kernel_size=(1, 3), padding=(0, 1))
        self.shortcut = lambda x: x
        if in_cha != out_cha:
            self.shortcut = nn.Sequential(nn.Conv2d(in_cha, out_cha, 1, bias=False))
        self.activation = nn.PReLU()
    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.activation(self.conv2(x) + self.conv1(x))
        return x + shortcut



class DConvolution(nn.Module):
    def __init__(self, in_cha, out_cha):
        super(DConvolution, self).__init__()
        self.asymmetric_convolutions = nn.ModuleList()
        for i in range(7):
            self.asymmetric_convolutions.append(AsymmetricConvolution(in_cha, out_cha))

        self.ve_output = nn.Sequential(nn.Linear(2, 8),
            nn.PReLU(),
            nn.Linear(8, 16))

        self.pl_output = nn.Sequential(nn.Linear(2, 8),
            nn.PReLU(),
            nn.Linear(8, 16))

    def forward(self, x, spatial_graph, obs_traj):

        ve_features = self.ve_output(spatial_graph)  # node_features (T N 16)
        pl_features = self.pl_output(obs_traj)  # node_features (T N 16)

        ve_temp = F.normalize(ve_features, p=2, dim=2)  # temp [batch, num_agent, 64]
        ve_mat = torch.matmul(ve_temp, ve_temp.permute(0, 2, 1))  # corr_mat [batch, num_agent, num_agent]

        pl_temp = F.normalize(pl_features, p=2, dim=2)  # temp [batch, num_agent, 64]
        pl_mat = torch.matmul(pl_temp, pl_temp.permute(0, 2, 1))  # corr_mat [batch, num_agent, num_agent]

        corr_mat = ve_mat * pl_mat

        x = x * corr_mat.unsqueeze(dim=1)

        for i in range(7):
            x = self.asymmetric_convolutions[i](x) + x

        return x




class S_Branch(nn.Module):
    def __init__(self, T_in, T_out):
        super(S_Branch, self).__init__()
        self.tcns = nn.Sequential(nn.Conv2d(T_in, T_out, 1, padding=0),
            nn.PReLU())
        self.Dconvolutions = DConvolution(4, 4)
        self.activation = nn.Sigmoid()
    def forward(self, x, spatial_graph, obs_traj):
        temporal_x = x.permute(1, 0, 2, 3)  # x (num_heads T N N)
        temporal_x = self.tcns(temporal_x) + temporal_x
        x = temporal_x.permute(1, 0, 2, 3)
        threshold = self.activation(self.Dconvolutions(x, spatial_graph, obs_traj))
        x_zero = torch.zeros_like(x, device='cuda')
        Sparse_x = torch.where(x > threshold, x, x_zero)
        return Sparse_x

class BinaryThresholdFunctionType(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        # 前向传播：应用自适应二值化阈值
        ctx.save_for_backward(input, threshold)
        return (input > 0).float()  # 阈值化操作
    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播：提供近似梯度
        input, threshold = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_threshold = None  # 默认不计算阈值的梯度
        # 对输入张量应用梯度近似
        grad_input[torch.abs(input) > threshold] = 0
        return grad_input, grad_threshold  # 返回梯度




class BinaryThreshold(nn.Module):
    def __init__(self):
        super(BinaryThreshold, self).__init__()
    def forward(self, input, threshold):
        return BinaryThresholdFunctionType.apply(input, threshold)



class STAdaptiveGroupEstimator(nn.Module):
    def __init__(self, in_dims=2):
        super().__init__()
        self.ste = BinaryThreshold()

        self.ve_output = nn.Sequential(nn.Linear(in_dims, 8),
            nn.PReLU(),
            nn.Linear(8, 16))

        self.pl_output = nn.Sequential(nn.Linear(in_dims, 8),
            nn.PReLU(),
            nn.Linear(8, 16))

        self.th = nn.Parameter(torch.Tensor([0.7]))
    def forward(self, node_features, obs_traj):
        # node_features = (T N 2)
        ve_features = self.ve_output(node_features)  # node_features (T N 16)
        pl_features = self.pl_output(obs_traj)  # node_features (T N 16)

        ve_temp = F.normalize(ve_features, p=2, dim=2)  # temp [batch, num_agent, 64]
        ve_mat = torch.matmul(ve_temp, ve_temp.permute(0, 2, 1))  # corr_mat [batch, num_agent, num_agent]

        pl_temp = F.normalize(pl_features, p=2, dim=2)  # temp [batch, num_agent, 64]
        pl_mat = torch.matmul(pl_temp, pl_temp.permute(0, 2, 1))  # corr_mat [batch, num_agent, num_agent]

        corr_mat = ve_mat * pl_mat

        G = self.ste((corr_mat - self.th.clamp(-0.9999, 0.9999)), self.th.clamp(-0.9999, 0.9999))  # G [batch, num_agent, num_agent]

        return G



class SparseWeightedAdjacency(nn.Module):
    def __init__(self, s_in_dims=2, t_in_dims=3, T=8, embedding_dims=64):
        super(SparseWeightedAdjacency, self).__init__()
        # AdaptiveGroupEstimator
        self.S_Group = STAdaptiveGroupEstimator(in_dims=2)
        # edge_inital
        self.S_edge_inital = Edge_inital(s_in_dims, embedding_dims)

        # dense interaction
        self.S_group_attention = SelfAttention(s_in_dims, embedding_dims)
        self.S_branch = S_Branch(T_in=T, T_out=T)
    def add_identity(self, x):
        T, num_heads, N, _ = x.shape
        identity = torch.eye(N, device=x.device).unsqueeze(0).unsqueeze(0)  # shape: [1, 1, N, N]
        return x + identity  # 自动广播到 [T, num_heads, N, N]

    def forward(self, graph, obs_traj):
        assert len(graph.shape) == 3
        spatial_graph = graph[:, :, 1:]  # (T N 2)

        S_G = self.S_Group(spatial_graph, obs_traj)  # (T N N)
        S_E = self.S_edge_inital(spatial_graph, S_G)  # (T 4 N 16)
        G_S = self.S_group_attention(spatial_graph, S_E, S_G)  # (T num_heads N N)
        G_S = self.S_branch(G_S, spatial_graph, obs_traj)  # (T num_heads N N)

        G_S1 = self.add_identity(G_S)

        return G_S1, S_G, S_E



class GraphConvolution(nn.Module):
    def __init__(self, in_dims=2, embedding_dims=16, dropout=0):
        super(GraphConvolution, self).__init__()
        self.edge_value = nn.Linear(embedding_dims, in_dims)
        self.C_edge_value = nn.Linear(embedding_dims, in_dims)

        self.g_embedding = nn.Linear(in_dims, embedding_dims, bias=False)
        self.Cg_embedding = nn.Linear(in_dims, embedding_dims, bias=False)

        self.activation = nn.PReLU()
        self.dropout = dropout
    def forward(self, graph, g_adjacency, G, edge_inital):
        # graph=[T, 1, N, 2](seq_len 1 num_p 2)
        div = torch.sum(G, dim=1)[:, None, :, None]
        edge = self.edge_value(edge_inital)
        value = graph + torch.einsum('bmn,bhnc->bhmc', G.transpose(-2, -1), edge) / div
        g_gcn_features = self.g_embedding(torch.matmul(g_adjacency, value))

        gcn_features = F.dropout(self.activation(g_gcn_features), p=self.dropout)
        return gcn_features  # [batch_size num_heads seq_len hidden_size]



class SparseGraphConvolution(nn.Module):
    def __init__(self, in_dims=16, embedding_dims=16, dropout=0):
        super(SparseGraphConvolution, self).__init__()
        self.dropout = dropout

        self.spatial_gcn = GraphConvolution(in_dims, embedding_dims)

    def forward(self, graph, G_S, S_G, S_E):
        # graph [1 seq_len num_pedestrians  3]
        graph = graph[:, :, :, 1:]
        spa_graph = graph.permute(1, 0, 2, 3)  # (seq_len 1 num_p 2)

        spatial_features = self.spatial_gcn(spa_graph, G_S, S_G, S_E)
        spatial_features = spatial_features.permute(2, 0, 1, 3)  # spatial_features [N, T, heads, 16]

        return spatial_features  # [N, T, heads, 16]



class TrajectoryModel(nn.Module):
    def __init__(self,embedding_dims=64, number_gcn_layers=1, dropout=0,obs_len=8, pred_len=12, n_tcn=5, num_heads=4):
        super(TrajectoryModel, self).__init__()
        self.number_gcn_layers = number_gcn_layers
        self.n_tcn = n_tcn
        self.dropout = dropout
        # sparse graph learning
        self.sparse_weighted_adjacency_matrices = SparseWeightedAdjacency(s_in_dims=2, T=8, embedding_dims=64)
        self.wt1_weighted_adjacency_matrices = SparseWeightedAdjacency(s_in_dims=2, T=4, embedding_dims=64)
        self.wt2_weighted_adjacency_matrices = SparseWeightedAdjacency(s_in_dims=2, T=2, embedding_dims=64)

        # graph convolution
        self.stsgcn = SparseGraphConvolution(in_dims=2, embedding_dims=embedding_dims // num_heads, dropout=dropout)
        self.stsgcn_CA = SparseGraphConvolution(in_dims=2, embedding_dims=embedding_dims // num_heads, dropout=dropout)
        self.stsgcn2_CA = SparseGraphConvolution(in_dims=2, embedding_dims=embedding_dims // num_heads, dropout=dropout)

        self.tcns = nn.ModuleList()
        self.tcns.append(nn.Sequential(nn.Conv2d(obs_len, pred_len, 3, padding=1),
            nn.PReLU()))
        for j in range(1, self.n_tcn):
            self.tcns.append(nn.Sequential(nn.Conv2d(pred_len, pred_len, 3, padding=1),
                nn.PReLU()))

        self.tcns_CA = nn.ModuleList()
        self.tcns_CA.append(nn.Sequential(nn.Conv2d(4, pred_len, 3, padding=1),
            nn.PReLU()))
        for j in range(1, self.n_tcn):
            self.tcns_CA.append(nn.Sequential(nn.Conv2d(pred_len, pred_len, 3, padding=1),
                nn.PReLU()))

        self.tcns2_CA = nn.ModuleList()
        self.tcns2_CA.append(nn.Sequential(nn.Conv2d(2, pred_len, 3, padding=1),
            nn.PReLU()))
        for j in range(1, self.n_tcn):
            self.tcns2_CA.append(nn.Sequential(nn.Conv2d(pred_len, pred_len, 3, padding=1),
                nn.PReLU()))

        self.output = nn.Linear(embedding_dims // num_heads, 2)
        self.multi_output = nn.Sequential(nn.Conv2d(num_heads, 16, 1, padding=0),
            nn.PReLU(),
            nn.Conv2d(16, 20, 1, padding=0),)
    def wavelet_transform_along_T(self, data, wavelet='db1', level=1):
        B, T, N, C = data.shape
        assert T % (2 ** level) == 0, "T must be divisible by 2^level"
        new_T = T // (2 ** level)
        # 初始化存储容器，形状为 (B, N, C, new_T)
        cA_array = np.zeros((B, N, C, new_T))
        cD_array = np.zeros((B, N, C, new_T))
        for b in range(B):  # 遍历批量
            for n in range(N):  # 遍历通道
                for c in range(C):  # 遍历特征维度
                    # 提取时间序列并转换为 NumPy 数组
                    signal = data[b, :, n, c].cpu().numpy()
                    # 小波分解
                    coeffs = pywt.wavedec(signal, wavelet, level=level)
                    # 填充系数
                    cA_array[b, n, c, :] = coeffs[0]  # 近似系数
                    cD_array[b, n, c, :] = coeffs[1]  # 细节系数
        # 调整维度顺序为 (B, new_T, N, C)
        cA_array = np.transpose(cA_array, (0, 3, 1, 2))
        cD_array = np.transpose(cD_array, (0, 3, 1, 2))

        # 转换为 PyTorch 张量
        cA_tensor = torch.tensor(cA_array, dtype=data.dtype, device=data.device)
        cD_tensor = torch.tensor(cD_array, dtype=data.dtype, device=data.device)

        return cA_tensor, cD_tensor

    def forward(self, graph, obs_traj):
        # graph 1 obs_len N 3   # obs_traj 1 obs_len N 2    # obs_traj 1 N 2 obs_len

        obs_traj = obs_traj.permute(0, 3, 1, 2)

        graph1_CA, graph1_CD = self.wavelet_transform_along_T(graph, level=1)
        graph2_CA, graph2_CD = self.wavelet_transform_along_T(graph, level=2)

        obs_traj1_CA, obs_traj1_CD = self.wavelet_transform_along_T(obs_traj, level=1)
        obs_traj2_CA, obs_traj2_CD = self.wavelet_transform_along_T(obs_traj, level=2)

        G_S, S_G, S_E = self.sparse_weighted_adjacency_matrices(graph.squeeze(), obs_traj.squeeze())
        G_S_CA, S_G_CA, S_E_CA = self.wt1_weighted_adjacency_matrices(graph1_CA.squeeze(), obs_traj1_CA.squeeze())
        G_S2_CA, S_G2_CA, S_E2_CA = self.wt2_weighted_adjacency_matrices(graph2_CA.squeeze(), obs_traj2_CA.squeeze())

        # gcn_representation = [N, T, heads, 16]
        gcn_representation = self.stsgcn(graph, G_S, S_G, S_E)
        gcn_representation_CA = self.stsgcn_CA(graph1_CA, G_S_CA, S_G_CA, S_E_CA)
        gcn_representation2_CA = self.stsgcn2_CA(graph2_CA, G_S2_CA, S_G2_CA, S_E2_CA)


        features = self.tcns[0](gcn_representation)
        for k in range(1, self.n_tcn):
            features = F.dropout(self.tcns[k](features) + features, p=self.dropout)

        features_CA = self.tcns_CA[0](gcn_representation_CA)
        for k in range(1, self.n_tcn):
            features_CA = F.dropout(self.tcns_CA[k](features_CA) + features_CA, p=self.dropout)

        features2_CA = self.tcns2_CA[0](gcn_representation2_CA)
        for k in range(1, self.n_tcn):
            features2_CA = F.dropout(self.tcns2_CA[k](features2_CA) + features2_CA, p=self.dropout)


        prediction = self.output(features + features_CA + features2_CA)   # prediction=[N, Tpred, nums, 2]
        prediction = self.multi_output(prediction.permute(0, 2, 1, 3))   # prediction=[N, 20, Tpred, 2]

        return prediction.permute(1, 2, 0, 3).contiguous()



