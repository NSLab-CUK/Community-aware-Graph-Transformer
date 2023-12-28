import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, GINConv, SAGEConv

import torch.nn as nn
import dgl
import dgl.nn as dglnn
import dgl.sparse as dglsp
import torch
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
import os
import os.path
import torch.optim as optim
import numpy as np
from dgl.data import AsGraphPredDataset
from dgl.dataloading import GraphDataLoader
from ogb.graphproppred import collate_dgl, DglGraphPropPredDataset, Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder
from tqdm import tqdm
import dgl.function as fn
import pyro


class Transformer(nn.Module):
	def __init__(self, in_dim, out_dim, n_classes, hidden_dim, num_layers, num_heads, k_transition, aug_check,
				 sim_check,
				 phi_check, alfa, beta):
		super().__init__()
		self.h = None
		self.embedding_h = nn.Linear(in_dim, hidden_dim, bias=False)
		self.in_dim = in_dim
		self.hidden_dim = hidden_dim
		self.k_transition = k_transition

		self.aug_check = aug_check
		self.sim_check = sim_check
		self.phi_check = phi_check
		self.afla = alfa
		self.beta = beta


		self.gcn = GCN(in_dim, hidden_dim, self.afla, self.beta)

		self.layers = nn.ModuleList(
			[GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, sim_check, phi_check) for _ in range(num_layers)])

		self.layers.append(GraphTransformerLayer(hidden_dim, out_dim, num_heads, sim_check, phi_check))

		self.MLP_layer_x = Reconstruct_X(out_dim, in_dim)

		self.embedding_phi = nn.Linear(1, hidden_dim)
		self.embedding_sim = nn.Linear(k_transition, hidden_dim)

	def extract_features(self, g, adj_org, X, current_epoch, edge_index, sim, phi, B, k_transition, device, device_2):

		adj_sampled = None

		if self.aug_check == 1:
			edge_index_sampled, x_gcn, adj_sampled, check_nan = self.gcn(g, adj_org, X, B, edge_index, current_epoch,device_2)
			g = dgl_renew(g, X, edge_index_sampled, sim, phi, k_transition, device,device_2)

		h = self.embedding_h(X)

		phi = g.edata['phi']
		sim = g.edata['sim']
		phi = self.embedding_phi(phi.float())
		sim = self.embedding_sim(sim.float())


		for layer in self.layers:
			h = layer(h, g, phi, sim, current_epoch)

		return h, adj_sampled

	def forward(self, g, adj_org, sim, phi, B, k_transition, current_epoch, device, device_2):

		X = g.ndata['x'].to(device_2)

		edge_index = torch.stack([g.edges()[0], g.edges()[1]]).to(device_2)

		h, adj_sampled = self.extract_features(g, adj_org, X, current_epoch, edge_index, sim, phi, B, k_transition, device, device_2)

		x_hat = self.MLP_layer_x(h)

		self.h = h

		return h, x_hat, adj_sampled


def dgl_renew(g, x0, edge_index_sampled, sim, phi, k_transition, device,device_2):
	g = dgl.graph((edge_index_sampled[0], edge_index_sampled[1]))


	g.edata['sim'] = sim[edge_index_sampled[0], edge_index_sampled[1]]
	g.edata['phi'] = phi[edge_index_sampled[0], edge_index_sampled[1]]

	return g

class GraphTransformerLayer(nn.Module):
	"""Graph Transformer Layer"""

	def __init__(self, in_dim, out_dim, num_heads, sim_check, phi_check):
		super().__init__()

		self.sim_check = sim_check
		self.phi_check = phi_check

		self.in_channels = in_dim
		self.out_channels = out_dim
		self.num_heads = num_heads

		self.attention = MultiHeadAttentionLayer(in_dim, out_dim // num_heads, num_heads, sim_check, phi_check)

		self.O = nn.Linear(out_dim, out_dim)

		self.batchnorm1 = nn.BatchNorm1d(out_dim)
		self.batchnorm2 = nn.BatchNorm1d(out_dim)
		self.layer_norm1 = nn.LayerNorm(out_dim)
		self.layer_norm2 = nn.LayerNorm(out_dim)

		self.FFN_layer1 = nn.Linear(out_dim, out_dim * 2)
		self.FFN_layer2 = nn.Linear(out_dim * 2, out_dim)

	def forward(self, h, g, phi, sim, current_epoch):
		h_in1 = h  # for first residual connection

		attn_out = self.attention(h, g, phi, sim, current_epoch)

		h = attn_out.view(-1, self.out_channels)

		h = self.O(h)

		h = h_in1 + h  # residual connection

		h = self.layer_norm1(h)

		h_in2 = h  # for second residual connection

		# FFN
		h = self.FFN_layer1(h)
		h = F.relu(h)

		h = F.dropout(h, 0.5, training=self.training)
		h = self.FFN_layer2(h)
		h = h_in2 + h  # residual connection
		h = self.layer_norm2(h)

		return h


class MultiHeadAttentionLayer(nn.Module):
	# in_dim, out_dim, num_heads
	def __init__(self, in_dim, out_dim, num_heads, sim_check, phi_check):
		super().__init__()

		self.sim_check = sim_check
		self.phi_check = phi_check

		self.out_dim = out_dim
		self.num_heads = num_heads
		self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
		self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
		self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)

		self.hidden_size = in_dim  # 80
		self.num_heads = num_heads  # 8
		self.head_dim = out_dim // num_heads  # 10

		self.scaling = self.head_dim ** -0.5

		self.q_proj = nn.Linear(in_dim, in_dim)
		self.k_proj = nn.Linear(in_dim, in_dim)
		self.v_proj = nn.Linear(in_dim, in_dim)

		self.proj_phi = nn.Linear(in_dim, out_dim * num_heads, bias=True)

		self.sim = nn.Linear(in_dim, out_dim * num_heads, bias=True)

	def propagate_attention(self, g):
		# Compute attention score
		if self.sim_check == 1:
			g.apply_edges(src_dot_dst_sim('K_h', 'Q_h', 'sim_h', 'score'))
		else:
			g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))

		g.apply_edges(scaling('score', np.sqrt(self.out_dim)))

		if self.phi_check == 1:
			g.apply_edges(imp_add_attn('score', 'proj_phi'))

		# softmax
		g.apply_edges(exp('score'))

		eids = g.edges()
		g.send_and_recv(eids, dgl.function.u_mul_e('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))  # src_mul_edge
		g.send_and_recv(eids, dgl.function.copy_e('score', 'score'), fn.sum('score', 'z'))  # copy_edge


	def forward(self, h, g, phi, sim, current_epoch):
		Q_h = self.Q(h)
		K_h = self.K(h)
		V_h = self.V(h)

		sim_h = self.sim(sim)

		proj_phi = self.proj_phi(phi)
		# proj_sim = self.proj_sim(sim)

		g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
		g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
		g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)
		g.edata['sim_h'] = sim_h.view(-1, self.num_heads, self.out_dim)

		g.edata['proj_phi'] = proj_phi.view(-1, self.num_heads, self.out_dim)

		self.propagate_attention(g)

		h_out = g.ndata['wV'] / (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-6))  # adding eps to all values here

		return h_out


class GCN(torch.nn.Module):
	# g,adj_org, X, B, edge_index, current_epoch, self.afla, self.beta)
	def __init__(self, num_features, hidden_dim=64, alfa=0.1, beta=0.95):
		super(GCN, self).__init__()
		self.conv1 = GCNConv(num_features, hidden_dim * 2)
		self.conv2 = GCNConv(hidden_dim * 2, hidden_dim)
		self.anfa = alfa
		self.beta = beta

		self.MLPA = torch.nn.Sequential(
			torch.nn.Linear(hidden_dim, hidden_dim),
			torch.nn.ReLU(),
			torch.nn.Linear(hidden_dim, hidden_dim))

	def forward(self, g, adj_org, x, B, edge_index, current_epoch,device_2):

		edge_probs = self.anfa * B + self.beta * adj_org

		edge_probs[edge_probs > 1] = 1

		edge_probs = edge_probs.cuda()
		check_nan = False
		while True:
			# try:
			adj_sampled = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=1, probs=edge_probs).rsample()

			adj_sampled = adj_sampled.triu(1)
			adj_sampled = adj_sampled + adj_sampled.T

			edge_index_sampled = adj_sampled.to_sparse()._indices()

			g_new = dgl.graph((edge_index_sampled[0], edge_index_sampled[1]))

			check_nan = True

			if g.num_nodes() == g_new.num_nodes():
				if current_epoch % 50 == 0:
					print(f'adj_sampled size: {edge_index_sampled.size()}')
				break
			else:
				print("----------------------------------------")
		edge_index_sampled = edge_index_sampled.to(device_2)
		adj_sampled = adj_sampled.to(device_2)

		return edge_index_sampled, x, adj_sampled, check_nan


class Reconstruct_X(torch.nn.Module):
	def __init__(self, inp, outp, dims=128):
		super().__init__()


		self.mlp = torch.nn.Sequential(
			torch.nn.Linear(inp, dims * 2),
			torch.nn.SELU(),
			torch.nn.Linear(dims * 2, outp))

	def forward(self, x):
		x = self.mlp(x)
		return x


class MLPA(torch.nn.Module):

	def __init__(self, in_feats, dim_h, dim_z):
		super(MLPA, self).__init__()

		self.gcn_mean = torch.nn.Sequential(
			torch.nn.Linear(in_feats, dim_h),
			torch.nn.ReLU(),
			torch.nn.Linear(dim_h, dim_z)
		)

	def forward(self, hidden):
		Z = self.gcn_mean(hidden)

		adj_logits = Z @ Z.T
		return adj_logits


class MLP(torch.nn.Module):

	def __init__(self, num_features, num_classes, dims=16):
		super(MLP, self).__init__()
		self.mlp = torch.nn.Sequential(
			torch.nn.Linear(num_features, dims), torch.nn.ReLU(),
			torch.nn.Linear(dims, num_classes))

	def forward(self, x):
		x = self.mlp(x)
		return x


class Transformer_class(nn.Module):
	def __init__(self, in_dim, out_dim, n_classes, hidden_dim, num_layers, num_heads, graph_name, cp_filename, aug_check ,sim_check ,phi_check ):
		super().__init__()

		print(f'Loading Transformer_class {cp_filename}')
		self.model = torch.load(cp_filename)
		if isinstance(self.model, torch.nn.DataParallel):
			self.model = self.model.module

		self.model.aug_check = aug_check
		self.model.sim_check = sim_check
		self.model.phi_check = phi_check

		for p in self.model.parameters():
			p.requires_grad = True


		self.MLP = MLP(out_dim, n_classes)


	def forward(self, g, adj_org, sim, phi, B, k_transition, current_epoch, device, device_2):

		X = g.ndata['x'].to(device_2)
		edge_index = torch.stack([g.edges()[0], g.edges()[1]])

		h, _ = self.model.extract_features(g, adj_org, X, current_epoch, edge_index, sim, phi, B, k_transition, device, device_2)

		h = self.MLP(h)
		h = F.softmax(h, dim=1)

		return h


class MLPReadout(nn.Module):

	def __init__(self, input_dim, output_dim, L=2):  # L = nb_hidden_layers
		super().__init__()
		list_FC_layers = [nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=True) for l in range(L)]
		list_FC_layers.append(nn.Linear(input_dim // 2 ** L, output_dim, bias=True))

		self.FC_layers = nn.ModuleList(list_FC_layers)
		self.L = L

	def forward(self, x):
		y = x
		for l in range(self.L):
			y = self.FC_layers[l](y)
			y = F.relu(y)
		y = self.FC_layers[self.L](y)

		return y


class Transformer_cluster(nn.Module):
	def __init__(self, in_dim, out_dim, n_classes, hidden_dim, num_layers, num_heads, graph_name, cp_filename, aug_check,sim_check,phi_check):
		super().__init__()

		print(f'Loading Transformer_class {cp_filename}')
		self.model = torch.load(cp_filename)
		if isinstance(self.model, torch.nn.DataParallel):
			self.model = self.model.module
		self.model.aug_check = aug_check
		self.model.sim_check = sim_check
		self.model.phi_check = phi_check

		for p in self.model.parameters():
			p.requires_grad = True

		self.MLP = MLPReadout(out_dim, n_classes)



	def forward(self, g, adj_org, sim, phi, B, k_transition, current_epoch, device, device_2):
		X = g.ndata['x'].to(device_2)
		edge_index = torch.stack([g.edges()[0], g.edges()[1]])


		h, _ = self.model.extract_features(g, adj_org, X, current_epoch, edge_index, sim, phi, B, k_transition, device, device_2)

		h = self.MLP(h)
		h = F.softmax(h, dim=1)

		return h


"""
	Util functions
"""


def exp(field):
	def func(edges):
		# clamp for softmax numerical stability
		return {field: torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-8, 8))}

	return func


def src_dot_dst(src_field, dst_field, out_field):
	def func(edges):
		return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}

	return func


def src_dot_dst_sim(src_field, dst_field, edge_field, out_field):
	def func(edges):
		return {out_field: ((edges.src[src_field] + edges.data[edge_field]) * (
					edges.dst[dst_field] + edges.data[edge_field])).sum(-1, keepdim=True)}

	return func


# Improving implicit attention scores with explicit edge features, if available
def scaling(field, scale_constant):
	def func(edges):
		return {field: (((edges.data[field])) / scale_constant)}

	return func


def imp_exp_attn(implicit_attn, explicit_edge):
	"""
		implicit_attn: the output of K Q
		explicit_edge: the explicit edge features
	"""

	def func(edges):
		return {implicit_attn: (edges.data[implicit_attn] * edges.data[explicit_edge])}

	return func


def imp_add_attn(implicit_attn, explicit_edge):
	"""
		implicit_attn: the output of K Q
		explicit_edge: the explicit edge features
	"""

	def func(edges):
		return {implicit_attn: (edges.data[implicit_attn] + edges.data[explicit_edge])}

	return func


# To copy edge features to be passed to FFN_e
def out_edge_features(edge_feat):
	def func(edges):
		return {'e_out': edges.data[edge_feat]}

	return func