"""
RNA_ClassQuery_Model - Multi-scale Class-Query Classification Model for RNA

This module implements the main model for RNA 12-class multi-label classification.
The model combines:
1. Parallel CNN for multi-scale local feature extraction
2. GCN for graph-structured feature propagation
3. Class-Query attention for per-class prediction

Sub-modules:
- ParallelCNNBlock: Multi-scale CNN feature extraction
- GCNBlock: Graph Convolutional Network block
- ClassQueryHead: Class-Query classification head using Cross-Attention
- HierarchicalClassQueryHeadPooling: Hierarchical head with direct 12-class queries
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.utils import softmax
from typing import Optional, Tuple

# Import GROUP_TO_CLASS_INDICES for hierarchical head
# from utils.common import GROUP_TO_CLASS_INDICES


# ============================================================================
# Sub-modules for RNA_ClassQuery_Model
# ============================================================================

class ParallelCNNBlock(nn.Module):
    """
    Multi-scale CNN feature extraction block
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden_dim: int = 64,
        kernel_sizes: Tuple[int, ...] = (1, 3, 5, 7),
        use_layer_norm: bool = True,
        dropout: float = 0.1,
        seq_len: int = 1001
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.kernel_sizes = kernel_sizes
        self.seq_len = seq_len
        # Each branch outputs hidden_dim // len(kernel_sizes), concatenated to hidden_dim
        self.out_channels = hidden_dim
        self.branch_out_channels = hidden_dim // len(kernel_sizes)

        self.conv_branches = nn.ModuleList([
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=self.branch_out_channels,
                kernel_size=k,
                padding='same',
                bias=True
            )
            for k in kernel_sizes
        ])

        if use_layer_norm:
            self.norm = nn.LayerNorm(normalized_shape=(self.out_channels, seq_len))
        else:
            self.norm = nn.BatchNorm1d(self.out_channels)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() == 3 and x.size(1) == self.seq_len and x.size(2) == 4:
            x = x.transpose(1, 2)
        elif x.dim() == 2 and x.size(1) == 4:
            if batch is not None:
                # Dynamic calculation of batch_size and seq_len for variable-length sequences
                batch_size = batch.max().item() + 1
                num_nodes_per_sample = batch.bincount()
                seq_len = num_nodes_per_sample[0].item()  # Assume all samples have same length
                x = x.view(batch_size, seq_len, 4).transpose(1, 2)
            else:
                x = x.t().unsqueeze(0)

        branch_outputs = []
        for conv in self.conv_branches:
            out = conv(x)
            branch_outputs.append(out)

        concatenated = torch.cat(branch_outputs, dim=1)

        if isinstance(self.norm, nn.LayerNorm):
            normalized = self.norm(concatenated)
        else:
            normalized = self.norm(concatenated)

        features = self.dropout(self.activation(normalized))
        features = features.transpose(1, 2)
        features = features.reshape(-1, self.out_channels)

        return features


class GCNBlock(nn.Module):
    """
    Graph Convolutional Network block
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 128,
        out_channels: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
        use_residual: bool = True
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.use_residual = use_residual

        self.input_proj = None
        if in_channels != hidden_dim:
            self.input_proj = nn.Linear(in_channels, hidden_dim)

        self.gcn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            self.gcn_layers.append(
                GCNConv(in_dim, out_channels if i == num_layers - 1 else hidden_dim)
            )
            self.norms.append(nn.LayerNorm(out_channels if i == num_layers - 1 else hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.input_proj is not None:
            x = self.input_proj(x)

        residual = x

        for i, (gcn, norm) in enumerate(zip(self.gcn_layers, self.norms)):
            x = gcn(x, edge_index)

            if i < self.num_layers - 1:
                x = norm(x)
                x = self.activation(x)
                x = self.dropout(x)

                if self.use_residual and x.shape == residual.shape:
                    x = x + residual
                    residual = x
            else:
                x = norm(x)

        return x


class ClassQueryHead(nn.Module):
    """
    Class-Query classification head using Cross-Attention
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_classes: int = 12,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_decoder: bool = True,
        seq_len: int = 1001
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_decoder = use_decoder
        self.seq_len = seq_len

        # Learnable class queries
        self.class_queries = nn.Parameter(torch.randn(num_classes, hidden_dim))

        if use_decoder:
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True
            )
            self.cross_attention = nn.TransformerDecoder(decoder_layer, num_layers=1)
            self.output_proj = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()  # Ensure outputs are probabilities (0-1)
            )
        else:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.output_proj = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()  # Ensure outputs are probabilities (0-1)
            )

    def prune_heads(self, valid_class_indices):
        """
        Physically prune the class queries to only include valid indices.
        """
        with torch.no_grad():
            new_queries = self.class_queries.data[valid_class_indices].clone()
            self.class_queries = nn.Parameter(new_queries)
            self.num_classes = len(valid_class_indices)
            print(f"ClassQueryHead Pruned: {len(valid_class_indices)} classes remaining.")

    def forward(
        self,
        node_features: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        batch_size = batch.max().item() + 1
        device = node_features.device

        # Calculate actual max_nodes from batch
        num_nodes_per_sample = batch.bincount()
        max_nodes = num_nodes_per_sample[0].item()  # Assume all samples have same length

        queries = self.class_queries.unsqueeze(0).expand(batch_size, -1, -1).to(device)

        memory = torch.zeros(batch_size, max_nodes, self.hidden_dim, device=device)
        mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool, device=device)

        for b in range(batch_size):
            batch_mask = batch == b
            batch_nodes = node_features[batch_mask]

            num_batch_nodes = batch_nodes.size(0)
            memory[b, :num_batch_nodes, :] = batch_nodes
            mask[b, num_batch_nodes:] = True

        memory_mask = mask

        attended_features = self.cross_attention(
            tgt=queries,
            memory=memory,
            memory_key_padding_mask=memory_mask
        )

        logits = self.output_proj(attended_features)
        logits = logits.squeeze(-1)

        return logits.to(node_features.device)


class ClassQueryHeadPooling(nn.Module):
    """
    Simplified Class-Query head using attention pooling
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_classes: int = 12,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.class_queries = nn.Parameter(torch.randn(num_classes, hidden_dim))
        self.attention_scale = hidden_dim ** 0.5

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Ensure outputs are probabilities (0-1)
        )

    def prune_heads(self, valid_class_indices):
        with torch.no_grad():
            new_queries = self.class_queries.data[valid_class_indices].clone()
            self.class_queries = nn.Parameter(new_queries)
            self.num_classes = len(valid_class_indices)

    def forward(
        self,
        node_features: torch.Tensor,
        batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with attention weight return for supervision.

        Args:
            node_features: Node features from GCN
            batch: Batch assignment vector

        Returns:
            logits: Classification logits [Batch_Size, Num_Classes]
            attn_weights: Attention weights [Batch_Size, Num_Classes, Seq_Len=1001]
        """
        batch_size = batch.max().item() + 1
        device = node_features.device

        queries = self.class_queries.to(device)

        logits_list = []
        attn_weights_list = []

        for b in range(batch_size):
            batch_mask = batch == b
            batch_nodes = node_features[batch_mask]

            # scores: [Num_Classes, Seq_Len]
            scores = torch.matmul(queries, batch_nodes.t()) / self.attention_scale
            # attn_weights: [Num_Classes, Seq_Len]
            attn_weights = torch.softmax(scores, dim=1)
            aggregated = torch.matmul(attn_weights, batch_nodes)
            class_logits = self.output_proj(aggregated)

            logits_list.append(class_logits.squeeze(-1))
            attn_weights_list.append(attn_weights)

        logits = torch.stack(logits_list, dim=0)
        # Stack attention weights: [Batch_Size, Num_Classes, Seq_Len]
        all_attn_weights = torch.stack(attn_weights_list, dim=0)

        return logits.to(node_features.device), all_attn_weights.to(node_features.device)


class HierarchicalClassQueryHeadPooling(nn.Module):
    """
    Hierarchical Head with direct 12-class queries (no 4-group queries)
    """

    def __init__(self, hidden_dim, num_classes, dropout=0.1, use_layer_norm=True, num_heads=8, seq_len=1001):
        """
        Hierarchical Head with Attention Pooling.
        Uses 12-class queries directly, no group-to-class derivation.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.seq_len = seq_len

        # 1. Class Queries (Trainable parameters) [12, Hidden_Dim]
        # Direct 12-class queries instead of 4-group queries
        self.class_queries = nn.Parameter(torch.randn(num_classes, hidden_dim))

        # 2. MultiheadAttention for 12-class task
        self.mha_12 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 3. Output Projection for 12-class
        self.output_proj_12 = nn.Sequential(
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Ensure outputs are probabilities (0-1)
        )

    def prune_heads(self, valid_class_indices, valid_group_indices=None):
        """
        Prune the head to only compute specific classes via index masking.

        Args:
            valid_class_indices: List of valid class indices to keep
            valid_group_indices: Not used in simplified version (kept for compatibility)
        """
        self.register_buffer('valid_class_indices', torch.tensor(valid_class_indices, dtype=torch.long))
        print(f"Hierarchical Head Pruned: Active Classes={valid_class_indices}")

    def forward(self, node_features: torch.Tensor, batch: torch.Tensor):
        """
        Args:
            node_features: [Total_Nodes, Dim]
            batch: [Total_Nodes]
        Returns:
            logits_12: [Batch, 12] or [Batch, Num_Valid_Classes] if pruned
            attn_weights_12: [Batch, 12, Seq_Len] or [Batch, Num_Valid_Classes, Seq_Len] if pruned
        """
        batch_size = batch.max().item() + 1
        device = node_features.device

        # Calculate actual sequence length from batch
        num_nodes_per_sample = batch.bincount()
        actual_seq_len = num_nodes_per_sample[0].item()

        # 1. Prepare Class Queries
        class_queries = self.class_queries  # [12, Dim]

        # 2. Apply pruning if valid indices are registered
        if hasattr(self, 'valid_class_indices'):
            class_queries = class_queries[self.valid_class_indices]

        # 3. Prepare Inputs (Dense Batch)
        # Reshape node_features from [Total_Nodes, Dim] to [Batch, Seq_Len, Dim]
        dense_nodes = node_features.view(batch_size, actual_seq_len, -1)

        # 4. Prepare Queries - Expand to batch dimension
        # [Num_Classes, Dim] -> [Batch, Num_Classes, Dim]
        queries_12 = class_queries.unsqueeze(0).expand(batch_size, -1, -1)

        # 5. MHA for 12-Class Task (Direct queries_12 from dense_nodes)
        # attn_out_12: [Batch, Num_Classes, Dim]
        # attn_weights_12: [Batch, Num_Classes, Seq_Len]
        attn_out_12, attn_weights_12 = self.mha_12(
            query=queries_12,
            key=dense_nodes,
            value=dense_nodes,
            average_attn_weights=True
        )

        # 6. Final Projection
        # [Batch, Num_Classes, Dim] -> [Batch, Num_Classes, 1] -> [Batch, Num_Classes]
        logits_12_final = self.output_proj_12(attn_out_12).squeeze(-1)

        # Return: 12-class Logits, 12-class Attention Weights
        return logits_12_final.to(device), attn_weights_12.to(device)


class RNA_ClassQuery_Model_Treex(nn.Module):
    """
    RNA Classification Model using Multi-scale CNN + LSTM + Class-Query Attention (No GCN)
    """

    def __init__(
        self,
        cnn_hidden_dim: int = 64,
        cnn_kernel_sizes: Tuple[int, ...] = (1, 3, 5, 7),
        cnn_dropout: float = 0.1,
        lstm_hidden_dim: int = 128,
        lstm_num_layers: int = 2,
        lstm_dropout: float = 0.3,
        num_classes: int = 12,
        num_attn_heads: int = 4,
        attn_dropout: float = 0.1,
        use_simple_pooling: bool = False,
        use_hierarchical: bool = False,
        use_layer_norm: bool = True,
        seq_len: int = 1001
    ):
        super().__init__()

        self.cnn_out_channels = cnn_hidden_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_classes = num_classes
        self.use_hierarchical = use_hierarchical
        self.seq_len = seq_len

        self.cnn_block = ParallelCNNBlock(
            in_channels=4,
            hidden_dim=cnn_hidden_dim,
            kernel_sizes=cnn_kernel_sizes,
            use_layer_norm=use_layer_norm,
            dropout=cnn_dropout,
            seq_len=seq_len
        )

        # LSTM layer between CNN and Class-Query Head
        self.lstm = nn.LSTM(
            input_size=cnn_hidden_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout,
            batch_first=True
        )

        # Note: GCN block is removed

        if use_hierarchical:
            self.class_query_head = HierarchicalClassQueryHeadPooling(
                hidden_dim=lstm_hidden_dim,
                num_classes=num_classes,
                dropout=attn_dropout,
                seq_len=seq_len
            )
        elif use_simple_pooling:
            self.class_query_head = ClassQueryHeadPooling(
                hidden_dim=cnn_hidden_dim,
                num_classes=num_classes,
                dropout=attn_dropout
            )
        else:
            self.class_query_head = ClassQueryHead(
                hidden_dim=cnn_hidden_dim,
                num_classes=num_classes,
                num_heads=num_attn_heads,
                dropout=attn_dropout,
                use_decoder=True,
                seq_len=seq_len
            )

    def prune_heads(self, valid_class_indices, valid_group_indices=None):
        """
        Public interface to prune the classification head for specific tasks.
        """
        if self.use_hierarchical:
            if valid_group_indices is None:
                raise ValueError("Hierarchical model requires valid_group_indices for pruning.")
            self.class_query_head.prune_heads(valid_class_indices, valid_group_indices)
        else:
            # For standard head, we just pass the class indices
            if hasattr(self.class_query_head, 'prune_heads'):
                 self.class_query_head.prune_heads(valid_class_indices)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> tuple:
        """
        Forward pass without edge_index and batch parameters.

        Args:
            x: Input features [Batch_Size, Seq_Len, 4] or [Batch_Size, 4, Seq_Len]
            return_attention: If True, return attention weights (only works with use_simple_pooling=True)

        Returns:
            If use_hierarchical: (logits_12class, attn_weights_12)
            Elif use_simple_pooling and return_attention: (logits, attn_weights)
            Else: logits
        """
        if x.dim() == 3:
            batch_size = x.size(0)
            seq_len = x.size(1)
            if seq_len == 4 and x.size(2) == self.seq_len:
                # [Batch, 4, Seq_Len] -> transpose to [Batch, Seq_Len, 4]
                x = x.transpose(1, 2)
                seq_len = x.size(1)
            assert seq_len == self.seq_len, f"Expected sequence length {self.seq_len}, got {seq_len}"

            # Create batch vector for class query head
            batch = torch.arange(
                batch_size, device=x.device
            ).repeat_interleave(self.seq_len)
        elif x.dim() == 2:
            # [Batch, 4] -> single sequence
            batch_size = x.size(0)
            batch = torch.arange(batch_size, device=x.device)
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        # CNN feature extraction
        node_features = self.cnn_block(x, batch)
        # Note: GCN block is removed

        node_features_3d = node_features.view(batch_size, seq_len, -1)  # [256, 1001, 128]
        lstm_out, _ = self.lstm(node_features_3d)  # [256, 1001, 256]
        lstm_out = lstm_out.reshape(-1, lstm_out.size(-1))  # [256256, 256] 给后续用

        if self.use_hierarchical:
            # Hierarchical head returns 2 values: logits_12class, attn_weights_12
            logits_12class, attn_weights_12 = self.class_query_head(lstm_out, batch)

            if self.training:
                return logits_12class, attn_weights_12
            else:
                return logits_12class, attn_weights_12

        elif self.use_simple_pooling:
            logits, attn_weights = self.class_query_head(node_features, batch)
            if self.training:
                return logits, attn_weights
            return logits
        else:
            logits = self.class_query_head(node_features, batch)
            # For compatibility, we assume non-pooling heads don't return attention in this setup
            if self.training:
                return logits, None
            return logits


# class RNA_ClassQuery_Model(nn.Module):
#     """
#     RNA Classification Model using Multi-scale CNN + GCN + Class-Query Attention
#     """

#     def __init__(
#         self,
#         cnn_hidden_dim: int = 64,
#         cnn_kernel_sizes: Tuple[int, ...] = (1, 3, 5, 7),
#         cnn_dropout: float = 0.1,
#         gcn_hidden_dim: int = 128,
#         gcn_out_channels: int = 128,
#         gcn_num_layers: int = 3,
#         gcn_dropout: float = 0.3,
#         num_classes: int = 12,
#         num_attn_heads: int = 4,
#         attn_dropout: float = 0.1,
#         use_simple_pooling: bool = False,
#         use_hierarchical: bool = False,
#         use_layer_norm: bool = True,
#         seq_len: int = 1001
#     ):
#         super().__init__()

#         self.cnn_out_channels = cnn_hidden_dim
#         self.gcn_out_channels = gcn_out_channels
#         self.num_classes = num_classes
#         self.use_hierarchical = use_hierarchical
#         self.seq_len = seq_len

#         self.cnn_block = ParallelCNNBlock(
#             in_channels=4,
#             hidden_dim=cnn_hidden_dim,
#             kernel_sizes=cnn_kernel_sizes,
#             use_layer_norm=use_layer_norm,
#             dropout=cnn_dropout,
#             seq_len=seq_len
#         )

#         self.gcn_block = GCNBlock(
#             in_channels=self.cnn_out_channels,
#             hidden_dim=gcn_hidden_dim,
#             out_channels=gcn_out_channels,
#             num_layers=gcn_num_layers,
#             dropout=gcn_dropout,
#             use_residual=True
#         )

#         if use_hierarchical:
#             self.class_query_head = HierarchicalClassQueryHeadPooling(
#                 hidden_dim=gcn_out_channels,
#                 num_classes=num_classes,
#                 dropout=attn_dropout,
#                 seq_len=seq_len
#             )
#         elif use_simple_pooling:
#             self.class_query_head = ClassQueryHeadPooling(
#                 hidden_dim=gcn_out_channels,
#                 num_classes=num_classes,
#                 dropout=attn_dropout
#             )
#         else:
#             self.class_query_head = ClassQueryHead(
#                 hidden_dim=gcn_out_channels,
#                 num_classes=num_classes,
#                 num_heads=num_attn_heads,
#                 dropout=attn_dropout,
#                 use_decoder=True,
#                 seq_len=seq_len
#             )

#     def prune_heads(self, valid_class_indices, valid_group_indices=None):
#         """
#         Public interface to prune the classification head for specific tasks.
#         """
#         if self.use_hierarchical:
#             if valid_group_indices is None:
#                 raise ValueError("Hierarchical model requires valid_group_indices for pruning.")
#             self.class_query_head.prune_heads(valid_class_indices, valid_group_indices)
#         else:
#             # For standard head, we just pass the class indices
#             if hasattr(self.class_query_head, 'prune_heads'):
#                  self.class_query_head.prune_heads(valid_class_indices)

#     def forward(
#         self,
#         x: torch.Tensor,
#         edge_index: torch.Tensor,
#         batch: Optional[torch.Tensor] = None,
#         return_attention: bool = False
#     ) -> tuple:
#         """
#         Forward pass with optional attention weight return.

#         Args:
#             x: Input features
#             edge_index: Graph edge indices
#             batch: Batch assignment vector
#             return_attention: If True, return attention weights (only works with use_simple_pooling=True)

#         Returns:
#             If use_hierarchical and return_attention: (logits_12class, attn_weights_12)
#             If use_hierarchical and not return_attention: (logits_12class, logits_4class)
#             Elif use_simple_pooling and return_attention: (logits, attn_weights)
#             Else: logits
#         """
#         if isinstance(x, Data) or isinstance(x, Batch):
#             batch_obj = x
#             x = batch_obj.x
#             edge_index = batch_obj.edge_index
#             batch = batch_obj.batch

#         if x.dim() == 3:
#             batch_size = x.size(0)
#             seq_len = x.size(1)
#             assert seq_len == self.seq_len, f"Expected sequence length {self.seq_len}, got {seq_len}"
#             if batch is None:
#                 batch = torch.arange(
#                     batch_size, device=x.device
#                 ).repeat_interleave(seq_len)

#         elif x.dim() == 2:
#             assert batch is not None, "batch vector must be provided for PyG format input"
#         else:
#             raise ValueError(f"Unexpected input shape: {x.shape}")

#         node_features = self.cnn_block(x, batch)
#         node_features = self.gcn_block(node_features, edge_index)

#         if self.use_hierarchical:
#             # Hierarchical head returns 2 values: logits_12class, attn_weights_12
#             logits_12class, attn_weights_12 = self.class_query_head(node_features, batch)

#             if self.training:
#                 return logits_12class, attn_weights_12
#             else:
#                 return logits_12class, attn_weights_12

#         elif self.use_simple_pooling:
#             logits, attn_weights = self.class_query_head(node_features, batch)
#             if self.training:
#                 return logits, attn_weights
#             return logits
#         else:
#             logits = self.class_query_head(node_features, batch)
#             # For compatibility, we assume non-pooling heads don't return attention in this setup
#             if self.training:
#                 return logits, None
#             return logits
