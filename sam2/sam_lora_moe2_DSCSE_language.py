import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from safetensors import safe_open
from safetensors.torch import save_file
from icecream import ic
# from .modeling.sam2_base_mem import SAM2Base
from sam2.modeling.sam2_base import SAM2Base
import torch.nn.init as init



class DualTextProjector(nn.Module):
    def __init__(self, text_dim: int, vis_dim: int, max_len: int = 16, dropout: float = 0.1):
        super().__init__()
        self.proj_rgb = nn.Linear(text_dim, vis_dim)
        self.proj_th  = nn.Linear(text_dim, vis_dim)
        self.cls_rgb  = nn.Linear(text_dim, vis_dim)
        self.cls_th   = nn.Linear(text_dim, vis_dim)
        self.max_len = max_len
        self.drop = nn.Dropout(dropout)

    def forward(self, tok_rgb, tok_th):
        target_dtype = self.proj_rgb.weight.dtype
        if tok_rgb is not None:
            tok_rgb = tok_rgb.to(dtype=target_dtype)
        if tok_th is not None:
            tok_th  = tok_th.to(dtype=target_dtype)
        def proj_tok(tok, proj):
            if tok is None:
                return None
            B, L, _ = tok.shape
            Lp = min(L, self.max_len)
            return self.drop(proj(tok[:, :Lp, :]))

        Tq_rgb = proj_tok(tok_rgb, self.proj_rgb)
        Tq_th  = proj_tok(tok_th,  self.proj_th)
        return Tq_rgb, Tq_th



class TokenGuidedFusionBi(nn.Module):
    """
    双路 Token 引导融合（D‑TGF）
      - 路R：Q=Tq_rgb,  K/V=[RGB_flat (主), TH_flat(辅)]
      - 路T：Q=Tq_th,   K/V=[TH_flat (主), RGB_flat(辅)]
    输出：
      F_rgb_tok, F_th_tok: 两路 token 融合图 (B,C,H,W)
      attn_maps: (ar_m, ar_a, at_m, at_a)
    """
    def __init__(self, vis_dim: int, num_heads: int = 8, aux_scale: float = 0.5, reduce_tokens: int = 8):
        super().__init__()
        self.mha = nn.MultiheadAttention(vis_dim, num_heads, batch_first=True)
        self.reduce_tokens = reduce_tokens
        self.aux_scale = aux_scale
        self.merge = nn.Conv2d(vis_dim, vis_dim, 1, bias=False)
        self.t2w_rgb = nn.Sequential(nn.Linear(vis_dim, vis_dim // 2), nn.ReLU(inplace=True), nn.Linear(vis_dim // 2, 2))
        self.t2w_th  = nn.Sequential(nn.Linear(vis_dim, vis_dim // 2), nn.ReLU(inplace=True), nn.Linear(vis_dim // 2, 2))

    @staticmethod
    def _flat(x):
        B, C, H, W = x.shape
        return x.flatten(2).transpose(1, 2), (B, C, H, W)

    def _one_path(self, Q, main_feat, aux_feat, t2w):
        if Q is None:
            return None, None, None
        B, C, H, W = main_feat.shape
        main_f, _ = self._flat(main_feat)
        aux_f, _  = self._flat(aux_feat)
        KV = torch.cat([main_f, self.aux_scale * aux_f], dim=1)

        if self.reduce_tokens and Q.size(1) > self.reduce_tokens:
            idx = Q.norm(dim=-1).topk(k=self.reduce_tokens, dim=1).indices
            bidx = torch.arange(B, device=Q.device).unsqueeze(-1).expand_as(idx)
            Q = Q[bidx, idx]

        out_tok, attn = self.mha(Q, KV, KV, need_weights=True, average_attn_weights=True)
        attn_main = attn[..., : H * W].reshape(B, -1, H, W)
        attn_aux  = attn[..., H * W :].reshape(B, -1, H, W)

        w = torch.softmax(t2w(out_tok), dim=-1)
        # (B,K,1,1,1) for broadcasting with (B,K,C,H,W)
        w_main = w[..., 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        w_aux  = w[..., 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        def norm_a(a):
            return a / (a.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-6))

        aM = norm_a(attn_main).unsqueeze(2)
        aA = norm_a(attn_aux).unsqueeze(2)
        M = main_feat.unsqueeze(1)
        A = aux_feat.unsqueeze(1)
        fused_tok = w_main * (aM * M) + w_aux * (aA * A)
        fused = fused_tok.sum(dim=1)
        fused = self.merge(fused)
        return fused, attn_main, attn_aux

    def forward(self, rgb, th, Tq_rgb, Tq_th):
        F_r, ar_m, ar_a = self._one_path(Tq_rgb, rgb, th, self.t2w_rgb)
        F_t, at_m, at_a = self._one_path(Tq_th,  th,  rgb, self.t2w_th)
        return F_r, F_t, (ar_m, ar_a, at_m, at_a)


class DualTextGate(nn.Module):
    """样本级文本可靠度门控，输出 alpha/beta 权重。"""
    def __init__(self, vis_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(vis_dim * 3, vis_dim),
            nn.ReLU(inplace=True),
            nn.Linear(vis_dim, 2),
        )

    def forward(self, feat, c_rgb, c_th):
        g = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        zeros = torch.zeros_like(g)
        x = torch.cat([
            g,
            zeros if c_rgb is None else c_rgb,
            zeros if c_th is None else c_th,
        ], dim=1)
        w = torch.softmax(self.fc(x), dim=-1)
        alpha = w[:, 0:1].unsqueeze(-1).unsqueeze(-1)
        beta  = w[:, 1:2].unsqueeze(-1).unsqueeze(-1)
        return alpha, beta


class DSCSE(nn.Module):
    """Deformable Structure-Conditioned Cross-Modal Attention."""
    def __init__(self, dim: int, num_heads: int = 8, num_points: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.num_points = num_points
        self.q_proj = nn.Conv2d(dim, dim, 1)
        self.kv_proj_rgb = nn.Conv2d(dim, dim*2, 1)
        self.kv_proj_th  = nn.Conv2d(dim, dim*2, 1)
        self.offset_rgb2th = nn.Conv2d(dim, 2*num_points, 3, padding=1)
        self.offset_th2rgb = nn.Conv2d(dim, 2*num_points, 3, padding=1)
        self.out = nn.Conv2d(dim*2, dim, 1)
        self.norm = nn.BatchNorm2d(dim)

    def _sample(self, feat, offset):

        B, C, H, W = feat.shape # (B,C,H,W)  
        P = offset.shape[1] // 2 #  4
        grid_y, grid_x = torch.meshgrid( 
            torch.linspace(-1,1,H,device=feat.device),  
            torch.linspace(-1,1,W,device=feat.device), indexing='ij') 
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(B,1,1,1)  
        #4,40,40,2
        outs = []
        for p in range(P):  
            off = offset[:, 2*p:2*p+2] 
            off_norm = torch.stack([ 
                off[:,0]/(W/2),  
                off[:,1]/(H/2)   
            ], dim=-1) 
            grid = base_grid + off_norm
            outs.append(F.grid_sample(feat, grid, mode='bilinear', align_corners=True))
        return torch.stack(outs, dim=0).mean(0) 

    def forward(self, rgb, th):
        # queries 来自各自模态
        q_rgb = self.q_proj(rgb)
        q_th  = self.q_proj(th)

        k_rgb, v_rgb = torch.chunk(self.kv_proj_rgb(rgb), 2, dim=1) 
        k_th,  v_th  = torch.chunk(self.kv_proj_th(th),  2, dim=1)

        # 变形采样，缓解配准误差
        v_rgb2th = self._sample(v_rgb, self.offset_rgb2th(th)) # 变形采样，缓解配准误差
        v_th2rgb = self._sample(v_th,  self.offset_th2rgb(rgb))

        # 简化版 cross-attn（点乘注意力可替换为门控/卷积注意力）
        attn_rgb = torch.sigmoid(q_rgb) * v_th2rgb
        attn_th  = torch.sigmoid(q_th)  * v_rgb2th

        fused = self.out(torch.cat([attn_rgb, attn_th], dim=1))
        return self.norm(fused + 0.5*(rgb + th))  # 残差 + 双模态先验


class _LoRA_qkv(nn.Module):
    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
            linear_a_q2: nn.Module = None,  # Second LoRA module
            linear_b_q2: nn.Module = None,  # Second LoRA module
            linear_a_v2: nn.Module = None,  # Second LoRA module
            linear_b_v2: nn.Module = None,  # Second LoRA module
            linear_a_q3: nn.Module = None,  # Second LoRA module
            linear_b_q3: nn.Module = None,  # Second LoRA module
            linear_a_v3: nn.Module = None,  # Second LoRA module
            linear_b_v3: nn.Module = None,  # Second LoRA module
    ):
        super().__init__()
        self.qkv = qkv  # 输入的qkv线性层
        self.linear_a_q = linear_a_q  # 用于Q的低秩适配A
        self.linear_b_q = linear_b_q  # 用于Q的低秩适配B
        self.linear_a_v = linear_a_v  # 用于V的低秩适配A
        self.linear_b_v = linear_b_v  # 用于V的低秩适配B

        self.linear_a_q2 = linear_a_q2
        self.linear_b_q2 = linear_b_q2
        self.linear_a_v2 = linear_a_v2
        self.linear_b_v2 = linear_b_v2

        self.linear_a_q3 = linear_a_q3
        self.linear_b_q3 = linear_b_q3
        self.linear_a_v3 = linear_a_v3
        self.linear_b_v3 = linear_b_v3

        self.dim = qkv.in_features

    def forward(self, x):
        qkv = self.qkv(x)  # 原始QKV计算
        new_q = self.linear_b_q(self.linear_a_q(x))  # 计算低秩适配后的Q
        new_v = self.linear_b_v(self.linear_a_v(x))  # 计算低秩适配后的V
        qkv[:, :, :, : self.dim] += new_q  # 将低秩Q加到原始Q部分
        qkv[:, :, :, -self.dim:] += new_v  # 将低秩V加到原始V部分

        
        # Apply the second LoRA module if they exist
        if self.linear_a_q2 and self.linear_b_q2:
            new_q2 = self.linear_b_q2(self.linear_a_q2(x))
            qkv[:, :, :, :self.dim] += new_q2

        if self.linear_a_v2 and self.linear_b_v2:
            new_v2 = self.linear_b_v2(self.linear_a_v2(x))
            qkv[:, :, :, -self.dim:] += new_v2

        # Apply the second LoRA module if they exist
        if self.linear_a_q3 and self.linear_b_q3:
            new_q3 = self.linear_b_q3(self.linear_a_q3(x))
            qkv[:, :, :, :self.dim] += new_q3

        if self.linear_a_v3 and self.linear_b_v3:
            new_v3 = self.linear_b_v3(self.linear_a_v3(x))
            qkv[:, :, :, -self.dim:] += new_v3

        return qkv


class TopKRouter(nn.Module):
    def __init__(self, in_channels: int, num_experts: int, k: int = 2, hidden_dim: int = None):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        hidden_dim = hidden_dim or max(8, in_channels // 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_experts, kernel_size=1),
        )

    def forward(self, x):
        # x: (B, C, H, W)
        h = self.pool(x)                    # (B, C, 1, 1)
        logits = self.fc(h).flatten(1)      # (B, num_experts)
        probs = torch.softmax(logits, dim=1)  # (B, num_experts) 目的：将logits转换为概率， 其中logits是
        k = min(self.k, probs.size(1))
        topk_val, topk_idx = torch.topk(probs, k=k, dim=1)  # (B, k), (B, k)
        return topk_idx, topk_val, probs  # probs是概率,作用: 用于负载均衡损失计算。 topk_idx是索引, topk_val是值



class SparseMoE(nn.Module):
    """
    稀疏 MoE 专家池：只执行 Router 选择的 Top-k 专家，减少计算。
    每个样本使用相同的 Top-k（按样本维度做选择），实现简单、稳定。
    """
    def __init__(self, input_dim: int, num_experts: int, k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(input_dim),
                nn.ReLU(inplace=True),
            ) for _ in range(num_experts)
        ])
        self.router = TopKRouter(input_dim, num_experts, k)
        # 可选：指示该 MoE 属于哪一路（'rgb' 或 'th'），用于双文本路由
        self.text_side = None

    def forward(self, x): # 做了两次路由
        # 第一次路由：根据输入特征 x 和文本特征 text_cls 计算每个样本对每个专家的权重，对输入特征图x进行路由（选择 top-k 个专家）；
        # x: (B, C, H, W)
        # 路由：兼容默认 TopKRouter 与 AdaptiveTopKRouterDualText

        topk_idx, topk_w, probs = self.router(x) #1. 路由选择：确定每个样本该用哪些专家
        B, C, H, W = x.shape
        E = self.num_experts  # 专家个数

        #2. 构造路由权重：每个样本对专家的总权重，用于后续选择专家；构造每个样本对各专家的权重 (B, E)，仅 Top-k 为非零  让选中的专家网络处理对应样本；
        routing_weights = F.one_hot(topk_idx, num_classes=E).to(x.dtype)  # (B, k, E)  one_hot 作用: 将topk_idx转换为one-hot编码
        routing_weights = routing_weights * topk_w.unsqueeze(-1)  # (B, k, E)  topk_w.unsqueeze(-1) 作用: 将topk_w扩展为(B, k, 1)
        routing_weights = routing_weights.sum(dim=1)  # (B, E)  sum(dim=1) 作用: 将topk_w扩展为(B, E)

        #3. 专家网络计算：每个专家处理被分配的样本，输出特征图；
        # 第二次路由：根据权重 routing_weights 选择每个样本的专家
        # 按专家维度循环（E 通常较小），对被选中的样本批量计算并聚合
        y = torch.zeros_like(x) #  torch.Size([4, 2048, 40, 40]) 构建一个与x形状相同的张量，并将其初始化为0，用于存储每个专家的输出
        # print(y.shape)
        selected_mask = routing_weights > 0  # (B, E)  selected_mask 作用: 选择被选中的专家 （每个样本的专家权重大于0）  4*8
        for i in range(E):
            idx_b = selected_mask[:, i].nonzero(as_tuple=False).squeeze(1) #  torch.Size([4, 1]) 作用: 选择被选中的专家 （每个样本的专家权重大于0）  【4，1】
            if idx_b.numel() == 0:
                continue
            x_i = x.index_select(0, idx_b)  # (n_i, C, H, W) 作用: 选择被选中的专家 （每个样本的专家权重大于0）  【4，2048，40，40】
            out_i = self.experts[i](x_i)    # (n_i, C, H, W) 作用: 选择被选中的专家 （每个样本的专家权重大于0）  【4，2048，40，40】
            w_i = routing_weights.index_select(0, idx_b)[:, i].view(-1, 1, 1, 1)  #  torch.Size([4, 1, 1, 1]) 作用: 选择被选中的专家 （每个样本的专家权重大于0）  【4，1，1，1】
            y[idx_b] += out_i * w_i
            
        #4. 残差连接与输出
        out = x + y  # # 残差连接：原始特征 + 专家处理结果
        return out, probs  # 返回 router 概率用于负载均衡损失


class LoRA_Sam(nn.Module):
    """Applies low-rank adaptation to a Sam model's image encoder.

    Args:
        sam_model: a vision transformer model, see base_vit.py
        r: rank of LoRA
        num_classes: how many classes the model output, default to the vit model
        lora_layer: which layer we apply LoRA.

    Examples::
        >>> model = ViT('B_16_imagenet1k')
        >>> lora_model = LoRA_ViT(model, r=4)
        >>> preds = lora_model(img)
        >>> print(preds.shape)
        torch.Size([1, 1000])
    """

    def __init__(self, sam_model: SAM2Base, r: int, num_classes: int, lora_layer=None, num_experts: int = 1, top_k: int = 2, text_dim: int = None):
        super(LoRA_Sam, self).__init__()

        assert r > 0  # 确保 LoRA 的秩参数 r 为正数
        if lora_layer:
            self.lora_layer = lora_layer  # 使用指定的层
        else:
            self.lora_layer = list(range(
                len(sam_model.image_encoder.trunk.blocks)))  # 默认将 LoRA 应用于所有的 trunk blocks        # Only apply lora to the image encoder by default

        self.w_As = []  # 第一组 LoRA 层的权重 W_A
        self.w_Bs = []  # 第一组 LoRA 层的权重 W_B
        self.w_As2 = []  # 第二组 LoRA 层的权重 W_A
        self.w_Bs2 = []  # 第二组 LoRA 层的权重 W_B
        self.w_As3 = []  # 第三组 LoRA 层的权重 W_A
        self.w_Bs3 = []  # 第三组 LoRA 层的权重 W_B

        # Freeze original SAM model parameters
        # 冻结原始模型的所有参数
        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        # Apply LoRA to specified layers
        # 遍历图像编码器的每一层
        for t_layer_i, blk in enumerate(sam_model.image_encoder.trunk.blocks):
            if t_layer_i not in self.lora_layer:
                continue  # 如果该层不在指定的 lora_layer 中，跳过
            w_qkv_linear = blk.attn.qkv  # 提取该层的 qkv 线性层
            self.dim = w_qkv_linear.in_features  # 获取线性层的输入维度

            # First LoRA module
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)

            # Second LoRA module
            w_a_linear_q2 = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q2 = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v2 = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v2 = nn.Linear(r, self.dim, bias=False)
            self.w_As2.append(w_a_linear_q2)
            self.w_Bs2.append(w_b_linear_q2)
            self.w_As2.append(w_a_linear_v2)
            self.w_Bs2.append(w_b_linear_v2)

            # Second LoRA module
            w_a_linear_q3 = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q3 = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v3 = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v3 = nn.Linear(r, self.dim, bias=False)
            self.w_As3.append(w_a_linear_q3)
            self.w_Bs3.append(w_b_linear_q3)
            self.w_As3.append(w_a_linear_v3)
            self.w_Bs3.append(w_b_linear_v3)

            # 使用 _LoRA_qkv 替换原始 qkv 线性层，使其包含 LoRA 的权重更新逻辑。  _LoRA_qkv -> nn.Module -> object
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,

                w_a_linear_q2,
                w_b_linear_q2,
                w_a_linear_v2,
                w_b_linear_v2,

                w_a_linear_q3,
                w_b_linear_q3,
                w_a_linear_v3,
                w_b_linear_v3
            )
        self.reset_parameters()  # 调用 reset_parameters 方法，初始化 LoRA 模块的权重（如使用 Kaiming 初始化）。
        self.sam = sam_model

        # 获取SAM掩码解码器中的Transformer特征维度大小，用于后续MLP和卷积层的输入维度
        transformer_dim = self.sam.sam_mask_decoder.transformer_dim  # 表示 sam_mask_decoder 的 Transformer 的维度大小。

        # --- 专家分支定义 ---
        # A. RGB 稀疏MoE专家池
        self.rgb_moe = SparseMoE(input_dim=transformer_dim, num_experts=num_experts, k=top_k)

        # B. Thermal 稀疏MoE专家池
        self.thermal_moe = SparseMoE(input_dim=transformer_dim, num_experts=num_experts, k=top_k)

        # C. 增强型跨模态结构专家
        self.DSCSE_expert = DSCSE(dim=transformer_dim, num_heads=8, num_points=4)

        # 融合层（取代原FusionModule）
        self.fuse_conv = nn.Conv2d(transformer_dim * 3, transformer_dim, kernel_size=1)

        # 文本相关模块（即插即用，默认不传文本时不生效）
        proj_in_dim = transformer_dim if text_dim is None else text_dim
        self.dual_text = DualTextProjector(text_dim=proj_in_dim, vis_dim=transformer_dim, max_len=16, dropout=0.1)
        self.tgf_bi    = TokenGuidedFusionBi(vis_dim=transformer_dim, num_heads=8, aux_scale=0.5, reduce_tokens=8)

        self.thermal_moe.text_side = 'th'  # 告诉 SparseMoE 这一路吃 Thermal 文本

        # token + 结构分支的融合
        self.fuse_token_struct = TriBranchSoftmaxFuse(transformer_dim)

        # --- 辅助分割头 ---
        self.aux_head_rgb = nn.Conv2d(transformer_dim, num_classes, kernel_size=1)
        self.aux_head_thermal = nn.Conv2d(transformer_dim, num_classes, kernel_size=1)
        self.aux_head_structure = nn.Conv2d(transformer_dim, num_classes, kernel_size=1)
        # --- 专家分支定义结束 ---



    def save_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.

        save both lora and fc parameters.
        """

        # 确保文件名以 `.pt` 或 `.pth` 结尾，表示支持的保存格式
        assert filename.endswith(".pt") or filename.endswith('.pth')

        # 获取 LoRA 模块的层数（注意：self.w_As 包含多组权重，这里计算层数）
        num_layer = len(self.w_As)  # 实际上它是 LoRA 的一半层数

        # 将第一组 LoRA 模块的权重保存为字典
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}

        # 用于存储其他模块的参数
        prompt_encoder_tensors = {}  # 存储 prompt encoder 的参数
        mask_decoder_tensors = {}  # 存储 mask decoder 的参数

        # 获取模型的 state_dict（支持 DataParallel 和 DistributedDataParallel）
        if isinstance(self.sam, torch.nn.DataParallel) or isinstance(self.sam,
                                                                     torch.nn.parallel.DistributedDataParallel):
            state_dict = self.sam.module.state_dict()  # 如果是并行模型，获取实际的模块
        else:
            state_dict = self.sam.state_dict()  # 普通模型直接获取 state_dict

        # 遍历所有参数，将属于 prompt_encoder 和 mask_decoder 的参数分类存储
        for key, value in state_dict.items():
            if 'prompt_encoder' in key:
                prompt_encoder_tensors[key] = value  # 保存 prompt encoder 参数
            if 'mask_decoder' in key:
                mask_decoder_tensors[key] = value  # 保存 mask decoder 参数

        # 合并所有参数，包括 LoRA 参数和其他模块参数
        merged_dict = {**a_tensors, **b_tensors, **prompt_encoder_tensors, **mask_decoder_tensors}

        # 使用 safetensors 库保存参数到指定文件
        save_file(merged_dict, filename)

    def reset_parameters(self):

        for w_a in self.w_As:
            init.kaiming_uniform_(w_a.weight, a=math.sqrt(5))

        for w_b in self.w_Bs:
            init.zeros_(w_b.weight)

        for w_a2 in self.w_As2:
            init.kaiming_uniform_(w_a2.weight, a=math.sqrt(5))

        for w_b2 in self.w_Bs2:
            init.zeros_(w_b2.weight)

        for w_a3 in self.w_As3:
            init.kaiming_uniform_(w_a3.weight, a=math.sqrt(5))

        for w_b3 in self.w_Bs3:
            init.zeros_(w_b3.weight)


    def forward(self, batched_input, multimask_output, text_tokens_rgb: torch.Tensor = None, text_tokens_th: torch.Tensor = None):
        # batched_input 是一个包含 [rgb_batch, thermal_batch] 的列表
        # 每个张量的形状为 (b, 3, h, w)
        b = batched_input[0].shape[0]  # 原始批次大小
        m = len(batched_input)  # 模态数量 (m=2)

        # 沿批次维度拼接，以适应共享骨干网络
        stacked_images = torch.cat(batched_input, dim=0)  # 新形状: (m*b, 3, h, w)

        # 第1步：共享骨干网络与特征提取
        image_embedding = self.sam.forward_image(stacked_images)

        vision_features = image_embedding['vision_features']  # 形状: (m*b, C, fH, fW)
        _, C, fH, fW = vision_features.shape
        vision_features = vision_features.view(m, b, C, fH, fW)  # 重塑以分离模态

        # 将特征分离为 RGB 和 Thermal
        rgb_features = vision_features[0]  # (b, C, fH, fW)
        thermal_features = vision_features[1]  # (b, C, fH, fW)

        # 文本投影与双路 Token 引导（若无文本输入将得到 None 分支）
        Tq_rgb, Tq_th = self.dual_text(text_tokens_rgb, text_tokens_th)
        F_rTok, F_tTok, _ = self.tgf_bi(rgb_features, thermal_features, Tq_rgb, Tq_th)

        # 稀疏 MoE 专家池（传入对应文本 cls 到路由器）
        F_rgb_expert, rgb_router_probs = self.rgb_moe(rgb_features)
        F_thermal_expert, thermal_router_probs = self.thermal_moe(thermal_features)
        # 结构分支
        F_structure = self.DSCSE_expert(rgb_features, thermal_features)
        # 融合：token+结构 再与两路专家融合，保持 3C 输入到 fuse_conv
        zero = torch.zeros_like(rgb_features)
        F_rTok = F_rTok if F_rTok is not None else zero
        F_tTok = F_tTok if F_tTok is not None else zero
        F_token_struct = self.fuse_token_struct(F_rTok, F_tTok, F_structure)
        fused_features = self.fuse_conv(torch.cat([F_rgb_expert, F_thermal_expert, F_token_struct], dim=1))

        # --- 关键修复：处理高分辨率FPN特征以匹配融合后的批次大小 ---
        fpn_features = image_embedding['backbone_fpn']
        processed_high_res_features = []
        for feat in fpn_features[:2]:  # 只处理解码器需要的 feat_s0 和 feat_s1
            # 将 (m*b, C, H, W) 重塑为 (m, b, C, H, W) 并对模态维度(m)求平均
            _, C_fpn, H_fpn, W_fpn = feat.shape
            feat_avg = feat.view(m, b, C_fpn, H_fpn, W_fpn).mean(dim=0)
            processed_high_res_features.append(feat_avg)
        # --- FPN特征处理结束 ---

        # 通过 SAM 解码器生成分割掩码和其他特征
        multi_mask_output = self.sam._forward_sam_heads(
            fused_features,  # 使用最终融合后的特征 (b, C, fH, fW)
            high_res_features=processed_high_res_features,  # 使用处理后、批次大小匹配的高分特征
            multimask_output=multimask_output
        )

        final_output = multi_mask_output[1]

        # --- 关键修复：上采样所有输出以匹配标签尺寸 ---
        # 从输入图像获取目标上采样尺寸
        target_size = batched_input[0].shape[-2:]

        # 上采样主输出
        final_output_upsampled = F.interpolate(final_output, size=target_size, mode='bilinear', align_corners=False)

        if self.training:
            # --- 计算并上采样辅助损失的输出 ---
            rgb_output = self.aux_head_rgb(F_rgb_expert)
            thermal_output = self.aux_head_thermal(F_thermal_expert)
            structure_output = self.aux_head_structure(F_structure)

            rgb_output_upsampled = F.interpolate(rgb_output, size=target_size, mode='bilinear', align_corners=False)
            thermal_output_upsampled = F.interpolate(thermal_output, size=target_size, mode='bilinear', align_corners=False)
            structure_output_upsampled = F.interpolate(structure_output, size=target_size, mode='bilinear', align_corners=False)
            # 语言-视觉对齐的可选辅助损失（无文本时为0）
            language_align_loss = torch.tensor(0.0, device=final_output_upsampled.device)

            return {
                "final": final_output_upsampled,
                "rgb": rgb_output_upsampled,
                "thermal": thermal_output_upsampled,
                "structure": structure_output_upsampled,
                "language_align_loss": language_align_loss,
            }
        else:
            # 在评估模式下，只返回上采样后的最终输出
            # 先前的 reshape 和 mean 操作不适用于当前架构
            return final_output_upsampled


class TriBranchSoftmaxFuse(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Conv2d(dim * 3, 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
    def forward(self, a, b, c):
        w = torch.softmax(self.gate(torch.cat([a, b, c], dim=1)), dim=1)  # (B,3,H,W)
        fused = w[:, 0:1] * a + w[:, 1:2] * b + w[:, 2:3] * c
        return self.proj(fused)


class TriSE(nn.Module):
    def __init__(self, dim, r=4):
        super().__init__()
        hid = max(8, dim // r)
        self.fc = nn.Sequential(nn.Linear(dim * 3, hid), nn.ReLU(inplace=True), nn.Linear(hid, 3))
        self.proj = nn.Conv2d(dim, dim, 1)
    def forward(self, a, b, c):
        g = torch.cat([
            F.adaptive_avg_pool2d(a, 1).flatten(1),
            F.adaptive_avg_pool2d(b, 1).flatten(1),
            F.adaptive_avg_pool2d(c, 1).flatten(1),
        ], dim=1)
        w = torch.softmax(self.fc(g), dim=1).view(-1, 3, 1, 1)
        fused = w[:, 0:1] * a + w[:, 1:2] * b + w[:, 2:3] * c
        return self.proj(fused)
