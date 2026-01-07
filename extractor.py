"""
extractor.py

Dinov3Extractor：
- 负责加载 DINOv3 模型（优先按 000.ipynb 的方式从本地 torch.hub 路径加载），
- 提取指定数据加载器（DataLoader）中的图像特征（支持同时提取多个中间层），
- 使用指定的聚合器（Avg/Max/GeM/VLAD）在这些层进行全局聚合，得到检索向量，
- 并按 EM-CVGL 的保存格式将 dro/sat 的特征、id、文件名写入磁盘。

严格对齐点：
- 数据转换（transform）沿用 utils/transform.py 的定义与 configs 中 eval.transform 的配置；
- 特征保存命名沿用 test.py：sat_feat/sat_id/sat_name 与 dro_feat/dro_id/dro_name；
- 路径前缀沿用 HOME（~）+ 相对路径的组织方式，保持与 data/dataset.py 一致的可读性与兼容性。
"""

from typing import List, Tuple, Optional, Union, Dict
import os
import os.path as osp
from tqdm import tqdm

import torch
import torch.nn.functional as F

from constants import MODEL_TO_NUM_LAYERS


def _mkdir_if_missing(path: str):
    if not osp.exists(path):
        os.makedirs(path, exist_ok=True)


class Dinov3Extractor:
    """
    DINOv3 特征提取器：
    - 支持从本地 torch.hub 仓库加载模型（source='local'），也可回退到远程（不推荐，受网路影响）；
    - 使用 ViT 的 get_intermediate_layers 获取中间层特征（reshape=True, norm=True），支持多层提取；
    - 对指定层的 [B, C, H, W] 特征进行聚合，得到 [B, D] 的检索向量；
    - 将 dro/sat 的特征与 id、name 打包保存。
    """

    def __init__(
        self,
        model_name: str = "dinov3_vits16",
        dinov3_local_path: Optional[str] = None,
        weights: Optional[str] = None,
        device: Optional[str] = None,
        aggregator=None,
        desc_layer: Union[int, List[int], None] = None,
        desc_facet: str = 'token',
        use_cls: bool = False,
    ) -> None:
        """
        参数：
        - model_name：DINOv3 模型名称，参考 constants.py
        - dinov3_local_path：本地 DINOv3 仓库路径，用于 torch.hub.load(source='local')
        - weights：可选权重文件路径（若模型构造支持）
        - device：'cuda' 或 'cpu'，默认自动探测
        - aggregator：聚合器（可调用对象）：输入 [C,H,W] 输出 [D]；可使用 aggregator.get_aggregator()
        - desc_layer：可以是单个整数或整数列表，指定要提取的层索引
        """
        self.model_name = model_name
        self.local_path = dinov3_local_path
        self.weights = weights
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.aggregator = aggregator
        
        # 处理 desc_layer，统一转换为列表
        self.num_layers = MODEL_TO_NUM_LAYERS.get(self.model_name, None)
        # 如果模型名不在常量表中，给个默认值防止立即报错，后续加载模型后再校验
        default_layers = 12
        num_layers_check = self.num_layers if self.num_layers is not None else default_layers
        
        if desc_layer is None:
            self.desc_layers = [num_layers_check - 1]
        elif isinstance(desc_layer, int):
            self.desc_layers = [desc_layer]
        else:
            self.desc_layers = list(desc_layer)

        self.desc_facet = str(desc_facet).lower()  # 'token' | 'query' | 'key' | 'value'
        self.use_cls = use_cls

        self.model = self._load_model()
        
        # 加载模型后再次确认层数限制（如果之前没获取到）
        if self.num_layers is None:
             # 尝试从模型属性推断（部分ViT实现有 n_blocks 或 blocks）
             if hasattr(self.model, 'blocks'):
                 self.num_layers = len(self.model.blocks)
             else:
                 # 这是一个兜底，假设为12（ViT-B）
                 self.num_layers = 12

        # 估计 patch_size（ViT 模型）
        self.patch_size = 16
        try:
            # 对 ViT，通常为 Conv2d kernel/stride 为 patch_size
            self.patch_size = int(getattr(self.model.patch_embed.proj, 'kernel_size')[0])
        except Exception:
            pass

        self.model = self.model.eval().to(self.device)
        # Hook 存放
        self._hook_outs = {}
        self._fh_handles = []

    def _load_model(self):
        """加载 DINOv3 模型，优先本地路径。"""
        if self.local_path is not None:
            # 某些 hub 实现不支持 weights 参数，增加容错
            try:
                model = torch.hub.load(
                    self.local_path,
                    self.model_name,
                    source='local',
                    pretrained=True,
                    weights=self.weights
                )
            except TypeError:
                model = torch.hub.load(
                    self.local_path,
                    self.model_name,
                    source='local',
                    pretrained=True
                )
        else:
            # 远程加载（可能受网络影响，且官方 hub 名称可能变动）
            try:
                model = torch.hub.load(
                    'facebookresearch/dinov3',
                    self.model_name,
                    pretrained=True,
                    weights=self.weights
                )
            except TypeError:
                model = torch.hub.load(
                    'facebookresearch/dinov3',
                    self.model_name,
                    pretrained=True
                )
        return model

    @torch.inference_mode()
    def _extract_batch_feats(self, batch_x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        从一批图像中提取指定多个层的特征并聚合。
        输入：
        - batch_x: [B, 3, H, W]
        输出：
        - results: 字典 {layer_idx: Tensor[B, D]}
        """
        batch_x = batch_x.to(self.device)
        
        # 处理负数索引转换为正数
        target_layers = []
        for l in self.desc_layers:
            if l < 0:
                target_layers.append(self.num_layers + l)
            else:
                target_layers.append(l)
        
        results = {}

        if self.desc_facet == 'token':
            # ViT：使用 get_intermediate_layers 一次性获取多层
            # reshape=True => [B, C, H', W']；norm=True => 对 token 做归一
            # n 参数接收一个索引列表
            feats_list: List[torch.Tensor] = self.model.get_intermediate_layers(
                batch_x, n=target_layers, reshape=True, norm=True
            )
            # feats_list 的顺序对应 target_layers 的顺序
            for i, layer_idx in enumerate(self.desc_layers):
                feat_l = feats_list[i]  # [B, C, H', W']
                results[layer_idx] = self._aggregate_feat_batch(feat_l)
        else:
            # 通过 forward hook 从指定层的 attn.qkv 提取 query/key/value 三种 facet
            # 这种方式需要对每一层注册 Hook
            self._hook_outs = {}
            self._fh_handles = []

            # 定义 Hook 函数
            def get_hook(idx):
                def _hook(module, inputs, output):
                    self._hook_outs[idx] = output
                return _hook

            # 注册 Hooks
            for l_idx in target_layers:
                try:
                    # 注意：target_layers 里是转换后的正整数索引
                    blk = self.model.blocks[l_idx]
                    handle = blk.attn.qkv.register_forward_hook(get_hook(l_idx))
                    self._fh_handles.append(handle)
                except Exception as e:
                    print(f"[Error] Failed to register hook for layer {l_idx}: {e}")
                    # 如果不支持 facet 提取，这里会报错

            # 触发前向传播
            _ = self.model(batch_x)

            # 处理每个层的输出
            # target_layers 和 self.desc_layers 是一一对应的
            for i, l_idx in enumerate(target_layers):
                original_layer_idx = self.desc_layers[i]
                if l_idx in self._hook_outs:
                    qkv = self._hook_outs[l_idx]
                    feat_l = self._process_qkv(qkv, batch_x)
                    results[original_layer_idx] = self._aggregate_feat_batch(feat_l)
                else:
                    raise RuntimeError(f"未捕获到层 {l_idx} 的 qkv 输出。")

            # 清理 Hooks
            for h in self._fh_handles:
                h.remove()
            self._fh_handles = []
            self._hook_outs = {}

        return results

    def _aggregate_feat_batch(self, feat_l: torch.Tensor) -> torch.Tensor:
        """
        对单个层的特征 Batch 进行聚合
        输入: feat_l [B, C, H', W']
        输出: vecs [B, D]
        """
        out_list = []
        for i in range(feat_l.shape[0]):
            f_i = feat_l[i].detach()  # [C, H', W']
            # 与 000.ipynb 一致：按通道做一次 L2 归一（可选）
            f_i = F.normalize(f_i, p=2, dim=0)
            vec_i = self.aggregator(f_i) if self.aggregator is not None else f_i.mean(dim=(1, 2))
            vec_i = F.normalize(vec_i, dim=0)  # 最终向量再做 L2
            out_list.append(vec_i)
        vecs = torch.stack(out_list, dim=0)  # [B, D]
        return vecs

    def _generate_forward_hook(self):
        # 这是一个旧的辅助函数，保留兼容性，但多层提取逻辑中使用了闭包 get_hook
        def _forward_hook(module, inputs, output):
            self._hook_out = output
        return _forward_hook

    def _process_qkv(self, qkv: torch.Tensor, batch_x: torch.Tensor) -> torch.Tensor:
        """
        解析 qkv 并 reshape 成 [B, C, H', W']
        """
        B, N, threeC = qkv.shape
        C = threeC // 3
        if self.desc_facet == 'query':
            res = qkv[:, :, :C]
        elif self.desc_facet == 'key':
            res = qkv[:, :, C:2 * C]
        elif self.desc_facet == 'value':
            res = qkv[:, :, 2 * C:]
        else:
            raise ValueError(f"未知的 desc_facet: {self.desc_facet}")

        # 是否包含 CLS token
        if self.use_cls:
            tokens = res  # [B, N, C]
        else:
            tokens = res[:, 1:, :]  # [B, N-1, C]

        # 计算网格尺寸 H'、W'（使用 patch_size 与输入图像大小）
        H, W = batch_x.shape[-2], batch_x.shape[-1]
        Hp, Wp = H // self.patch_size, W // self.patch_size
        if Hp * Wp != tokens.shape[1]:
            # 兜底：根据 token 数量估计网格（假设为正方形）
            import math
            grid = int(math.sqrt(tokens.shape[1]))
            Hp, Wp = grid, tokens.shape[1] // grid

        # reshape 成 [B, C, H', W']
        feat = tokens.reshape(B, Hp, Wp, C).permute(0, 3, 1, 2).contiguous()
        # 归一化（与 000.ipynb 保持一致）
        feat = F.normalize(feat, p=2, dim=1)
        return feat

    @torch.inference_mode()
    def _extract_facet_features(self, batch_x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        保留旧接口以防兼容性问题，但主要逻辑已被 _extract_batch_feats 接管
        """
        # 复用 _extract_batch_feats 的逻辑，临时修改 desc_layers
        old_layers = self.desc_layers
        self.desc_layers = [layer_idx]
        try:
            results = self._extract_batch_feats(batch_x)
            # 这里 _extract_batch_feats 返回的是聚合后的向量 [B, D]，
            # 但旧接口期望返回 reshape 后的特征图 [B, C, H', W'] 用于聚合。
            # 由于重构了逻辑，这个旧接口仅作参考，现在的调用链直接使用 _extract_batch_feats 返回最终向量。
            # 为了完全兼容旧代码（如果还有其他地方调用），这里抛出异常或重新实现
            # 鉴于此文件只在 extract_and_save.py 中被调用，且我们修改了 extract_loader，
            # 此方法其实不再被使用了。
            pass
        finally:
            self.desc_layers = old_layers
        return torch.empty(0) # Placeholder

    def extract_loader(self, dataloader) -> Tuple[Dict[int, torch.Tensor], torch.Tensor, List[str]]:
        """
        对给定的 DataLoader 提取并聚合特征
        返回：
        - final_feats: 字典 {layer_idx: Tensor[N, D]}
        - all_ids:   [N]
        - all_names: List[str]
        """
        # 初始化字典，键为层号，值为列表
        feats_dict = {l: [] for l in self.desc_layers}
        ids, names = [], []
        
        for batch in tqdm(dataloader, desc='DINOv3 提取特征'):
            x = batch['x']  # [B, 3, H, W] 或 [B, 1, 3, H, W]
            # 兼容 U1652_Image_S2D 中的额外维度
            if hasattr(x, 'ndim') and x.ndim == 5 and x.shape[1] == 1:
                x = x.squeeze(1)
            y = batch['y']  # [B]
            name = batch['name']  # List[str]
            
            # 提取批量特征，返回 {layer: tensor}
            batch_results = self._extract_batch_feats(x)
            
            for l, v in batch_results.items():
                feats_dict[l].append(v.cpu())
                
            ids.append(y)
            names.extend(name)

        # 拼接所有批次
        final_feats = {}
        for l, v_list in feats_dict.items():
            if len(v_list) > 0:
                final_feats[l] = torch.cat(v_list, dim=0).to(torch.float32)
            else:
                final_feats[l] = torch.empty(0)
                
        all_ids = torch.cat(ids, dim=0)
        return final_feats, all_ids, names

    @staticmethod
    def save_view(savedir: str, view: str, feats: torch.Tensor, ids: torch.Tensor, names: List[str]):
        """
        按 EM-CVGL 的格式保存某一视角（sat/dro）的特征与元信息。
        - savedir: 目录
        - view: 'sat' 或 'dro'（也支持 'sat_160k' 与其他兼容名）
        - feats: [N, D]
        - ids: [N]
        - names: List[str]
        """
        _mkdir_if_missing(savedir)
        torch.save(feats, osp.join(savedir, f'{view}_feat'))
        torch.save(ids, osp.join(savedir, f'{view}_id'))
        torch.save(names, osp.join(savedir, f'{view}_name'))
        print(f'[保存完成] {view} 视角特征保存在: {savedir}')