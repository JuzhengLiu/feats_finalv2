import torch
from utils.utils import LOSS
from torch import nn
import torch.nn.functional as F


def filter_InfoNCE(sim_mat, sim_mat2, logit_scale, loss_fn, label1, label2):
    logits_per_image1 = logit_scale * sim_mat
    logits_per_image2 = logit_scale * sim_mat2
    loss = (loss_fn(logits_per_image1, label1) + loss_fn(logits_per_image2, label2)) / 2
    return loss


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(0), dim=1)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(1), dim=0)
    return Z + u.unsqueeze(1) + v.unsqueeze(0)


@LOSS.register
class CycleAELoss(nn.Module):
    def __init__(self, weight, args, recorder, device):
        super().__init__()
        self.loss_function = nn.CrossEntropyLoss(label_smoothing=args.train.label_smoothing)
        
        # JSD / KL 基础函数
        self.kl_div_fn = nn.KLDivLoss(reduction='batchmean')
        
        # === 优化点 1: 提升温度系数 ===
        # 原 0.07 过于尖锐，导致忽略次级特征。
        # 提升至 0.15 以保留更多 Top-K 上下文信息 (Dark Knowledge)。
        self.T_recon = 0.13
        # ============================

        self.device = device
        self.w = weight
        self.recorder = recorder
        self.dro_num = args.train.dro_num
        self.idt_w = args.train.idt_w 
        self.feat_w = args.train.feat_w
        self.thr = args.train.pseudo_thr
        self.mutual_match = args.train.mutual_match
        self.keep_neg = args.train.keep_neg
        self.distill_w = getattr(args.train, 'distill_w', 0.5)

        # Sinkhorn Config
        self.sk_eps_start = getattr(args.train, 'sinkhorn_eps_start', 0.2)
        self.sk_eps_end = getattr(args.train, 'sinkhorn_eps_end', 0.05)
        self.sk_iters = getattr(args.train, 'sinkhorn_iters', 3)
        self.total_epochs = args.train.epochs

    def get_current_epsilon(self, current_epoch):
        if self.total_epochs == 0: return self.sk_eps_start
        progress = min(1.0, current_epoch / self.total_epochs)
        return self.sk_eps_start + (self.sk_eps_end - self.sk_eps_start) * progress

    def compute_jsd_loss(self, pred, target):
        """
        计算 Jensen-Shannon Divergence (JSD) 损失
        JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), where M = 0.5*(P+Q)
        相比单纯的 KL，JSD 是对称的且有界的，能更稳定地对齐分布形状。
        """
        # 1. 温度缩放
        pred_logits = pred / self.T_recon
        target_logits = target / self.T_recon

        # 2. 计算概率分布 (Probabilities)
        p_prob = F.softmax(pred_logits, dim=1)
        # target 需要 detach，防止梯度传回 Teacher (虽不传回但在计算图中截断更安全)
        q_prob = F.softmax(target_logits, dim=1).detach()
        
        # 3. 计算混合分布 M
        m_prob = 0.5 * (p_prob + q_prob)
        
        # 4. 计算对数概率 (Log Probabilities) 用于 KL 输入
        p_log_prob = F.log_softmax(pred_logits, dim=1)
        # 注意: 虽然 KL 的 target 不需要 log，但为了计算 KL(Q||M)，我们需要 Q 的 log 或者直接用 M 作为 target
        # 这里利用 torch.nn.KLDivLoss(input, target) = sum(target * (log_target - input)) 的性质
        # 但直接调用 pytorch API 更方便，需注意 KLDivLoss 接收 (log_prob, prob)
        
        m_log_prob = torch.log(m_prob + 1e-8) # 数值稳定性

        # 5. JSD 计算
        # Term 1: KL(P || M)
        # input=p_log, target=m_prob
        loss_1 = self.kl_div_fn(p_log_prob, m_prob)
        
        # Term 2: KL(Q || M)
        # 这一项其实对于"优化 Pred"来说，梯度来源于 M 里的 Pred 成分
        # 为了简化计算且保持 pytorch 风格，我们通常只计算 KL(P || Q) 的变体，
        # 但标准的 JSD 需要两边。
        # input=m_log, target=q_prob (反向) -> 这种很难用标准 API 表达梯度。
        
        # === 简化版实战 JSD (Symmetric KL) ===
        # 为了 PyTorch 自动微分友好，我们实现: 0.5 * KL(P||Q) + 0.5 * KL(Q||P)
        # 这在效果上近似 JSD，且能强迫 P 和 Q互相覆盖
        
        q_log_prob = F.log_softmax(target_logits, dim=1)
        
        # Forward KL: Pred 拟合 Target (Zero-forcing, 锐利)
        loss_fwd = self.kl_div_fn(p_log_prob, q_prob)
        
        # Reverse KL: Target 拟合 Pred (Mass-covering, 覆盖)
        # 注意：这里我们优化 Pred 使得 Target 在 Pred 分布下概率高
        loss_rev = self.kl_div_fn(q_log_prob, p_prob) 
        
        # 综合损失
        return 0.5 * (loss_fwd + loss_rev)

    def forward(self, data, logit_scale, args=None):
        fake_AA = data['fake_AA']
        fake_BB = data['fake_BB']
        enc_b = data['enc_b']
        enc_a = data['enc_a']
        real_A = data['x_s']
        real_B = data['x_t'].squeeze()
        A_id = data['y_s']
        B_id = data['y_t']

        # 1. 局部相似度
        sim_mat = torch.einsum('md, nd-> mn', enc_a, enc_b)
        m, n = sim_mat.shape
        sim_mat_multi = sim_mat.reshape(m, n // self.dro_num, self.dro_num)
        sim_mat_mean = sim_mat_multi.mean(-1) 
        
        # 2. Sinkhorn
        C = 1.0 - sim_mat_mean
        device = sim_mat_mean.device
        mu = torch.ones(m, device=device) / m
        nu = torch.ones(n // self.dro_num, device=device) / (n // self.dro_num)
        
        current_epoch = getattr(args, 'current_epoch', 0)
        epsilon = self.get_current_epsilon(current_epoch)
        
        with torch.no_grad():
             P_log = log_sinkhorn_iterations(-C / epsilon, torch.log(mu), torch.log(nu), self.sk_iters)
             P = torch.exp(P_log)
             target_AB = P / P.sum(dim=1, keepdim=True)
             target_BA = (P / P.sum(dim=0, keepdim=True)).T

        # 3. Hard Loss
        _, idx_P_row = P.max(dim=1)
        _, idx_P_col = P.max(dim=0)
        sk_mutual_mask = (idx_P_col[idx_P_row] == torch.arange(m, device=device))
        sim_val = sim_mat_mean[torch.arange(m, device=device), idx_P_row]
        hard_mask = sk_mutual_mask & (sim_val > self.thr)
        
        if hard_mask.sum() > 0:
            valid_A_idx = torch.nonzero(hard_mask).squeeze()
            if valid_A_idx.ndim == 0: valid_A_idx = valid_A_idx.unsqueeze(0)
            valid_B_idx = idx_P_row[hard_mask]
            
            logit_scale_exp = logit_scale["t"].exp()
            l_hard_a = self.loss_function(logit_scale_exp * sim_mat_mean[valid_A_idx], valid_B_idx)
            l_hard_b = self.loss_function(logit_scale_exp * sim_mat_mean.T[valid_B_idx], valid_A_idx)
            loss_hard = (l_hard_a + l_hard_b) / 2
        else:
            loss_hard = torch.tensor(0.0, device=device, requires_grad=True)

        # 4. Soft Loss
        logit_scale_exp = logit_scale["t"].exp()
        log_probs_AB = F.log_softmax(logit_scale_exp * sim_mat_mean, dim=1)
        loss_soft_AB = self.kl_div_fn(log_probs_AB, target_AB)
        log_probs_BA = F.log_softmax(logit_scale_exp * sim_mat_mean.T, dim=1)
        loss_soft_BA = self.kl_div_fn(log_probs_BA, target_BA)
        loss_soft = (loss_soft_AB + loss_soft_BA) / 2

        # 5. 重构损失: 采用对称 KL (近似 JSD) + T=0.15
        loss_recon_A = self.idt_w * self.compute_jsd_loss(fake_AA, real_A)
        loss_recon_B = self.idt_w * self.compute_jsd_loss(fake_BB, real_B)

        # 统计真实准确率
        gt_mask = A_id.unsqueeze(1) == B_id[:n].unsqueeze(0)
        _, gt_idx = gt_mask.max(-1)
        real_acc = (idx_P_row == gt_idx.to(idx_P_row)).float().mean()

        self.recorder.update('Lrec_A', loss_recon_A.item(), args.train.batch_size, type='f')
        self.recorder.update('Lrec_B', loss_recon_B.item(), args.train.batch_size, type='f')
        self.recorder.update('L_hard', loss_hard.item(), args.train.batch_size, type='f')
        self.recorder.update('L_soft', loss_soft.item(), args.train.batch_size, type='f')
        self.recorder.update('real_acc', real_acc.item(), args.train.batch_size, type='%')
        self.recorder.update('sk_eps', epsilon, args.train.batch_size, type='f')

        final_loss = loss_hard + self.distill_w * loss_soft + loss_recon_A + loss_recon_B
        return final_loss