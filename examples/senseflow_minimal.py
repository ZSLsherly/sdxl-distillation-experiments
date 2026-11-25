"""SenseFlow 最小教学示例。

这不是作者的官方实现，而是一份用于学习数据流和梯度流的简化代码。它保留了
SenseFlow 最重要的四部分：

1. DMD：利用 teacher direction 与 fake direction 的差异更新学生；
2. ISG：让一次学生大步拟合“教师前半步 + 学生后半步”的组合路径；
3. IDA：学生更新后，立刻把 fake model 的参数轻微拉向新学生；
4. VFM-GAN：冻结视觉特征骨干，只训练判别头，并按时间步缩放 GAN 信号。

真实实现使用 SD3.5/FLUX、VAE、CLIP/DINOv2、多卡训练和严格的
velocity-to-score 转换。本文件把这些重型组件换成小型 MLP，便于单步调试。

本文件的时间约定：
    t = 1：纯噪声端；
    t = 0：干净数据端；
    采样方向：1 -> 0。

建议阅读顺序：euler_step -> isg_loss -> dmd_surrogate_loss -> apply_ida
              -> ToyVFMDiscriminator -> main。
"""

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DIM = 32
BATCH = 16
COARSE_TIMESTEPS = [1.0, 0.75, 0.50, 0.25, 0.0]

# 本教学例统一使用以下形状：
#   B：batch size，本例为 16
#   D：样本/latent 特征维度，本例为 32
#   x、noise、velocity：[B, D]
#   t、start_t、end_t：[B]
#   扩展后的时间：[B, 1]，可广播到 [B, D]
# 真实图像模型中，[B, D] 往往会替换成 [B, C, H, W] 或 [B, N, C]。


def expand_time(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """把形状 [B] 的时间扩展成 [B, 1]，以便与向量样本广播运算。"""
    # t: [B]
    # x: [B, D]
    # return: [B, 1]
    # 若 x 是图像 [B,C,H,W]，这里会返回 [B,1,1,1]。
    return t.reshape(-1, *([1] * (x.ndim - 1)))


def flow_forward_noise(x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Rectified Flow 的直线路径：x_t = (1-t)x_0 + t*noise。"""
    # 输入：x0 [B,D]，noise [B,D]，t [B]
    t = expand_time(t, x0)
    # 变化：t [B] -> [B,1]
    # 广播：(1-t)[B,1] * x0[B,D] -> [B,D]
    # 输出：x_t [B,D]
    return (1.0 - t) * x0 + t * noise


class VelocityNet(nn.Module):
    """用小型 MLP 代替真实实验中的 teacher/student/fake DiT。"""

    def __init__(self, dim: int = DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: [B,D]，t: [B]
        # t[:,None]: [B] -> [B,1]
        # cat([x,t]): [B,D] + [B,1] -> [B,D+1]
        # MLP 输出 velocity: [B,D]
        return self.net(torch.cat([x, t[:, None]], dim=1))


def euler_step(
    model: nn.Module,
    x_start: torch.Tensor,
    start_t: torch.Tensor,
    end_t: torch.Tensor,
) -> torch.Tensor:
    """执行一次 Euler ODE 步：x_end=x_start+(end_t-start_t)*v(x,t)。

    模型只预测当前位置的速度场 v；到底走多远由 dt 决定。因为生成时从
    t=1 走向 t=0，所以 end_t-start_t 通常为负数。
    """
    # x_start: [B,D]
    # start_t/end_t: [B]
    velocity = model(x_start, start_t)
    # velocity: [B,D]
    dt = expand_time(end_t - start_t, x_start)
    # end_t-start_t: [B] -> dt [B,1]
    # dt[B,1] * velocity[B,D] -> [B,D]
    # x_end: [B,D]
    return x_start + dt * velocity


class ToyVFMDiscriminator(nn.Module):
    """VFM 判别器的教学替代：冻结特征骨干，只训练真假判别头。"""

    def __init__(self, dim: int = DIM):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(dim, 64), nn.SiLU())
        self.head = nn.Linear(64, 1)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 这里不能使用 torch.no_grad()。
        # 冻结 backbone 只表示不更新它的参数；更新 generator 时，判别器仍须把
        # 关于输入 x 的梯度穿过 backbone 传回 generator。
        # x: [B,D]
        features = self.backbone(x)       # [B,D] -> [B,64]
        logits = self.head(features)      # [B,64] -> [B,1]
        return logits.squeeze(1)          # [B,1] -> [B]


@torch.no_grad()
def apply_ida(fake_model: nn.Module, generator: nn.Module, decay: float = 0.99) -> None:
    """IDA，对应论文 Eq. 9：phi <- decay*phi + (1-decay)*theta。

    theta 是刚更新的 generator 参数，phi 是 fake model 参数。它与 EMA 形式
    相同，但目的不是制作推理用 EMA 模型，而是减少 fake model 对学生分布的
    跟踪滞后。IDA 只是快速软对齐，不能替代后面的 denoising training。
    """
    for fake_parameter, generator_parameter in zip(
        fake_model.parameters(), generator.parameters()
    ):
        fake_parameter.mul_(decay).add_(generator_parameter, alpha=1.0 - decay)


def dmd_surrogate_loss(
    generator_output: torch.Tensor,
    teacher: nn.Module,
    fake_model: nn.Module,
) -> torch.Tensor:
    """注入 DMD 的 teacher-score 与 fake-score 差异梯度。

    为保持示例简洁，这里直接使用 velocity difference 代表经过参数化转换后的
    score difference。真实 SD3.5/FLUX 实现必须根据噪声日程，把 velocity/noise
    prediction 正确转换成 score 或论文采用的等价梯度。
    """
    batch = generator_output.shape[0]
    t = torch.rand(batch, device=generator_output.device).clamp_(0.02, 0.98)
    eps = torch.randn_like(generator_output)
    # generator_output/eps: [B,D]；t: [B]
    x_t = flow_forward_noise(generator_output, eps, t)
    # x_t: [B,D]

    # 教师描述目标分布，fake model 描述学生当前的输出分布。
    # 两者都不在这一步更新，所以只需计算给 generator 使用的方向。
    with torch.no_grad():
        teacher_direction = teacher(x_t, t)
        fake_direction = fake_model(x_t, t)
        # teacher_direction/fake_direction: [B,D]
        dmd_gradient = fake_direction - teacher_direction
        # dmd_gradient: [B,D]

        # 做按样本归一化，避免 direction 数值过大。这对应实际 DMD 中常见的
        # 梯度尺度稳定措施，但不等同于某个具体仓库的完整 weighting 公式。
        scale = dmd_gradient.abs().mean(dim=1, keepdim=True).clamp_min(1e-5)
        # 对 D 维求均值：[B,D] -> scale [B,1]
        # [B,D] / [B,1] 通过广播仍得到 [B,D]
        dmd_gradient = dmd_gradient / scale

    # 这是 gradient injection 技巧。target 并不是真实监督图片；detach 后，
    # MSE 对 generator_output 产生的梯度就是预先计算好的 dmd_gradient。
    target = (generator_output - dmd_gradient).detach()
    # generator_output/target: [B,D]；loss 为标量 []
    return 0.5 * F.mse_loss(generator_output, target)


def isg_loss(
    generator: nn.Module,
    teacher: nn.Module,
    x_tau_i: torch.Tensor,
    tau_i: torch.Tensor,
    tau_prev: torch.Tensor,
) -> torch.Tensor:
    """ISG，对应 Eq. 12：直接学生大步拟合 teacher+student 的拆分路径。

    直接路径：student tau_i -> tau_{i-1}
    目标路径：teacher tau_i -> t_mid，再由 student t_mid -> tau_{i-1}

    目标路径整体 stop-gradient，只让直接路径收到梯度。这样，每个稀疏推理
    锚点可以吸收所在区间内部的教师信息，同时目标仍与当前学生能力相容。
    """
    # x_tau_i: [B,D]
    # tau_i/tau_prev: [B]
    # 直接路径：学生一次跨过完整区间。本分支接收梯度。
    x_direct = euler_step(generator, x_tau_i, tau_i, tau_prev)
    # x_direct: [B,D]

    # 在当前粗粒度区间中独立、均匀地采样中间时间点。
    u = torch.rand_like(tau_i)
    t_mid = tau_prev + u * (tau_i - tau_prev)
    # u/t_mid: [B]，因此同一 batch 内每个样本可拥有不同的中间时间点。

    # 目标路径：teacher tau_i -> t_mid；student t_mid -> tau_{i-1}。
    # 整条目标路径不建计算图，因此不会出现 direct 与 target 相互追逐。
    with torch.no_grad():
        x_mid = euler_step(teacher, x_tau_i, tau_i, t_mid)
        # x_mid: [B,D]
        x_target = euler_step(generator, x_mid, t_mid, tau_prev)
        # x_target: [B,D]，且整个张量不携带计算图。

    # [B,D] 与 [B,D] 做逐元素平方后，默认取全部 B*D 元素的均值 -> 标量 []。
    return F.mse_loss(x_direct, x_target)


def discriminator_loss(
    discriminator: nn.Module,
    real_x0: torch.Tensor,
    fake_x0: torch.Tensor,
) -> torch.Tensor:
    # real_x0/fake_x0: [B,D]
    real_logit = discriminator(real_x0)
    fake_logit = discriminator(fake_x0.detach())
    # real_logit/fake_logit: [B]；mean 后 loss 为标量 []。
    return F.relu(1.0 - real_logit).mean() + F.relu(1.0 + fake_logit).mean()


def fake_model_denoising_loss(fake_model: nn.Module, fake_x0: torch.Tensor) -> torch.Tensor:
    """用学生样本训练 fake model，使其描述学生当前的输出分布。

    IDA 只是在参数空间中快速拉近；本 denoising loss 才是 fake model 正式拟合
    学生分布的训练信号。这两者与 TTUR 配合使用，而非互相替代。
    """
    batch = fake_x0.shape[0]
    t = torch.rand(batch, device=fake_x0.device).clamp_min_(0.02)
    noise = torch.randn_like(fake_x0)
    # fake_x0/noise: [B,D]；t: [B]
    x_t = flow_forward_noise(fake_x0.detach(), noise, t)
    target_velocity = noise - fake_x0.detach()
    # x_t/target_velocity/fake_model(x_t,t): 均为 [B,D]
    return F.mse_loss(fake_model(x_t, t), target_velocity)


def sample_real(batch: int) -> torch.Tensor:
    """生成一个双峰玩具真实分布，用来代替真实图文数据集。"""
    mode = torch.randint(0, 2, (batch, 1), device=DEVICE) * 2.0 - 1.0
    # mode: [B,1]
    center = mode.expand(batch, DIM) * 1.5
    # center: [B,1] -> [B,D]
    # return: [B,D]
    return center + 0.25 * torch.randn(batch, DIM, device=DEVICE)


def main(steps: int = 5) -> None:
    torch.manual_seed(7)

    # 与论文一致：generator 和 fake model 都从预训练 teacher 初始化。
    teacher = VelocityNet().to(DEVICE).eval()
    generator = deepcopy(teacher).train()
    fake_model = deepcopy(teacher).train()
    discriminator = ToyVFMDiscriminator().to(DEVICE).train()

    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    opt_g = torch.optim.AdamW(generator.parameters(), lr=1e-4)
    opt_fake = torch.optim.AdamW(fake_model.parameters(), lr=1e-4)
    opt_d = torch.optim.AdamW(discriminator.head.parameters(), lr=1e-4)

    for step in range(steps):
        real_x0 = sample_real(BATCH)
        noise = torch.randn_like(real_x0)
        # real_x0/noise: [B,D]

        # 从固定的少步推理日程中随机选择一个区间 tau_i -> tau_{i-1}。
        segment = torch.randint(0, len(COARSE_TIMESTEPS) - 1, ()).item()
        tau_value = COARSE_TIMESTEPS[segment]
        prev_value = COARSE_TIMESTEPS[segment + 1]
        tau_i = torch.full((BATCH,), tau_value, device=DEVICE)
        tau_prev = torch.full((BATCH,), prev_value, device=DEVICE)
        # tau_i/tau_prev: [B]

        # 教学简化：这里只通过真实数据前向加噪构造 x_tau_i。
        # 论文 Algorithm 1 中有 50% 概率使用 backward simulation，使训练输入
        # 同时覆盖真实数据轨迹和学生实际推理时会遇到的轨迹。
        x_tau_i = flow_forward_noise(real_x0, noise, tau_i)
        # x_tau_i: [B,D]
        fake_x0 = euler_step(
            generator,
            x_tau_i,
            tau_i,
            torch.zeros_like(tau_i),
        )
        # fake_x0: [B,D]

        # ==================== 1. 更新 generator ====================
        # DMD：匹配 teacher 与当前 student 的输出分布。
        loss_dmd = dmd_surrogate_loss(fake_x0, teacher, fake_model)
        loss_isg = isg_loss(generator, teacher, x_tau_i, tau_i, tau_prev)

        # ISG：让一次大步吸收区间内部的教师去噪信息。

        # Eq. 15：高噪声时 alpha_t 小，减弱不可靠的图像 GAN 监督；接近干净端
        # 时 alpha_t 大，加强真实性、细节与语义监督。
        alpha_t = 1.0 - tau_i
        # alpha_t: [B]
        # discriminator(fake_x0): [B]
        # 两者逐样本相乘仍为 [B]，mean 后 loss_gan 为标量 []。
        loss_gan = -(alpha_t.square() * discriminator(fake_x0)).mean()
        # loss_dmd/loss_isg/loss_gan/loss_g 均为零维标量张量 []。
        loss_g = loss_dmd + 0.5 * loss_gan + 1.0 * loss_isg

        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()

        # generator 一更新就立刻做 IDA，先纠正 fake model 的分布跟踪滞后。
        apply_ida(fake_model, generator, decay=0.99)

        # ==================== 2. 更新 fake model ===================
        # denoising loss 正式拟合刚才 generator 产生的分布。
        loss_fake = fake_model_denoising_loss(fake_model, fake_x0)
        opt_fake.zero_grad()
        loss_fake.backward()
        opt_fake.step()

        # =================== 3. 更新 discriminator =================
        # fake_x0 必须 detach，避免训练判别器时误更新 generator。
        loss_d = discriminator_loss(discriminator, real_x0, fake_x0)
        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()

        print(
            f"step={step:02d} segment={tau_value:.2f}->{prev_value:.2f} "
            f"dmd={loss_dmd.item():.4f} isg={loss_isg.item():.4f} "
            f"g_gan={loss_gan.item():.4f} fake={loss_fake.item():.4f} "
            f"disc={loss_d.item():.4f}"
        )


if __name__ == "__main__":
    main()
