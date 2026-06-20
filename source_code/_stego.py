"""Steganography-related functions.

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import torch
torch._dynamo.config.cache_size_limit = 16  # allowed versions per compiled function
torch._functorch.config.donated_buffer = False  # disable donated buffers
import torch.nn.functional as F
from types import SimpleNamespace
from typing import Tuple, Callable, Optional
import warnings


# ========== PROJECTION MATRICES ==========
def get_deltas_ternary(
    q: int,
    device = torch.device('cpu'),
    return_index: bool = False,
    one_hot: bool = False,
) -> torch.Tensor:
    """Generate all combinations of q ternary digits.

    :param q: number of dimensions
    :param device: target device
    :param return_index: return {0,1,2} instead of {0,+1,-1}
    :param one_hot: return one-hot encoded instead of index
    """
    base = 3
    # all numbers from 0 to 3^q-1
    deltas = torch.arange(0, base ** q, device=device)
    # convert each number to base-3 representation
    digits = (deltas[:, None] // (base ** torch.arange(q, device=device))) % base  # 3^q, q
    if one_hot:
        digits = torch.tensor([
            [
                value
                for digit in row
                for value in (int(digit == 0), int(digit == 1), int(digit == 2))
            ]
            for row in digits
        ]).to(device)
        return digits
    if return_index:
        return digits
    # map {0, 1, 2} → {0, 1, -1}
    mapping = torch.tensor([0, 1, -1], device=device)
    deltas = mapping[digits]
    return deltas


H27 = SimpleNamespace()  # precomputed (module-level persistent) projection matrices
def initialize(device: torch.device):
    """Precompute the projection matrices.

    :param device: target device
    """
    # projection matrix for 27 -> 3 (values)
    H27._3 = get_deltas_ternary(3, device=device, return_index=False).float()  # 27 3
    H27._3 = H27._3[None, :, :, None, None]  # 1 27 3 1 1
    assert H27._3.shape == (1, 27, 3, 1, 1)
    # projection matrix for 27 -> 9 (one-hot), ordering [0, +1, -1]
    H27._3_onehot = (get_deltas_ternary(3, one_hot=True, device=device) != 0).float()  # 27 9
    H27._3_onehot = H27._3_onehot[None, :, :, None, None]  # 1 27 9 1 1
    assert H27._3_onehot.shape == (1, 27, 9, 1, 1)


# ========== BETA CONVERSIONS ==========
@torch.compile
def p3_sep_axes(
    p3: torch.Tensor,
) -> torch.Tensor:
    """Separate axis of the asymmetric probability map, resulting in shape (B 3 2 H W).

    :param p3: asymmetric probability map, of shape (B 9 H W)
    """
    assert p3.shape[1] == 9
    B, C, H, W = p3.size()
    p3 = torch.stack([
        torch.stack([
            p3[:, 1],  # p_p1[R]
            p3[:, 2],  # p_m1[R]
        ], axis=1),
        torch.stack([
            p3[:, 4],  # p_p1[G]
            p3[:, 5],  # p_m1[G]
        ], axis=1),
        torch.stack([
            p3[:, 7],  # p_p1[B]
            p3[:, 8],  # p_m1[B]
        ], axis=1),
    ], axis=1)  # B 3 2 H W
    assert p3.shape == (B, 3, 2, H, W)
    return p3


@torch.compile
def p27_to_p3(
    p27: torch.Tensor,
    separate_axes: bool = False,
) -> torch.Tensor:
    """Convert 27-ary probability map to asymmetric probability, by applying per-channel additivity.
    The resulting shape is (B 9 H W) or (B 3 2 H W), depending on the separate_axis.

    :param p27: 27-ary probability map, of shape (B 27 H W)
    :param separate_axis: separate axis after projection
    """
    assert p27.size(1) == 27
    B, C, H, W = p27.shape
    # get change mask
    p27 = p27[:, :, None]  # B 27 1 H W
    p3 = (p27 * H27._3_onehot).sum(dim=1)  # B 9 H W
    assert p3.shape == (B, 9, H, W)
    #
    if separate_axes:
        p3 = p3_sep_axes(p3)  # B 3 2 H W
    return p3


@torch.compile
def p3_to_p2(
    p3: torch.Tensor,
) -> torch.Tensor:
    """Symmetrize the probability map.

    :param p3: asymmetric probability map
    """
    if p3.size(1) == 9:
        p3 = p3_sep_axes(p3)  # B 3 2 H W
    return torch.sum(p3, dim=2)  # B 3 H W


@torch.compile
def p27_to_p2(
    p27: torch.Tensor,
) -> torch.Tensor:
    """Project 27-ary probability map to binary probabilities.

    :param p27: 27-ary probability
    """
    assert p27.size(1) == 27
    B, C, H, W = p27.size()
    p3 = p27_to_p3(p27)  # B 9 H W
    assert p3.shape == (B, 9, H, W)
    p2 = p3_to_p2(p3)  # B 3 H W
    assert p2.shape == (B, 3, H, W)
    return p2


# ========== ADJUSTING ==========
@torch.compile
def adjust_rho27(
    rho27: torch.Tensor,
    x0: torch.Tensor,
    wet_cost: float = 10**10,
) -> torch.Tensor:
    """Adjust 27-ary cost w.r.t. cover saturation.

    :param rho27: 27-ary cost map
    :param x0: cover
    :param wet_cost: wet cost
    """
    # replace inf/nan/>wet
    rho27 = torch.where(
        torch.isfinite(rho27) & (rho27 < wet_cost),
        rho27,
        torch.tensor(wet_cost, device=rho27.device, dtype=rho27.dtype),
    )  # B 27 H W

    # boundaries (1/255 = .00392)
    at_max = (x0 >= .999).unsqueeze(1)  # B 1 3 H W
    at_min = (x0 <= .001).unsqueeze(1)  # B 1 3 H W
    # violation
    violation = (at_max & (H27._3 > 0)) | (at_min & (H27._3 < 0))  # B 27 3 H W
    mask27 = violation.any(dim=2)  # B 27 H W
    # masking
    rho27 = torch.where(mask27, torch.tensor(wet_cost, device=rho27.device), rho27)  # B 27 H W
    return rho27  # B 27 H W


# ========== ENTROPY CONSTRAINT ==========
@torch.compile
def qary_entropy(
    p: torch.Tensor,
    dim: Optional[int | Tuple[int]] = None,
) -> torch.Tensor:
    """Calculate q-ary entropy.

    :param p: q-ary probability map, of shape (B, ...)
    :param dim: dimension to aggregate over, by default from 1 onwards
    """
    assert p.size(1) in {2, 3, 6, 9, 27}, 'strange q-arity encountered'
    #
    eps = torch.finfo(p.dtype).eps
    p = torch.clamp(p, eps, 1-eps)  # B 27 H W
    #
    if dim is None:
        dim = tuple(range(1, p.ndim))
    h_hat = -torch.sum(p * torch.log2(p), dim=dim)
    return h_hat


@torch.compile
def calc_lambda_solene(
    rho: torch.Tensor,
    m: int | float,
    objective: Callable,
    xtol: float = 1e-10,
    max_iter: int = 100,
    min_value: float = 1e-30,
) -> Tuple[float, float]:
    """Lambda search via binary search, to be called per batch.

    :param rho: cost map
    :param m: message length
    :param objective: objective function
    :param xtol: numerical tolerance
    :param max_iter: maximum number of iterations
    :param min_value: minimal lambda
    """
    device = rho.device
    B = rho.shape[0]
    if isinstance(m, (int, float)):
        m = torch.full((B,), m, dtype=rho.dtype, device=device)
    # bracket search
    lbd_l = torch.full((B,), min_value, dtype=rho.dtype, device=device)
    lbd_r = torch.full((B,), min_value, dtype=rho.dtype, device=device)
    active_mask = torch.ones(B, dtype=torch.bool, device=device)
    for it in range(50): # 1e-30 to 1e16 in 10x steps is ~46 iterations
        _, v = objective(rho, lbd_r.view(-1, 1, 1, 1))  # current payload for active images
        still_too_high = (v >= m) & (lbd_r < 1e16)  # check which images remain active
        lbd_l = torch.where(still_too_high, lbd_r, lbd_l)  # shift lower bound
        lbd_r = torch.where(still_too_high, lbd_r * 10, lbd_r)  # multiply upper bound
        active_mask = still_too_high  # carry the active images
        if not active_mask.any():
            break
    # bisection
    for it in range(max_iter):  # log2((max-min)/xtol) steps needed (70 is enough for f32 and 1e-3)
        mid = (lbd_l + lbd_r) / 2  # middle
        _, v = objective(rho, mid.view(-1, 1, 1, 1))  # current payload
        too_high = v > m  # splitting rule
        lbd_l = torch.where(too_high, mid, lbd_l)  # lower half
        lbd_r = torch.where(~too_high, mid, lbd_r)  # upper half
        if (lbd_r - lbd_l).max() < xtol:  # early exit
            break
    #
    lbd = (lbd_l + lbd_r) / 2  # midpoint is the final lambda
    _, h_hat = objective(rho, lbd.view(-1, 1, 1, 1))  # final entropy
    return lbd, h_hat


def calc_lambda_dde(
    rhos: torch.Tensor,
    m: float,
    n: int,
    objective: Callable,
    *,
    alpha_max: float = 1.0,
    num_bisection_steps: Optional[int] = 20,
    max_expansion_steps: int = 10,  # small number of expansions
    init_l1: float = 0.0,  # guess for alpha=0.4
    init_l3: float = 1000.0,  # guess for alpha=0.4
    **kw,
) -> Tuple[float, float]:
    """DDE (Binghamton) solver for lambda.

    :param rhos: cost map
    :param m: message length
    :param n: cover size
    :param objective: objective function
    :param alpha_max: maximal embedding rate
    :param num_bisection_steps:
    :param max_expansion_steps:
    :param init_l1: initial lower lambda
    :param init_l3: initial upper lambda
    """
    if len(rhos) == 0:
        raise ValueError("rhos must contain at least one tensor to infer device/dtype")

    # target m
    m_t = m

    # initialize bounds
    l1 = torch.tensor(float(init_l1), device=rhos.device, dtype=rhos.dtype)
    l3 = torch.tensor(float(init_l3), device=rhos.device, dtype=rhos.dtype)
    # conservative initialization for m1 (lower bound) and m3
    m1 = torch.tensor(float(n * alpha_max), device=rhos.device, dtype=rhos.dtype)
    # initial m3
    _, m3 = objective(lbda=l3, rho=rhos, **kw)  # entropy at l3

    # expansion
    for _ in range(max_expansion_steps):
        mask = (m3 > m_t)
        l3_doubled = l3 * 2.0
        _, m3_doubled = objective(lbda=l3_doubled, rho=rhos, **kw)

        # update l3 and m3 according to mask
        l3 = torch.where(mask, l3_doubled, l3)
        m3 = torch.where(mask, m3_doubled, m3)

    # bisection
    lbda = (l1 + l3) / 2.0
    for _ in range(num_bisection_steps):
        lbda = (l1 + l3) / 2.0
        _, m2 = objective(lbda=lbda, rho=rhos, **kw)
        cond = (m2 < m_t)
        #
        l3 = torch.where(cond, lbda, l3)
        m3 = torch.where(cond, m2, m3)
        l1 = torch.where(cond, l1, lbda)
        m1 = torch.where(cond, m1, m2)

    # final lambda
    lbda = (l1 + l3) / 2.0
    if torch.abs(lbda - init_l3) < .01:
        warnings.warn("optimization might not have converged", RuntimeWarning)
    return lbda, m2


@torch.compile
def average_payload(
    rho: torch.Tensor = None,
    lbda: float = None,
) -> Tuple[torch.Tensor, float]:
    """Objective function matching the code-native entropy.

    :param rho: cost map
    :param lbda: inverse temperature
    """
    p = F.softmax(-lbda * rho, dim=1)  # get selection channel
    h_hat = qary_entropy(p)
    return p, h_hat


@torch.compile
def average_payload_27to3(
    rho: torch.Tensor = None,
    lbda: float = None,
) -> Tuple[torch.Tensor, float]:
    """Objective function matching the (projected) asymmetric entropy.

    :param rho: cost map
    :param lbda: inverse temperature
    """
    assert rho.size(1) == 27
    #
    p27, _ = average_payload(rho=rho, lbda=lbda)
    p3 = p27_to_p3(p27)  # B 9 ...
    h3_hat = qary_entropy(p3)
    return p27, h3_hat


class find_lambda(torch.autograd.Function):
    """Differentiable lambda-search."""
    xtol = 1e-10
    max_iter = 30
    min_value = 1e-30
    solver = 'solene'
    @classmethod
    def forward(cls, ctx, rho, m, n, objective):
        """Apply solver to find the lambda.

        :param rho: cost vector
        :param m: message length
        :param n: cover size
        :param objective: objective funciton
        """
        B = rho.size(0)
        # # get probabilities
        if cls.solver == 'solene':
            lbdas, _ = calc_lambda_solene(
                rho=rho,
                m=m,
                objective=objective,
                xtol=cls.xtol,
                max_iter=cls.max_iter,
                min_value=cls.min_value,
            )
            lbdas = lbdas.view(B, 1, 1, 1)
        elif cls.solver == 'dde':
            lbdas, _ = calc_lambda_dde(
                rhos=rho,
                m=m,
                n=n,
                objective=objective,
                max_expansion_steps=4,
                num_bisection_steps=18,
            )
        else:
            raise NotImplementedError(f'unknown solver {cls.solver}')
        with torch.enable_grad():
            r = rho.detach().requires_grad_(True)
            l = lbdas.detach().requires_grad_(True)

        ctx.save_for_backward(r, l)
        ctx.objective = objective
        return lbdas.detach().clone().requires_grad_(True)

    @staticmethod
    def backward(ctx, grad_output):
        """Implicit differentiation of the constraint."""
        # local computation graph
        with torch.enable_grad():
            rho, lbda = ctx.saved_tensors  # 1 9 H W
            _, h_hat = ctx.objective(rho, lbda)

            # required gradients
            grad_h_lbda = torch.autograd.grad(h_hat, lbda, torch.ones_like(h_hat), retain_graph=True)[0]
            grad_h_rho = torch.autograd.grad(h_hat, rho, torch.ones_like(h_hat), retain_graph=False)[0]

            # implicit function gradient
            g = -grad_h_rho / (grad_h_lbda - 1e-9)

        return grad_output * g, None, None, None


# ========== SIMULATION ==========
@torch.compile
def _simulate(
    p: torch.Tensor,
    rand: torch.Tensor,
) -> torch.Tensor:
    """Non-differentiable categorical simulation according to the distribution p.

    :param p: probability map, of shape (B 27 H W)
    :param rand: uniform randomness, of shape (B 1 H W)
    """
    B, C, H, W = p.size()
    p_cum = torch.cumsum(p, dim=1)
    #
    indices = torch.argmax((rand < p_cum).to(torch.int8), dim=1)  # B H W
    delta3_hard = F.one_hot(indices, num_classes=C).permute(0, 3, 1, 2).to(p.dtype)
    delta3_hard = torch.round(delta3_hard)
    delta3 = delta3_hard
    return delta3


@torch.compile
def _simulate_differentiable(
    p: torch.Tensor,
    rand: torch.Tensor,
    tau: float = 1.0,
) -> torch.Tensor:
    """Differentiable categorical simulation according to the distribution p.
    It uses the Gumbel-softmax reparametrization combined with straight-through estimation.

    :param p: probability map, of shape (B 27 H W)
    :param rand: uniform randomness, of shape (B 27 H W)
    :param tau: temperature
    """
    # gumbel noise
    eps = 1e-20
    gumbel_noise = -torch.log(-torch.log(rand + eps) + eps)
    # soft categorical (for backward)
    logits = torch.log(p + eps)  # avoid log(0)
    delta_soft = torch.softmax((logits + gumbel_noise) / tau, dim=1)  # B 27 H W
    # hard categorical (for forward)
    idx = delta_soft.argmax(dim=1, keepdim=True)  # B 1 H W
    delta_hard = torch.zeros_like(p).scatter_(1, idx, 1.0)
    # straight-through combining soft and hard
    delta3 = (delta_hard - delta_soft).detach() + delta_soft  # B 27 H W
    return delta3


def generate_randomness(
    shape: Tuple[int],
    stego_seed: int,
    device: torch.device,
) -> torch.Generator:
    """Generates randomness.

    :param shape: randomness shape
    :param stego_seed: randomness seed
    :param device: target device
    """
    # reproducibility
    rng = torch.Generator(device=device)
    if stego_seed is not None:
        rng.manual_seed(stego_seed)
    # randomness
    rand = torch.rand(shape, generator=rng, device=device)
    return rand


def simulate(
    p: torch.Tensor,  # B C H W, per-pixel categorical probabilities
    deltas: torch.Tensor = None,  # q^3 q, delta patterns corresponding to each category
    tau: float = 1.0,  # temperature for Gumbel-Softmax (1.0 is default)
    simulator_method: str = 'hard',  # hard gumbel-softmax
    stego_seed: int = None,
) -> torch.Tensor:
    """Simulator of categorical distribution.

    :param p: probability map, of shape (B 27 H W)
    :param deltas: matrix converting categories (0-26) to the output symbols
    :param tau: temperature
    :param simulator_method: simulator method, hard (non-differentiable) or gumbel-softmax (differentiable)
    :param stego_seed: steganographic seed
    """
    device = p.device
    B, C, H, W = p.shape
    if C == 27 and deltas is None:
        deltas = H27._3
    # --------------------------------------
    # # simulate
    if simulator_method == 'hard':
        rand = generate_randomness((B, 1, H, W), stego_seed=stego_seed, device=device)
        delta = _simulate(
            p=p,
            rand=rand,
        )  # B C H W
    elif simulator_method == 'gumbel-softmax':
        rand = generate_randomness((B, C, H, W), stego_seed=stego_seed, device=device)
        delta = _simulate_differentiable(
            p=p,
            rand=rand,
            tau=tau,
        )  # B C H W
    else:
        raise NotImplementedError(f'unknown simulator method {simulator_method}')
    # --------------------------------------
    # Map categorical index to delta pattern
    assert delta.shape == (B, C, H, W)
    if deltas is not None:
        delta3 = torch.einsum('bchw, ncqxy -> bqhw', delta, deltas)  # B 3 H W
    else:
        delta3 = delta
    return delta3, delta


# ========== GAN ==========
class StegoGAN27(torch.nn.Module):
    """27-ary steganographic GAN."""
    def __init__(
        self,
        net_a: torch.nn.Module,
        net_e: torch.nn.Module = None,
        pad: int = None,
    ):
        """Construct the GAN.

        :param net_a: Alice' model
        :param net_e: Eve's model
        :param pad: pad size for embedding
        """
        super().__init__()
        self.net_a = net_a
        self.net_e = net_e
        self.pad = pad

    def get_probabilities(
        self,
        x0: torch.Tensor,
        alpha: torch.Tensor,
        tau: float = 1.,
        mode: bool = 'a',
    ) -> SimpleNamespace:
        """Calculate the probability map.

        :param x0: cover
        :param alpha: embedding rate
        :param tau: temperature
        :param mode: mode (a for Alice, e for Eve)
        """
        context = torch.enable_grad() if mode == 'a' else torch.no_grad()
        with context:
            B, C, H, W = x0.size()
            assert C == 3
            N = C * H * W
            # pad
            if mode == 'a': x0.requires_grad_(True)  # capture gradients
            x0_ = F.pad(x0, (self.pad, self.pad, self.pad, self.pad), mode='reflect') if self.pad else x0
            # cost
            rho27 = self.net_a(x0_)
            # unpad
            rho27 = rho27[..., self.pad:H+self.pad, self.pad:W+self.pad] if self.pad else rho27
            rho27 = adjust_rho27(rho27, x0)  # adjust
            # selection channel
            objective = average_payload_27to3  # average_payload_27to3 average_payload
            lbda = find_lambda.apply(rho27, alpha * N, N, objective)
            p27, _ = objective(rho27, lbda)
            p3 = p27_to_p3(p27)  # B 9 H W
            p2 = p27_to_p2(p27)  # B 3 H W
            h27, h3 = qary_entropy(p27), qary_entropy(p3)
            alpha3_ch = qary_entropy(p3.view(B*C, 3, H, W)).view(B, C) / H / W  # B C
            beta2 = torch.mean(torch.sum(p2, dim=1, keepdim=True), dim=(1, 2, 3))  # B
            beta2_ch = torch.mean(p2, dim=(2, 3))  # B C
            # effective selection channel (accounting for temperature)
            p27e, _ = objective(rho27, lbda / tau)
            p3e = p27_to_p3(p27e)
            h27e, h3e = qary_entropy(p27e), qary_entropy(p3e)
            alpha3e_ch = qary_entropy(p3e.view(B*C, 3, H, W)).view(B, C) / H / W  # B C
        #
        if mode == 'a':
            x0.retain_grad()
            rho27.retain_grad()
            lbda.retain_grad()
            p27.retain_grad()
        #
        return SimpleNamespace(
            rho27=rho27,
            lbda=lbda,
            p27=p27, h27=h27, alpha27=h27 / N,
            p3=p3, h3=h3, alpha3=h3 / N, alpha3_ch=alpha3_ch,
            p2=p2, beta2=beta2, beta2_ch=beta2_ch,
            p27e=p27e, h27e=h27e, alpha27e=h27e / N,
            p3e=p3e, h3e=h3e, alpha3e=h3e / N, alpha3e_ch=alpha3e_ch,
            x0=x0)

    def simulate(
        self,
        x0: torch.Tensor,
        p27: torch.Tensor,
        tau: float = 1.,
        stego_seed: int = None,
        simulator_method: str = 'gumbel-softmax',
        mode: bool = 'a',
    ) -> SimpleNamespace:
        """Simulate steganography.

        :param x0: cover
        :param p27: probability map
        :param tau: temperature
        :param stego_seed: steganographic seed
        :param simulator_method: simulator method
        :param mode: mode (a for Alice, e for Eve)
        """
        context = torch.enable_grad() if mode == 'a' else torch.no_grad()
        with context:
            delta3, delta27 = simulate(
                p=p27,  # expand draws
                tau=tau,
                simulator_method=simulator_method,
                stego_seed=stego_seed,
            )
            delta3 = delta3.to(x0.dtype)
            delta27 = delta27.to(x0.dtype)
            x1 = x0 + delta3 / 255.
        #
        if mode == 'a':
            x1.retain_grad()
        #
        return SimpleNamespace(
            delta27=delta27,
            delta3=delta3,
            x1=x1,
            x0=x0)

    def discriminate(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        mode: bool = 'a'
    ) -> SimpleNamespace:
        """Apply Eve's detector.

        :param x0: cover
        :param x1: stego
        :param mode: mode (a for Alice, e for Eve)
        """
        if self.net_e is None:
            return SimpleNamespace(l0=None, l1=None, s0=None, s1=None)
        # D forward
        context = torch.enable_grad() if mode in {'a', 'e'} else torch.no_grad()
        with context:
            l0 = self.net_e(x0)
            l1 = self.net_e(x1)
            # activation
            s0, s1 = F.softmax(l0, dim=1), F.softmax(l1, dim=1)
        return SimpleNamespace(l0=l0, l1=l1, s0=s0, s1=s1)  # f0=f0, f1=f1,

    def forward(
        self,
        x0: torch.Tensor,
        alpha: torch.Tensor,
        simulator_method: str,
        stego_seed: int,
        no_eve: bool = False,
        tau: float = 1.,
        mode: bool = 'pass',  # a, e, pass
    ) -> SimpleNamespace:
        """Wrapper for forward.

        :param x0: cover
        :param alpha: embedding rate
        :param simulator_method: simulator method
        :param stego_seed: steganographic seed
        :param no_eve: skip Eve's detector
        :param tau: temperature
        :param mode: mode (a for Alice, e for Eve)
        """
        # 1. Decision
        res_p = self.get_probabilities(x0, alpha, tau=tau, mode=mode)
        # 2. Action
        res_s = self.simulate(
            x0, res_p.p27, tau=tau,
            stego_seed=stego_seed,
            simulator_method=simulator_method,
            mode=mode)
        # 3. Judgment
        if no_eve:
            res_d = SimpleNamespace(l0=None, l1=None, s0=None, s1=None)
        else:
            res_d = self.discriminate(x0, res_s.x1, mode=mode)
        # Merge all namespaces
        delattr(res_s, 'x0')
        return SimpleNamespace(**vars(res_p), **vars(res_s), **vars(res_d))
