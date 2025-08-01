from .conv import Conv
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
USE_FLASH_ATTN = False


def b_l_hp2b_l_h_p(x, p: int):
    b, l, hp = x.shape
    h = hp // p
    return x.reshape(b, l, h, p)


def b_l_gn2b_l_g_n(x, g: int):
    b, l, gn = x.shape
    n = gn // g
    return x.reshape(b, l, g, n)


def b_l_h_p2b_l_hp(x):
    b, l, h, p = x.shape
    return x.reshape(b, l, h * p)


def b_n_hd2b_h_n_d(x, h: int):
    b, n, hd = x.shape
    d = hd // h
    return x.reshape(b, n, h, d).transpose(1, 2)


class DWSConvLSTM2d(nn.Module):
    def __init__(self,
                 dim: int = 256,
                 dws_conv: bool = True,
                 dws_conv_only_hidden: bool = True,
                 dws_conv_kernel_size: int = 3,
                 cell_update_dropout: float = 0.):
        super().__init__()

        self.dim = dim
        xh_dim = dim * 2
        gates_dim = dim * 4
        conv3x3_dws_dim = dim if dws_conv_only_hidden else xh_dim
        self.conv3x3_dws = nn.Conv2d(in_channels=conv3x3_dws_dim,
                                     out_channels=conv3x3_dws_dim,
                                     kernel_size=dws_conv_kernel_size,
                                     padding=dws_conv_kernel_size // 2,
                                     groups=conv3x3_dws_dim) if dws_conv else nn.Identity()
        self.conv1x1 = nn.Conv2d(in_channels=xh_dim,
                                 out_channels=gates_dim,
                                 kernel_size=1)
        self.conv_only_hidden = dws_conv_only_hidden
        self.cell_update_dropout = nn.Dropout(p=cell_update_dropout)

    def forward(self, x, hc_previous):
        if hc_previous is None:
            hidden = torch.zeros_like(x)
            cell = torch.zeros_like(x)
            hc_previous = (hidden, cell)

        h_t0, c_t0 = hc_previous

        if self.conv_only_hidden:
            h_t0 = self.conv3x3_dws(h_t0)

        xh = torch.cat((x, h_t0), dim=1)

        if not self.conv_only_hidden:
            xh = self.conv3x3_dws(xh)

        mix = self.conv1x1(xh)
        cell_input, gates = torch.tensor_split(mix, [self.dim], dim=1)
        gates = torch.sigmoid(gates)
        forget_gate, input_gate, output_gate = torch.tensor_split(gates, 3, dim=1)

        cell_input = self.cell_update_dropout(torch.tanh(cell_input))

        c_t1 = forget_gate * c_t0 + input_gate * cell_input
        h_t1 = output_gate * torch.tanh(c_t1)

        return h_t1, (h_t1, c_t1)


def segsum(x):
    T = x.size(-1)
    x = x[..., None].repeat(1, 1, 1, 1, T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def discrete(X, A, B, C, block_len: int, initial_states):
    X = X.reshape(X.shape[0], X.shape[1] // block_len, block_len, X.shape[2], X.shape[3], )
    B = B.reshape(B.shape[0], B.shape[1] // block_len, block_len, B.shape[2], B.shape[3], )
    C = C.reshape(C.shape[0], C.shape[1] // block_len, block_len, C.shape[2], C.shape[3], )
    A = A.reshape(A.shape[0], A.shape[1] // block_len, block_len, A.shape[2])
    A = A.permute(0, 3, 1, 2)
    A_cumsum = torch.cumsum(A, dim=-1)
    L = torch.exp(segsum(A))
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)
    decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))[0]
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1:]
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp', C, states, state_decay_out)
    Y = Y_diag + Y_off
    Y = Y.reshape(Y.shape[0], Y.shape[1] * Y.shape[2], Y.shape[3], Y.shape[4], )
    return Y, final_state


def chunk_scan(X, dt, A, B, C, chunk_size: int, state):
    Y, final_state = discrete(X * dt.unsqueeze(-1), A * dt, B, C, chunk_size, state)
    return Y, final_state


class StreamingMamba2(nn.Module):
    def __init__(
            self,
            d_model=64,
            expand=2,
            headdim=64,
            ngroups=1,
            dt_min=0.001,
            dt_max=0.1,
            bias=False,
            A_init_range=(1, 16),
    ):
        super().__init__()
        self.d_model = d_model
        self.expand = expand
        self.headdim = headdim
        self.ngroups = ngroups
        self.d_state = headdim
        self.d_inner = int(self.expand * self.d_model)
        self.nheads = self.d_inner // self.headdim
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, int(d_in_proj), bias=bias)

        dt = torch.exp(
            torch.rand(self.nheads) *
            (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state

        self.conv2d = nn.Conv2d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            groups=conv_dim,
            bias=True,
            kernel_size=3,
            padding=1,
        )
        self.act = nn.SiLU()

        A = torch.empty(self.nheads, dtype=torch.float32).uniform_(*A_init_range)
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        self.norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

    def forward(self, u, state):
        B_, C_, H_, W_ = u.shape
        self.chunk_size = H_ * W_
        if state is None:
            state = torch.zeros((B_, H_ * W_ // self.chunk_size, self.nheads, self.d_state, self.d_state)).to(u.device).to(u.dtype)

        u_ = u.view(B_, C_, H_ * W_).permute(0, 2, 1)
        zxbcdt = self.in_proj(u_)

        A = -torch.exp(self.A_log)

        z, xBC, dt = torch.split(zxbcdt, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1)
        dt = F.softplus(dt + self.dt_bias)
        xBC = xBC.view(B_, H_, W_, -1).permute(0, 3, 1, 2).contiguous()

        xBC = self.act(self.conv2d(xBC))

        xBC = xBC.permute(0, 2, 3, 1).view(B_, H_ * W_, -1).contiguous()
        x, B, C = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)

        y, state = chunk_scan(
            b_l_hp2b_l_h_p(x, p=self.headdim),
            dt,
            A,
            b_l_gn2b_l_g_n(B, g=self.ngroups),
            b_l_gn2b_l_g_n(C, g=self.ngroups),
            self.chunk_size,
            state=state,
        )

        y = b_l_h_p2b_l_hp(y)
        y = self.norm(y)
        y = y * z

        out = self.out_proj(y)
        out = out.permute(0, 2, 1).view(B_, C_, H_, W_)

        out = out + u
        return out, state


class DFL(nn.Module):
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, _, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Conv_Block(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class C3k2(Conv_Block):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class AAttn(nn.Module):
    def __init__(self, dim, num_heads, area=1):
        super().__init__()
        self.area = area
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads
        self.qk = Conv(dim, all_head_dim * 2, 1, act=False)
        self.v = Conv(dim, all_head_dim, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        self.pe = Conv(all_head_dim, dim, 5, 1, 2, g=dim, act=False)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qk = self.qk(x).flatten(2).transpose(1, 2)
        v = self.v(x)
        pp = self.pe(v)
        v = v.flatten(2).transpose(1, 2)
        if self.area > 1:
            qk = qk.reshape(B * self.area, N // self.area, C * 2)
            v = v.reshape(B * self.area, N // self.area, C)
            B, N, _ = qk.shape
        q, k = qk.split([C, C], dim=2)

        q = q.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
        k = k.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
        v = v.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
        attn = (q.transpose(-2, -1) @ k) * (self.head_dim ** -0.5)
        max_attn = attn.max(dim=-1, keepdim=True).values
        exp_attn = torch.exp(attn - max_attn)
        attn = exp_attn / exp_attn.sum(dim=-1, keepdim=True)
        x = (v @ attn.transpose(-2, -1))
        x = x.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return self.proj(x + pp)


class ABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        super().__init__()
        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class A2C2f(nn.Module):
    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True):
        super().__init__()
        c_ = int(c2 * e)
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."

        num_heads = c_ // 32
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)
        init_values = 0.01
        self.gamma = nn.Parameter(init_values * torch.ones((c2)), requires_grad=True) if a2 and residual else None

        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, num_heads, mlp_ratio, area) for _ in range(2))) if a2 else C3k(c_, c_, 2, shortcut, g) for _ in range(n)
        )

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        if self.gamma is not None:
            return x + self.gamma.view(1, -1, 1, 1) * self.cv2(torch.cat(y, 1))
        return self.cv2(torch.cat(y, 1))


class SMC(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.n = n
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m1 = StreamingMamba2(d_model=self.c)
        if n > 1:
            self.m2 = nn.ModuleList(
                Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n - 1)
            )

    def forward(self, x, state):
        y = list(self.cv1(x).chunk(2, 1))
        out, state = self.m1(y[-1], state)
        y.append(out)
        if self.n > 1:
            y.extend(m(y[-1]) for m in self.m2)
        return self.cv2(torch.cat(y, 1)), state

