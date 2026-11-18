"""Numpy inference engine for Underdog Woof (speech model, one-step flow head and audio decoder). Imports numpy and
the standard library only."""
import json, math, time, wave
import numpy as np

SR = 24000
EPS = float(np.finfo(np.float32).eps)
f32 = np.float32

CHARS = list(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:'\"-()&/%$€£@#+=*_<>[]") + ["…", "—"]
VOCAB = {c: i + 3 for i, c in enumerate(CHARS)}


def encode_text(s):
    return [1] + [VOCAB.get(c, 2) for c in s]


# ------------------------------------------------------------------------------------------------ torch-compatible RNG
class TorchRandn:
    """torch.manual_seed(seed); torch.randn(n) (float32, CPU, n a multiple of 16) without torch: torch draws n 24-bit
    uniforms from mt19937, then Box-Muller per 16: out[j] = r cos, out[j+8] = r sin with u1 = 1 - u[j], u2 = u[j+8]."""

    def __init__(self, seed=0):
        self.seed(seed)

    def seed(self, seed):
        self.bg = np.random.MT19937(0)
        self.bg._legacy_seeding(seed)  # init_genrand(seed), as torch's mt19937

    def randn(self, n):
        u = (self.bg.random_raw(n) & 0xFFFFFF).astype(np.float64) * 2.0 ** -24
        u = u.reshape(-1, 2, 8)
        r = np.sqrt(-2.0 * np.log(1.0 - u[:, 0]))
        th = 2.0 * np.pi * u[:, 1]
        return np.stack([r * np.cos(th), r * np.sin(th)], 1).reshape(-1).astype(f32)


RNG = TorchRandn(0)


# ------------------------------------------------------------------------------------------------ weights
def unflatten(npz):
    """npz written by export(): '__tree__' holds the nested structure, arrays are referenced as {"@": key}."""
    def build(node):
        if isinstance(node, dict):
            if "@" in node and len(node) == 1:
                return npz[node["@"]]
            return {k: build(v) for k, v in node.items()}
        if isinstance(node, list):
            return [build(v) for v in node]
        return node
    return build(json.loads(str(npz["__tree__"])))


def lin_w(e):
    """Speech-model linear entry -> fp32 (out, in) weight (int8 * per-row scale) and fp32 bias or None."""
    w = e["q"].astype(f32) * e["s"].astype(f32)[:, None] if "q" in e else e["w"].astype(f32)
    b = e.get("b")
    return np.ascontiguousarray(w), (None if b is None else b.astype(f32))


def deq(t):
    """Decoder tensor: fp16/fp32 array or grouped-int8 dict {"q8", "s", "shape", "axis", "g"} -> fp32."""
    if not isinstance(t, dict):
        return t.astype(f32)
    q, s, shape, axis, g = t["q8"], t["s"].astype(f32), tuple(t["shape"]), t["axis"], t["g"]
    n, ng = q.shape[1], s.shape[1]
    m = np.zeros((q.shape[0], ng * g), f32)
    m[:, :n] = q
    m = (m.reshape(q.shape[0], ng, g) * s[..., None]).reshape(q.shape[0], -1)[:, :n]
    rest = [shape[d] for d in range(len(shape)) if d != axis]
    return np.ascontiguousarray(np.moveaxis(m.reshape(shape[axis], *rest), 0, axis))


# ------------------------------------------------------------------------------------------------ speech model
def _layer_norm(h):
    mu = h.mean(-1, keepdims=True)
    c = h - mu
    return c / np.sqrt((c * c).mean(-1, keepdims=True) + f32(1e-5))


def _silu(x):
    return x / (f32(1) + np.exp(-x))


class Speech:
    def __init__(self, st, max_len=1100):
        c = st["cfg"]
        self.L, self.d, self.H, self.ff, self.hw, self.latent = c["L"], c["d"], c["H"], c["ff"], c["hw"], c["latent"]
        self.Dh = self.d // self.H
        self.max_len = max_len
        self.layers = [[lin_w(e) for e in l] for l in st["layers"]]           # (W, None) x 4 per layer
        self.ce, self.ada = lin_w(st["ce"]), lin_w(st["ada"])
        self.hblocks = [[lin_w(e) for e in fs] for fs in st["hblocks"]]
        self.lat_in, self.h_inp, self.h_out = lin_w(st["lat_in"]), lin_w(st["h_inp"]), lin_w(st["h_out"])
        self.text_emb, self.bos = st["text_emb"].astype(f32), st["bos"].astype(f32)
        self.depth = c["depth"]
        Dh = self.Dh
        inv = (1.0 / (c["rope_base"] ** (np.arange(0, Dh, 2, dtype=f32) / f32(Dh)))).astype(f32)
        ang = np.arange(max_len, dtype=f32)[:, None] * inv
        cs, sn = np.cos(ang), np.sin(ang)
        self.cos = np.concatenate([cs, cs], -1).astype(f32)                     # (P, Dh)
        self.sin = np.concatenate([-sn, sn], -1).astype(f32)
        self.kv = np.zeros((self.L, 2, self.H, max_len, Dh), f32)              # pages are touched only as filled
        self.scale = f32(Dh ** -0.5)

    @staticmethod
    def _rinv(x):
        return f32(1.0) / np.sqrt(np.dot(x, x) / f32(x.shape[-1]) + f32(EPS))

    def _prefix_block(self, i, x, n):
        (wqkv, _), (wo, _), (w1, _), (w2, _) = self.layers[i]
        H, Dh, d, h2 = self.H, self.Dh, self.d, self.Dh // 2
        xn = x / np.sqrt((x * x).mean(-1, keepdims=True) + f32(EPS))
        qkv = (xn @ wqkv.T).reshape(n, 3, H, Dh)
        qk = qkv[:, :2]
        qk = qk * self.cos[:n, None, None] + np.concatenate([qk[..., h2:], qk[..., :h2]], -1) * self.sin[:n, None, None]
        K, V = self.kv[i, 0], self.kv[i, 1]
        K[:, :n] = qk[:, 1].transpose(1, 0, 2)
        V[:, :n] = qkv[:, 2].transpose(1, 0, 2)
        q = qk[:, 0].transpose(1, 0, 2)                                          # (H, n, Dh)
        s = np.matmul(q, K[:, :n].transpose(0, 2, 1)) * self.scale              # (H, n, n)
        s[:, np.triu(np.ones((n, n), bool), 1)] = -np.inf
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        a = np.matmul(s, V[:, :n]).transpose(1, 0, 2).reshape(n, d)
        x = x + a @ wo.T
        xn = x / np.sqrt((x * x).mean(-1, keepdims=True) + f32(EPS))
        gu = xn @ w1.T
        return x + (_silu(gu[:, :self.ff]) * gu[:, self.ff:]) @ w2.T

    def _block1(self, i, x, p):
        (wqkv, _), (wo, _), (w1, _), (w2, _) = self.layers[i]
        H, Dh = self.H, self.Dh
        qkv = (wqkv @ x) * self._rinv(x)
        qkv = qkv.reshape(3, H, Dh)
        qk = qkv[:2]
        qk = qk * self.cos[p] + qk.reshape(2, H, 2, Dh // 2)[:, :, ::-1].reshape(2, H, Dh) * self.sin[p]
        K, V = self.kv[i, 0], self.kv[i, 1]
        K[:, p] = qk[1]
        V[:, p] = qkv[2]
        s = np.matmul(K[:, :p + 1], qk[0][:, :, None])[:, :, 0] * self.scale     # (H, p+1)
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        a = np.matmul(s[:, None, :], V[:, :p + 1])[:, 0]                          # (H, Dh)
        x = x + wo @ a.reshape(-1)
        gu = (w1 @ x) * self._rinv(x)
        return x + w2 @ (_silu(gu[:self.ff]) * gu[self.ff:])

    def _cond(self, x):
        w, b = self.ce
        ce = (w @ x) * self._rinv(x) + b
        return _silu(ce[:self.hw]), float(ce[self.hw])

    def _head(self, s, z):
        hw = self.hw
        A = self.ada[0] @ s + self.ada[1]
        h = self.h_inp[0] @ z + self.h_inp[1]
        for j, ((w1, b1), (w2, b2)) in enumerate(self.hblocks):
            o = 3 * hw * j
            y = A[o:o + hw] + _layer_norm(h) * A[o + hw:o + 2 * hw]
            h = h + A[o + 2 * hw:o + 3 * hw] * (w2 @ _silu(w1 @ y + b1) + b2)
        o = 3 * hw * self.depth
        return self.h_out[0] @ (A[o:o + hw] + _layer_norm(h) * A[o + hw:o + 2 * hw]) + self.h_out[1]

    def generate(self, voice, text_ids, max_frames=750, temp=0.5, eos_thresh=0.9, min_frames_per_char=0.0, force=None):
        """As FastSpeech.generate: one randn(32) per frame, drawn after the EOS check. Returns (T, latent)."""
        if force is not None:
            max_frames, eos_thresh = len(force), 1.0 - 1e-12
        x = np.concatenate([voice, self.text_emb[np.asarray(text_ids)], self.bos[None]], 0).astype(f32)
        p0 = x.shape[0]
        max_frames = min(max_frames, self.max_len - p0)
        min_frames = int(min_frames_per_char * len(text_ids))
        for i in range(self.L):
            x = self._prefix_block(i, x, p0)
        x = np.ascontiguousarray(x[-1])
        s, e = self._cond(x)
        eos_t = math.log(eos_thresh / (1 - eos_thresh))
        out = np.empty((max_frames, self.latent), f32)
        n = 0
        win, wb = self.lat_in
        for f in range(max_frames):
            if f > max(3, min_frames) and e > eos_t:
                break
            noise = RNG.randn(self.latent) * f32(temp)
            z = noise + self._head(s, noise)
            out[f] = z
            n = f + 1
            x = win @ (z if force is None else force[f]) + wb
            for i in range(self.L):
                x = self._block1(i, x, p0 + f)
            s, e = self._cond(x)
        return out[:n].copy()


# ------------------------------------------------------------------------------------------------ decoder
def _erf(x):
    """Abramowitz-Stegun 7.1.26, |error| <= 1.5e-7 (numpy has no erf)."""
    a = np.abs(x)
    t = f32(1) / (f32(1) + f32(0.3275911) * a)
    y = f32(1) - (((((f32(1.061405429) * t - f32(1.453152027)) * t) + f32(1.421413741)) * t - f32(0.284496736)) * t
                  + f32(0.254829592)) * t * np.exp(-a * a)
    return np.copysign(y, x)


def _gelu(x):
    return f32(0.5) * x * (f32(1) + _erf(x * f32(1 / math.sqrt(2))))


def _elu_(x):
    neg = np.minimum(x, f32(0))
    np.expm1(neg, out=neg)
    np.maximum(x, f32(0), out=x)
    x += neg
    return x


class _Conv:
    def __init__(self, st, lazy):
        self.st, self.lazy = st, lazy
        self.k, self.cin = st["w"]["shape"][2] if isinstance(st["w"], dict) else st["w"].shape[2], \
            st["w"]["shape"][1] if isinstance(st["w"], dict) else st["w"].shape[1]
        self.b = st["b"].astype(f32)
        self.w = None if lazy else self._weights()

    def _weights(self):
        w = deq(self.st["w"])
        return [np.ascontiguousarray(w[:, :, j].T) for j in range(self.k)]       # (Cin, Cout) per tap

    def reset(self):
        self.state = np.zeros((self.k - 1, self.cin), f32)

    def __call__(self, x, res=None):
        w = self.w if self.w is not None else self._weights()
        T = x.shape[0]
        if self.k > 1:
            x = np.concatenate([self.state, x])
            self.state = x[T:].copy()
        out = x[:T] @ w[0]
        out += self.b if res is None else res + self.b
        for j in range(1, self.k):
            out += x[j:j + T] @ w[j]
        return out


class _ConvT:
    def __init__(self, st, lazy):
        self.st, self.lazy, self.s = st, lazy, st["stride"]
        shape = st["w"]["shape"] if isinstance(st["w"], dict) else st["w"].shape
        cin, self.cout, k = shape
        self.sc = self.s * self.cout
        self.b = np.tile(st["b"].astype(f32), self.s)
        self.w = None if lazy else self._weights()

    def _weights(self):
        w = deq(self.st["w"])
        cin, cout, k = w.shape
        return np.ascontiguousarray(w.transpose(0, 2, 1).reshape(cin, k * cout))

    def reset(self):
        self.carry = np.zeros(self.sc, f32)

    def __call__(self, x):
        w = self.w if self.w is not None else self._weights()
        T = x.shape[0]
        Z = x @ w
        out = Z[:, :self.sc] + self.b
        out[0] += self.carry
        out[1:] += Z[:-1, self.sc:]
        self.carry = Z[-1, self.sc:].copy()
        return out.reshape(T * self.s, self.cout)


class Decoder:
    """mode fp32: all weights dequantized at load. mode int8: the whole-sentence stages (transformer, SEANet up to
    1200 Hz) keep int8 weights and dequantize them once per sentence; the small high-rate stages stay fp32."""

    def __init__(self, st, chunk=8, split=6, mode="fp32"):
        c = st["cfg"]
        self.nh, self.hd, self.window, self.eps = c["heads"], c["head_dim"], c["window"], f32(c["eps"])
        self.chunk, self.split, self.lazy = chunk, split, mode == "int8"
        self.inp_w, self.inp_b = deq(st["inp"][0]).T.copy(), deq(st["inp"][1])
        self.inv_freq = st["inv_freq"].astype(f32)
        self.tf_st = st["tf"]
        self.tf = None if self.lazy else [self._tf_layer(l) for l in st["tf"]]
        self.D = self.nh * self.hd
        self.ops = []
        for idx, (kind, p) in enumerate(st["seanet"]):
            lazy = self.lazy and idx < split
            self.ops.append((kind, _Conv(p, lazy) if kind == "conv" else _ConvT(p, lazy) if kind == "convt" else
                             (_Conv(p[0], lazy), _Conv(p[1], lazy)) if kind == "res" else None))

    @staticmethod
    def _tf_layer(l):
        return {"ln1": [deq(t) for t in l["ln1"]], "ln2": [deq(t) for t in l["ln2"]],
                **{k: np.ascontiguousarray(deq(l[k]).T) for k in ("qkv", "o", "fc1", "fc2")}}

    def _ln(self, h, wb):
        mu = h.mean(-1, keepdims=True)
        c = h - mu
        return c / np.sqrt((c * c).mean(-1, keepdims=True) + self.eps) * wb[0] + wb[1]

    def transformer(self, h):
        T, D, nh, hd = h.shape[0], self.D, self.nh, self.hd
        fr = np.arange(T, dtype=f32)[:, None] * self.inv_freq
        emb = np.concatenate([fr, fr], -1)
        cos, sin = np.cos(emb), np.sin(emb)
        i = np.arange(T)
        bad = (i[None] > i[:, None]) | (i[None] <= i[:, None] - self.window)
        for st in (self.tf if self.tf is not None else self.tf_st):
            l = st if self.tf is not None else self._tf_layer(st)
            y = self._ln(h, l["ln1"])
            qkv = (y @ l["qkv"]).reshape(T, 3, nh, hd).transpose(1, 2, 0, 3)     # (3, nh, T, hd)
            qk = qkv[:2]
            qk = qk * cos + np.concatenate([-qk[..., hd // 2:], qk[..., :hd // 2]], -1) * sin
            s = np.matmul(qk[0], qk[1].transpose(0, 2, 1)) * f32(1 / math.sqrt(hd))
            s[:, bad] = -np.inf
            s = np.exp(s - s.max(-1, keepdims=True))
            s /= s.sum(-1, keepdims=True)
            a = np.matmul(s, qkv[2]).transpose(1, 0, 2).reshape(T, D)
            h = h + a @ l["o"]
            y = self._ln(h, l["ln2"])
            h = h + _gelu(y @ l["fc1"]) @ l["fc2"]
        return h

    @staticmethod
    def _run(x, ops):
        for kind, op in ops:
            if kind == "elu":
                x = _elu_(x)
            elif kind == "res":
                x = op[1](_elu_(op[0](_elu_(x.copy()))), res=x)
            else:
                x = op(x)
        return x

    def __call__(self, lat):
        h = self.transformer(lat @ self.inp_w + self.inp_b)
        for kind, op in self.ops:
            if kind == "res":
                op[0].reset(); op[1].reset()
            elif op is not None:
                op.reset()
        x = self._run(h, self.ops[:self.split])
        rows = self.chunk * x.shape[0] // h.shape[0]
        return np.concatenate([self._run(np.ascontiguousarray(x[c:c + rows]), self.ops[self.split:])
                               for c in range(0, x.shape[0], rows)])[:, 0]


# ------------------------------------------------------------------------------------------------ full system
class WoofNP:
    def __init__(self, path, voice="juno", dec_mode="fp32"):
        with np.load(path) as z:
            st = unflatten(z)
        self.speech = Speech(st["speech"])
        self.dec = Decoder(st["decoder"], mode=dec_mode)
        self.mean, self.std = st["mean"].astype(f32), st["std"].astype(f32)
        self.voices = st["voices"]
        self.voice = self.voices[voice].astype(f32)
        self.frames_max, self.fps = st["max_frames"], st["fps"]
        del st
        self.t_speech = self.t_dec = 0.0

    def latents(self, text, temp=0.5, eos=0.9, min_fpc=0.0):
        return self.speech.generate(self.voice, encode_text(text), self.frames_max, temp, eos, min_fpc * self.fps / 12.5)

    def __call__(self, text, **kw):
        t0 = time.perf_counter()
        lat = self.latents(text, **kw)
        t1 = time.perf_counter()
        wav = self.dec(lat * self.std + self.mean)
        self.t_speech += t1 - t0
        self.t_dec += time.perf_counter() - t1
        return wav


def write_wav(path, x):
    pcm = (np.clip(x, -1, 1) * 32767).round().astype("<i2")  # same PCM_16 rounding as soundfile
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())


