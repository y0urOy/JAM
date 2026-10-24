"""Standardized training-cost accounting for the ImageWAM (Ovis-U1) pair, Appendix B convention.

Forward = 2NT FLOPs over N active dense-transformer parameters and T tokens; gradient-bearing
blocks cost four forward-equivalents (activation checkpointing); target and VAE passes are
forward-only.  Parameter sums come from the Ovis-U1-3B checkpoint tensor shapes
(visual_generator.backbone.*) and the ActionDiTYak checkpoint; the VAE encoder count is an
operator-level torch FlopCounterMode measurement of the Ovis AutoencoderKL on one 224x448 frame;
the text-context length is the mean over the 40 LIBERO instructions through the Ovis prompt
template (3 prefix + 2 indicator + 276 visual + suffix tokens).
"""
D = 1536
qkv, proj, mlp = 3 * D * D, D * D, 2 * D * 4 * D
double_img = qkv + proj + qkv + proj + mlp        # joint attention, auxiliary image self-attention, MLP
double_txt = qkv + proj + mlp
single = 7 * D * D + 5 * D * D                    # linear1 (qkv + mlp in), linear2 (proj + mlp out)
txt_in = 4096 * D + 2 * (qkv + proj + mlp)        # input embedder and two refiner blocks
N_DOUBLE, N_SINGLE, TEACHER_LAYER = 6, 12, 11
img_tok = N_DOUBLE * double_img + N_SINGLE * single
txt_tok = txt_in + N_DOUBLE * double_txt + N_SINGLE * single
act_tok = img_tok                                 # ActionDiTYak mirrors the image path
t_img = N_DOUBLE * double_img + (TEACHER_LAYER - N_DOUBLE) * single
t_txt = txt_in + N_DOUBLE * double_txt + (TEACHER_LAYER - N_DOUBLE) * single
H = 768
head = D * H + 2 * (3 * H * H + H * H + 2 * H * 4 * H) + H * D   # projector trunk and one readout
COND, TARGET, TXT, ACT = 392, 392, 296, 16
VAE = 0.4143e12                                   # per frame, FlopCounterMode
G = 4 * 2
rows = {
    "generation":        dict(visual=G * (COND + TARGET) * img_tok + G * TXT * txt_tok, action=G * ACT * act_tok, head=0, target=0, vae=2 * VAE),
    "foresight, cached": dict(visual=G * COND * img_tok + G * TXT * txt_tok, action=G * ACT * act_tok, head=G * COND * head, target=0, vae=VAE),
    "foresight, online": dict(visual=G * COND * img_tok + G * TXT * txt_tok, action=G * ACT * act_tok, head=G * COND * head, target=2 * ((COND + TARGET) * t_img + TXT * t_txt), vae=2 * VAE),
}
print(f"active params per token (M): image/action {img_tok/1e6:.1f}, text {txt_tok/1e6:.1f}; teacher to block {TEACHER_LAYER}: {t_img/1e6:.1f}/{t_txt/1e6:.1f}; head {head/1e6:.1f}")
base = None
for name, r in rows.items():
    tot = sum(r.values()); base = base or tot
    print(f"{name:18s} " + " ".join(f"{k} {v/1e12:5.2f}" for k, v in r.items()) + f"  total {tot/1e12:.2f}  change {100*(tot/base-1):+.1f}%")
