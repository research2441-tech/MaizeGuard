"""Frozen CLIP ViT-B/32 feature extraction (ONNX fp16 export, CPU).
Outputs global image embedding and dense value-path patch embeddings (MaskCLIP-style)."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np, onnx, onnxruntime as ort
from onnx import helper, TensorProto, numpy_helper
from PIL import Image
from tokenizers import Tokenizer

MD = os.environ.get("MG_MODELS", "./models")
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
VNAME = "/vision_model/encoder/layers.11/self_attn/v_proj/Add_output_0"

def _build_vision():
    p = os.path.join(MD, "clip_vis_dense.onnx")
    m = onnx.load(os.path.join(MD, "mg_vision_fp16.onnx"))
    ini = {i.name: numpy_helper.to_array(i).astype(np.float32) for i in m.graph.initializer}
    W = dict(Wo=ini["onnx::MatMul_1856"], bo=ini["vision_model.encoder.layers.11.self_attn.out_proj.bias"],
             g=ini["vision_model.post_layernorm.weight"], b=ini["vision_model.post_layernorm.bias"],
             P=ini["onnx::MatMul_1859"])
    if not os.path.exists(p):
        m.graph.output.append(helper.make_tensor_value_info(VNAME, TensorProto.FLOAT16, None))
        onnx.save(m, p)
    np.savez(os.path.join(MD, "dense_head.npz"), **W)
    return p, W

class CLIP:
    def __init__(self, threads=2):
        so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        so.intra_op_num_threads = threads
        p, self.W = _build_vision()
        self.vis = ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])
        self.txt = ort.InferenceSession(os.path.join(MD, "mg_text_fp16.onnx"), so, providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(os.path.join(MD, "mg_tokenizer.json"))

    def encode_text(self, texts):
        out = []
        for t in texts:
            ids = np.array([self.tok.encode(t).ids[:77]], np.int64)
            out.append(self.txt.run(None, {"input_ids": ids})[0][0])
        e = np.stack(out).astype(np.float32)
        return e / np.linalg.norm(e, axis=1, keepdims=True)

    def _dense(self, v):
        # value-path dense tokens: out_proj(v) -> post LN -> projection (no residual, MaskCLIP)
        W = self.W
        x = v.astype(np.float32) @ W["Wo"] + W["bo"]
        mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)
        x = (x - mu) / np.sqrt(var + 1e-5) * W["g"] + W["b"]
        return x @ W["P"]

    def encode_batch(self, arr):
        """arr: (B,224,224,3) float32 in [0,1] -> global (B,512), dense (B,49,512)"""
        x = ((arr - MEAN) / STD).transpose(0, 3, 1, 2).astype(np.float32)
        g, v = self.vis.run(None, {"pixel_values": x})
        d = self._dense(v[:, 1:, :])
        return g.astype(np.float32), d.astype(np.float32)

def load_rgb(path):
    im = Image.open(path); im = im.convert("RGB"); return im

def views(im):
    """global 224 center view + 2x2 tiles of a 448 center view"""
    def rc(im, s):
        w, h = im.size; r = s / min(w, h)
        im2 = im.resize((max(s, round(w * r)), max(s, round(h * r))), Image.BICUBIC)
        w, h = im2.size; l = (w - s) // 2; t = (h - s) // 2
        return im2.crop((l, t, l + s, t + s))
    g = np.asarray(rc(im, 224), np.float32) / 255.
    b = np.asarray(rc(im, 448), np.float32) / 255.
    tiles = [b[:224, :224], b[:224, 224:], b[224:, :224], b[224:, 224:]]
    return np.stack([g] + tiles)
