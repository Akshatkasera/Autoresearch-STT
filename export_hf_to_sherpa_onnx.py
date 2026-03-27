import argparse
import json
import math
from pathlib import Path

import onnx
import torch
import torch.nn.functional as F
import whisper
from torch import Tensor, nn
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def add_meta_data(filename: Path, meta_data: dict[str, str | int | float]) -> None:
    model = onnx.load(str(filename))

    while len(model.metadata_props):
        model.metadata_props.pop()

    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    onnx.save(model, str(filename))


def build_token_table(processor: WhisperProcessor, output_path: Path) -> None:
    whisper_dir = Path(whisper.__file__).parent
    token_path = whisper_dir / "assets" / "multilingual.tiktoken"
    if not token_path.is_file():
        raise FileNotFoundError(f"Could not find Whisper tokenizer assets at {token_path}")

    with output_path.open("w", encoding="utf-8") as f:
        for line in token_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            token, rank = line.split()
            f.write(f"{token} {rank}\n")


class EncoderForSherpa(nn.Module):
    def __init__(self, model: WhisperForConditionalGeneration):
        super().__init__()
        self.encoder = model.model.encoder
        self.decoder_layers = model.model.decoder.layers

    def forward(self, mel: Tensor) -> tuple[Tensor, Tensor]:
        x = F.gelu(self.encoder.conv1(mel))
        x = F.gelu(self.encoder.conv2(x))
        x = x.permute(0, 2, 1)

        pos = self.encoder.embed_positions.weight[: x.shape[1]]
        x = (x + pos).to(x.dtype)

        for layer in self.encoder.layers:
            x = layer(x, attention_mask=None)[0]

        x = self.encoder.layer_norm(x)

        cross_k = []
        cross_v = []
        for layer in self.decoder_layers:
            cross_k.append(layer.encoder_attn.k_proj(x))
            cross_v.append(layer.encoder_attn.v_proj(x))

        return torch.stack(cross_k), torch.stack(cross_v)


class CrossAttention(nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.scale = attn.head_dim ** -0.5

    def forward(self, x: Tensor, k: Tensor, v: Tensor) -> Tensor:
        bsz, tgt_len, embed_dim = x.shape
        num_heads = self.attn.num_heads
        head_dim = self.attn.head_dim

        q = self.attn.q_proj(x) * self.scale
        q = q.view(bsz, tgt_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(bsz, -1, num_heads, head_dim).transpose(1, 2)
        v = v.view(bsz, -1, num_heads, head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1))
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v)
        out = out.transpose(1, 2).contiguous().view(bsz, tgt_len, embed_dim)
        return self.attn.out_proj(out)


class SelfAttentionWithCache(nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.scale = attn.head_dim ** -0.5

    def forward(
        self,
        x: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
        attn_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        bsz, tgt_len, embed_dim = x.shape
        num_heads = self.attn.num_heads
        head_dim = self.attn.head_dim

        q = self.attn.q_proj(x) * self.scale
        k = self.attn.k_proj(x)
        v = self.attn.v_proj(x)

        k_cache[:, -tgt_len:, :] = k
        v_cache[:, -tgt_len:, :] = v

        q = q.view(bsz, tgt_len, num_heads, head_dim).transpose(1, 2)
        k_states = k_cache.view(bsz, -1, num_heads, head_dim).transpose(1, 2)
        v_states = v_cache.view(bsz, -1, num_heads, head_dim).transpose(1, 2)

        scores = torch.matmul(q, k_states.transpose(-2, -1))
        scores = scores + attn_mask.to(dtype=scores.dtype, device=scores.device)
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v_states)
        out = out.transpose(1, 2).contiguous().view(bsz, tgt_len, embed_dim)
        return self.attn.out_proj(out), k_cache, v_cache


class DecoderLayerForSherpa(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.self_attn = SelfAttentionWithCache(layer.self_attn)
        self.cross_attn = CrossAttention(layer.encoder_attn)

    def forward(
        self,
        x: Tensor,
        self_k_cache: Tensor,
        self_v_cache: Tensor,
        cross_k: Tensor,
        cross_v: Tensor,
        attn_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        residual = x
        hidden = self.layer.self_attn_layer_norm(x)
        hidden, self_k_cache, self_v_cache = self.self_attn(
            hidden, self_k_cache, self_v_cache, attn_mask
        )
        x = residual + hidden

        residual = x
        hidden = self.layer.encoder_attn_layer_norm(x)
        hidden = self.cross_attn(hidden, cross_k, cross_v)
        x = residual + hidden

        residual = x
        hidden = self.layer.final_layer_norm(x)
        hidden = self.layer.activation_fn(self.layer.fc1(hidden))
        hidden = self.layer.fc2(hidden)
        x = residual + hidden
        return x, self_k_cache, self_v_cache


class DecoderForSherpa(nn.Module):
    def __init__(self, model: WhisperForConditionalGeneration):
        super().__init__()
        decoder = model.model.decoder
        self.embed_tokens = decoder.embed_tokens
        self.embed_positions = decoder.embed_positions
        self.layer_norm = decoder.layer_norm
        self.layers = nn.ModuleList([DecoderLayerForSherpa(layer) for layer in decoder.layers])
        self.proj_weight = model.proj_out.weight
        ctx = decoder.embed_positions.num_embeddings
        mask = torch.full((ctx, ctx), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(
        self,
        tokens: Tensor,
        n_layer_self_k_cache: Tensor,
        n_layer_self_v_cache: Tensor,
        n_layer_cross_k: Tensor,
        n_layer_cross_v: Tensor,
        offset: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        start = offset[0]
        seq_len = tokens.shape[1]

        x = self.embed_tokens(tokens) + self.embed_positions.weight[start : start + seq_len]
        x = x.to(n_layer_cross_k.dtype)

        mask = self.causal_mask[start : start + seq_len, : start + seq_len]
        mask = mask.unsqueeze(0).unsqueeze(0)

        updated_k = n_layer_self_k_cache.clone()
        updated_v = n_layer_self_v_cache.clone()

        for i, layer in enumerate(self.layers):
            self_k_cache = updated_k[i, :, : start + seq_len, :]
            self_v_cache = updated_v[i, :, : start + seq_len, :]
            x, self_k_cache, self_v_cache = layer(
                x,
                self_k_cache=self_k_cache,
                self_v_cache=self_v_cache,
                cross_k=n_layer_cross_k[i],
                cross_v=n_layer_cross_v[i],
                attn_mask=mask,
            )
            updated_k[i, :, : start + seq_len, :] = self_k_cache
            updated_v[i, :, : start + seq_len, :] = self_v_cache

        x = self.layer_norm(x)
        logits = torch.matmul(x, self.proj_weight.t()).float()
        return logits, updated_k, updated_v


def export_model(
    model_dir: Path,
    output_dir: Path,
    language: str,
    task: str,
    opset: int,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = WhisperProcessor.from_pretrained(str(model_dir))
    model = WhisperForConditionalGeneration.from_pretrained(
        str(model_dir),
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model = model.float().eval()

    encoder = EncoderForSherpa(model)
    decoder = DecoderForSherpa(model)

    config = model.config
    encoder_filename = output_dir / "encoder.onnx"
    decoder_filename = output_dir / "decoder.onnx"
    tokens_filename = output_dir / "tokens.txt"

    model_dtype = next(model.parameters()).dtype
    mel = torch.randn(
        1,
        config.num_mel_bins,
        config.max_source_positions * 2,
        dtype=model_dtype,
    )
    n_layer_cross_k, n_layer_cross_v = encoder(mel)

    torch.onnx.export(
        encoder,
        mel,
        str(encoder_filename),
        opset_version=opset,
        dynamo=False,
        input_names=["mel"],
        output_names=["n_layer_cross_k", "n_layer_cross_v"],
        dynamic_axes={
            "mel": {0: "n_audio", 2: "T"},
            "n_layer_cross_k": {1: "n_audio", 2: "T_out"},
            "n_layer_cross_v": {1: "n_audio", 2: "T_out"},
        },
    )

    whisper_tokenizer = whisper.tokenizer.get_tokenizer(
        multilingual=True,
        num_languages=99,
        language=language,
        task=task,
    )
    metadata = {
        "model_type": "whisper-small",
        "version": "1",
        "maintainer": "autoresearch-stt",
        "n_mels": config.num_mel_bins,
        "n_audio_ctx": config.max_source_positions,
        "n_audio_state": config.d_model,
        "n_audio_head": config.encoder_attention_heads,
        "n_audio_layer": config.encoder_layers,
        "n_vocab": config.vocab_size,
        "n_text_ctx": config.max_target_positions,
        "n_text_state": config.d_model,
        "n_text_head": config.decoder_attention_heads,
        "n_text_layer": config.decoder_layers,
        "sot_sequence": ",".join(map(str, whisper_tokenizer.sot_sequence)),
        "all_language_tokens": ",".join(map(str, whisper_tokenizer.all_language_tokens)),
        "all_language_codes": ",".join(whisper_tokenizer.all_language_codes),
        "sot": config.decoder_start_token_id,
        "sot_index": 0,
        "eot": config.eos_token_id,
        "blank_id": whisper_tokenizer.encode(" ")[0],
        "is_multilingual": 1,
        "no_speech": whisper_tokenizer.no_speech,
        "non_speech_tokens": ",".join(map(str, whisper_tokenizer.non_speech_tokens)),
        "transcribe": whisper_tokenizer.transcribe,
        "translate": whisper_tokenizer.translate,
        "sot_prev": whisper_tokenizer.sot_prev,
        "sot_lm": whisper_tokenizer.sot_lm,
        "no_timestamps": whisper_tokenizer.no_timestamps,
    }
    add_meta_data(encoder_filename, metadata)

    tokens = torch.tensor([[config.decoder_start_token_id] * 3], dtype=torch.int64)
    n_layer_self_k_cache = torch.zeros(
        config.decoder_layers,
        1,
        config.max_target_positions,
        config.d_model,
        dtype=model_dtype,
    )
    n_layer_self_v_cache = torch.zeros_like(n_layer_self_k_cache)
    offset = torch.tensor([0], dtype=torch.int64)

    torch.onnx.export(
        decoder,
        (
            tokens,
            n_layer_self_k_cache,
            n_layer_self_v_cache,
            n_layer_cross_k,
            n_layer_cross_v,
            offset,
        ),
        str(decoder_filename),
        opset_version=opset,
        dynamo=False,
        input_names=[
            "tokens",
            "in_n_layer_self_k_cache",
            "in_n_layer_self_v_cache",
            "n_layer_cross_k",
            "n_layer_cross_v",
            "offset",
        ],
        output_names=["logits", "out_n_layer_self_k_cache", "out_n_layer_self_v_cache"],
        dynamic_axes={
            "tokens": {0: "n_audio", 1: "n_tokens"},
            "in_n_layer_self_k_cache": {1: "n_audio"},
            "in_n_layer_self_v_cache": {1: "n_audio"},
            "n_layer_cross_k": {1: "n_audio", 2: "T"},
            "n_layer_cross_v": {1: "n_audio", 2: "T"},
            "offset": {},
            "logits": {0: "n_audio", 1: "n_tokens"},
            "out_n_layer_self_k_cache": {1: "n_audio"},
            "out_n_layer_self_v_cache": {1: "n_audio"},
        },
    )

    build_token_table(processor, tokens_filename)

    return {
        "encoder": str(encoder_filename),
        "decoder": str(decoder_filename),
        "tokens": str(tokens_filename),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a merged HF Whisper model to sherpa-compatible ONNX.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--language", default="en")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    result = export_model(
        model_dir=Path(args.model_dir).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        language=args.language,
        task=args.task,
        opset=args.opset,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
