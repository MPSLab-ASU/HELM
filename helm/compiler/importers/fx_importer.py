import logging
import operator
import re
import torch
from typing import Optional

from ..IR.graph import HelmGraph, HelmNode

logger = logging.getLogger(__name__)


# ── Semantic tag keyword sets ─────────────────────────────────────────────────
# Extend these sets to support new architecture naming conventions.

_ATTN_KEYWORDS: frozenset[str] = frozenset(
    {
        "attn",
        "attention",
        "self_attn",
        "cross_attn",
        "mha",
        "multi_head",
        "multihead",
        "qkv",
        "query_key_value",
    }
)

_MLP_KEYWORDS: frozenset[str] = frozenset(
    {
        "mlp",
        "ffn",
        "feed_forward",
        "feedforward",
        "down_proj",
        "up_proj",
        "gate_proj",
        "fc1",
        "fc2",
        "c_fc",
        "c_proj",
        "wi",
        "wo",  # T5 MLP naming
        "dense_h_to_4h",
        "dense_4h_to_h",  # Falcon / Megatron
    }
)

_NORM_KEYWORDS: frozenset[str] = frozenset(
    {
        "norm",
        "layernorm",
        "layer_norm",
        "rmsnorm",
        "rms_norm",
        "ln_f",
        "ln_1",
        "ln_2",  # GPT-2 / Falcon
        "post_attention_layernorm",
        "input_layernorm",
        "final_layer_norm",
    }
)

_EMBED_KEYWORDS: frozenset[str] = frozenset(
    {
        "embed",
        "embedding",
        "embed_tokens",
        "wte",
        "wpe",
        "word_embeddings",
        "token_embeddings",
        "shared",  # T5 shared embedding
    }
)

_HEAD_KEYWORDS: frozenset[str] = frozenset(
    {
        "lm_head",
        "output_head",
        "logits",
        "embed_out",  # Pythia / GPT-NeoX
        "cls",  # BERT-style
    }
)

_KV_KEYWORDS: frozenset[str] = frozenset(
    {
        "k_proj",
        "v_proj",
        "k_norm",
        "v_norm",
        "key",
        "value",
        "kv_proj",  # some fused KV projections
        "query_key_value",  # Falcon fused QKV (also tagged attn above)
    }
)


class FXImporter:
    """
    Enriches a HelmGraph by propagating information from the underlying
    torch.fx.GraphModule.

    Responsibilities (in order, via ``run()``):
    1. Assign ``module_path`` to every node (exact for call_module, inferred
       from neighbours for other ops).
    2. Assign ``layer_id`` by pattern-matching the module path.
    3. Assign semantic tags (attention, MLP, norm, embedding, output head, KV).
    4. Populate graph-level indexes (layer→node map, cross-layer / residual edges).

    ``run()`` is idempotent: repeated calls overwrite previous annotations rather
    than accumulating them.
    """

    # Layer-index extraction patterns.
    # Each pattern must capture the integer layer index in group 1.
    # Patterns are tried in order; first match wins.
    _LAYER_PATTERNS: list[re.Pattern] = [
        # Standard: model.layers.0, transformer.layers.0
        re.compile(r"\blayers\.(\d+)\b"),
        # GPT-2 / Falcon: transformer.h.0
        re.compile(r"\bh\.(\d+)\b"),
        # Mistral / some custom models: model.blocks.0
        re.compile(r"\bblocks\.(\d+)\b"),
        # T5 / BART encoder+decoder: encoder.block.0, decoder.block.0
        re.compile(r"\bblock\.(\d+)\b"),
        # BERT / RoBERTa: encoder.layer.0
        re.compile(r"\blayer\.(\d+)\b"),
        # GPT-NeoX / Pythia (gpt_neox.layers.0), Mamba (backbone.layers.0),
        # Gemma-3 / Cohere (model.layers.0) — all covered by the first pattern.
    ]

    _ADD_FUNC_TARGETS: frozenset = frozenset({operator.add, operator.iadd, torch.add})
    _ADD_METHOD_TARGETS: frozenset[str] = frozenset({"add", "add_"})

    def __init__(self, gm: torch.fx.GraphModule, helm_graph: HelmGraph) -> None:
        self.graph = helm_graph
        # Materialise once at construction; modules are not expected to change.
        self.modules: dict[str, torch.nn.Module] = dict(gm.named_modules())

    def run(self) -> None:
        """Annotate the HelmGraph in-place.  Safe to call multiple times."""
        self._assign_module_paths()
        self._assign_layer_ids()
        self._assign_semantic_tags()
        self._populate_graph_indexes()

    # ── Module path assignment ────────────────────────────────────────────────

    def _assign_module_paths(self) -> None:
        # Pass 1: assign exact paths for call_module nodes.
        # This must run before pass 2 so that neighbour-inference can read
        # already-set call_module paths from both in-edges and out-edges.
        for node in self.graph.nodes:
            if node.fx_node.op == "call_module":
                node.module_path = str(node.fx_node.target)

        # Pass 2: infer paths for all other ops from their neighbours.
        for node in self.graph.nodes:
            if node.fx_node.op != "call_module":
                node.module_path = self._infer_module_path_from_neighbors(node)

    def _infer_module_path_from_neighbors(self, node: HelmNode) -> str:
        """Infer module path from both producers (in-edges) and consumers (out-edges).

        Consulting only producers misses ``get_attr`` nodes (weight tensors) that
        have no in-edges but belong to a specific layer.  Including consumer paths
        gives them a non-empty module path.
        """
        paths: list[str] = []

        for e in node.in_edges:
            src = self.graph.get_node(e.src_id)
            if src.module_path:
                paths.append(src.module_path)

        for e in node.out_edges:
            dst = self.graph.get_node(e.dst_id)
            if dst.module_path:
                paths.append(dst.module_path)

        if not paths:
            return ""
        return self._longest_common_module_prefix(paths)

    def _longest_common_module_prefix(self, paths: list[str]) -> str:
        split_paths = [p.split(".") for p in paths if p]
        if not split_paths:
            return ""

        prefix = split_paths[0]
        for parts in split_paths[1:]:
            i = 0
            while i < min(len(prefix), len(parts)) and prefix[i] == parts[i]:
                i += 1
            prefix = prefix[:i]
            if not prefix:
                break

        return ".".join(prefix)

    # ── Layer ID assignment ───────────────────────────────────────────────────

    def _extract_layer_id_from_path(self, module_path: str) -> Optional[int]:
        """Return the integer layer index encoded in ``module_path``, or None."""
        if not module_path:
            return None
        for pattern in self._LAYER_PATTERNS:
            m = pattern.search(module_path)
            if m:
                return int(m.group(1))
        return None

    def _assign_layer_ids(self) -> None:
        for node in self.graph.nodes:
            node.layer_id = self._extract_layer_id_from_path(node.module_path)

    # ── Semantic tag assignment ───────────────────────────────────────────────

    def _text_of_node(self, node: HelmNode) -> str:
        return " ".join(
            [
                str(node.name or ""),
                str(node.target or ""),
                str(node.module_path or ""),
                str(node.fx_node_name or ""),
            ]
        ).lower()

    def _child_text_of_module(self, target: str) -> str:
        """Return a space-joined string of all child module names for ``target``.

        Used to propagate semantic tags to opaque leaf-module nodes (e.g. an
        entire DecoderLayer traced as a single call_module) whose internals are
        not individually visible in the FX graph.
        """
        submod = self.modules.get(target)
        if submod is None:
            return ""
        return " ".join(name for name, _ in submod.named_modules() if name).lower()

    def _assign_semantic_tags(self) -> None:
        """Tag each node.  Idempotent: tags are boolean flags set to True/False."""
        for node in self.graph.nodes:
            text = self._text_of_node(node)

            child_text = ""
            if node.op_type == "call_module":
                child_text = self._child_text_of_module(str(node.target))

            combined = text + " " + child_text

            # Reset all tags so repeated run() calls don't accumulate stale state.
            node.is_attention = False
            node.is_mlp = False
            node.is_norm = False
            node.is_embedding = False
            node.is_output_head = False
            node.is_kv_projection = False

            if any(k in combined for k in _ATTN_KEYWORDS):
                node.mark_attention()

            if any(k in combined for k in _MLP_KEYWORDS):
                node.mark_mlp()

            if any(k in combined for k in _NORM_KEYWORDS):
                node.mark_norm()

            if any(k in combined for k in _EMBED_KEYWORDS):
                node.mark_embedding()

            if any(k in combined for k in _HEAD_KEYWORDS):
                node.mark_output_head()

            if any(k in combined for k in _KV_KEYWORDS):
                node.mark_kv_projection()

    # ── Graph index population ────────────────────────────────────────────────

    def _populate_graph_indexes(self) -> None:
        """Build layer→node index and classify cross-layer / residual edges."""
        self.graph.layer_to_node_ids.clear()

        for node in self.graph.nodes:
            if node.layer_id is not None:
                self.graph.register_layer(node.layer_id, node.id)

        for edge in self.graph.edges:
            # Reset flags so repeated run() calls don't carry stale state
            # from a previous annotation pass.
            edge.crosses_layer_boundary = False
            edge.is_residual = False

            src = self.graph.get_node(edge.src_id)
            dst = self.graph.get_node(edge.dst_id)

            if src.layer_id is not None and dst.layer_id is not None:
                edge.crosses_layer_boundary = src.layer_id != dst.layer_id

            if edge.crosses_layer_boundary:
                fx_node = dst.fx_node
                is_add = (
                    fx_node.op == "call_function"
                    and fx_node.target in self._ADD_FUNC_TARGETS
                ) or (
                    fx_node.op == "call_method"
                    and fx_node.target in self._ADD_METHOD_TARGETS
                )
                if is_add:
                    edge.is_residual = True

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def print_summary(self) -> None:
        """Log a per-node annotation summary at INFO level."""
        logger.info("[FXImporter] Annotation Summary")
        for node in self.graph.nodes:
            logger.info(
                "%s: module=%s, layer=%s, attn=%s, mlp=%s, norm=%s",
                node.name,
                node.module_path,
                node.layer_id,
                node.is_attention,
                node.is_mlp,
                node.is_norm,
            )
