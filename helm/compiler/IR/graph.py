import logging
import torch
from typing import Any, Union

logger = logging.getLogger(__name__)

# Type alias for node arguments: HelmNode references or raw constants
HelmArg = Union["HelmNode", int, float, str, bool, None]


class HelmEdge:
    """
    Represents a directed edge between two HelmNodes.
    """

    def __init__(
        self,
        src_id: int,
        dst_id: int,
        tensor_shape: list[int] | None = None,
        tensor_bytes: int = 0,
        is_residual: bool = False,
        crosses_layer_boundary: bool = False,
    ) -> None:
        self.src_id = src_id
        self.dst_id = dst_id
        self.tensor_shape = list(tensor_shape) if tensor_shape is not None else []
        self.tensor_bytes = tensor_bytes
        self.is_residual = is_residual
        self.crosses_layer_boundary = crosses_layer_boundary

    def __repr__(self) -> str:
        return f"HelmEdge({self.src_id} -> {self.dst_id} | Shape: {self.tensor_shape} | Bytes: {self.tensor_bytes})"


class HelmNode:
    """
    Represents a node in the Helm Graph, mirroring an FX node.
    """

    def __init__(self, node_id: int, name: str, fx_node: torch.fx.Node) -> None:
        self.id = node_id
        self.name = name
        self.fx_node = fx_node
        self.fx_node_name = fx_node.name
        self.op_type = fx_node.op
        self.target = fx_node.target
        self.module_path: str = ""
        self.layer_id: int | None = None
        self.args: list[HelmArg] = []
        self.kwargs: dict[str, Any] = {}

        # Core Properties
        self.input_shapes: list[list[int]] = []
        self.output_shapes: list[list[int]] = []
        self.flops_prefill: int = 0
        self.flops_decode: int = 0
        self.activation_bytes: int = 0
        self.param_bytes: int = 0
        self.bytes_read: int = 0
        self.bytes_written: int = 0
        self.kv_bytes_per_token: int = 0
        self.batch_size: int = 1
        self.sequence_length: int = 1

        self.in_edges: list["HelmEdge"] = []
        self.out_edges: list["HelmEdge"] = []

        self.output_dtype: torch.dtype | None = None

        # Semantic Tags
        self.is_attention: bool = False
        self.is_mlp: bool = False
        self.is_norm: bool = False
        self.is_embedding: bool = False
        self.is_output_head: bool = False
        self.is_kv_projection: bool = False

    def mark_attention(self) -> None:
        self.is_attention = True

    def mark_mlp(self) -> None:
        self.is_mlp = True

    def mark_norm(self) -> None:
        self.is_norm = True

    def mark_embedding(self) -> None:
        self.is_embedding = True

    def mark_output_head(self) -> None:
        self.is_output_head = True

    def mark_kv_projection(self) -> None:
        self.is_kv_projection = True

    def get_output_bytes_str(self) -> str:
        """Formatted string for activation bytes (e.g. '4.00 MB')."""
        b = self.activation_bytes
        if b <= 0:
            return ""
        if b < 1024:
            return f"{b} B"
        elif b < 1024**2:
            return f"{b/1024:.2f} KB"
        elif b < 1024**3:
            return f"{b/(1024**2):.2f} MB"
        else:
            return f"{b/(1024**3):.2f} GB"

    @staticmethod
    def _fmt_arg(arg: Any) -> str:
        """Format an arg for __repr__, recursing into containers."""
        if isinstance(arg, HelmNode):
            return arg.name
        if isinstance(arg, (list, tuple)):
            inner = ", ".join(HelmNode._fmt_arg(item) for item in arg)
            if isinstance(arg, list):
                return f"[{inner}]"
            # Preserve single-element tuple syntax: (x,) vs multi: (x, y)
            return f"({inner},)" if len(arg) == 1 else f"({inner})"
        return str(arg)

    def __repr__(self) -> str:
        fmt_args = [self._fmt_arg(arg) for arg in self.args]
        return (
            f"{self.name} = {self.op_type}({self.target}, args={fmt_args})"
            f" | Layer: {self.layer_id}"
            f" | FLOPs: {self.flops_prefill}"
            f" | Out: {self.get_output_bytes_str()}"
            f" | Shapes: {self.output_shapes}"
        )


class HelmGraph:
    """
    A mirrored graph representation of the FX Graph.
    """

    def __init__(self, fx_graph: torch.fx.Graph) -> None:
        self.nodes: list[HelmNode] = []
        self.edges: list[HelmEdge] = []

        self.fx_to_helm: dict[torch.fx.Node, HelmNode] = {}
        self.helm_id_to_node: dict[int, HelmNode] = {}

        self.layer_to_node_ids: dict[int, list[int]] = {}

        self.input_node_ids: list[int] = []
        self.output_node_ids: list[int] = []

        # Global Metadata
        self.hardware_meta: dict[str, Any] = {}

        self._build_from_fx(fx_graph)

    def _extract_dependencies(self, arg: Any) -> tuple[Any, list["HelmNode"]]:
        """Recursively resolve FX node references in arg to HelmNodes.

        Returns (mapped_arg, deps) where FX nodes are replaced by HelmNodes
        and deps is the flat list of HelmNodes found.
        """
        found_deps: list[HelmNode] = []

        def recursive_map(x: Any) -> Any:
            if isinstance(x, torch.fx.Node):
                if x in self.fx_to_helm:
                    helm_node = self.fx_to_helm[x]
                    found_deps.append(helm_node)
                    return helm_node
                else:
                    raise KeyError(
                        f"FX node '{x.name}' (op={x.op}) not found in "
                        f"fx_to_helm mapping graph may be corrupted or "
                        f"built out of order"
                    )
            elif isinstance(x, tuple):
                mapped = tuple(recursive_map(item) for item in x)
                try:
                    return type(x)(*mapped)
                except TypeError:
                    return mapped
            elif isinstance(x, list):
                return [recursive_map(item) for item in x]
            elif isinstance(x, dict):
                return {k: recursive_map(v) for k, v in x.items()}
            else:
                return x

        mapped_arg = recursive_map(arg)
        return mapped_arg, found_deps

    def _build_from_fx(self, fx_graph: torch.fx.Graph) -> None:
        for idx, fx_node in enumerate(fx_graph.nodes):
            helm_name = f"N{idx}"
            helm_node = HelmNode(node_id=idx, name=helm_name, fx_node=fx_node)

            self.nodes.append(helm_node)
            self.fx_to_helm[fx_node] = helm_node
            self.helm_id_to_node[helm_node.id] = helm_node

            if fx_node.op == "placeholder":
                self.input_node_ids.append(helm_node.id)
            elif fx_node.op == "output":
                assert (
                    len(self.output_node_ids) == 0
                ), "FX graph must have exactly one output node"
                self.output_node_ids.append(helm_node.id)

        for helm_node in self.nodes:
            original_args = helm_node.fx_node.args

            new_args: list[Any] = []
            all_deps: list[HelmNode] = []

            for arg in original_args:
                mapped_arg, deps = self._extract_dependencies(arg)
                new_args.append(mapped_arg)
                all_deps.extend(deps)

            # Also extract deps from kwargs (e.g. tensor args passed as keywords)
            new_kwargs: dict[str, Any] = {}
            for k, v in helm_node.fx_node.kwargs.items():
                mapped_v, deps = self._extract_dependencies(v)
                new_kwargs[k] = mapped_v
                all_deps.extend(deps)

            helm_node.args = new_args
            helm_node.kwargs = new_kwargs

            for dep in all_deps:
                edge = HelmEdge(src_id=dep.id, dst_id=helm_node.id)
                self.edges.append(edge)
                dep.out_edges.append(edge)
                helm_node.in_edges.append(edge)

    def register_layer(self, layer_id: int, node_id: int) -> None:
        """Register a node as belonging to a layer. Called by compiler passes."""
        self.layer_to_node_ids.setdefault(layer_id, []).append(node_id)

    def print_graph(self) -> None:
        logger.info("--- Helm Graph (Mirrored) ---")
        if self.hardware_meta:
            logger.info("Hardware Context: %s", self.hardware_meta)
        for node in self.nodes:
            deps = [self.helm_id_to_node[e.src_id].name for e in node.in_edges]
            logger.info("%s [Depends on: %s]", node, deps)
        logger.info("-----------------------------")

    def get_node(self, node_id: int) -> HelmNode:
        if node_id not in self.helm_id_to_node:
            raise KeyError(
                f"Node {node_id} not found in graph (total nodes: {len(self.nodes)})"
            )
        return self.helm_id_to_node[node_id]

    def get_nodes_by_layer(self, layer_id: int) -> list[HelmNode]:
        ids = self.layer_to_node_ids.get(layer_id, [])
        return [self.get_node(i) for i in ids]

    def nodes_in_order(self) -> list[HelmNode]:
        """Return nodes in topological order.

        FX graphs are constructed in topological order, so this invariant holds
        as long as nodes are only appended via _build_from_fx.
        """
        return list(self.nodes)

    def get_outgoing_edges(self, node_id: int) -> list[HelmEdge]:
        if node_id not in self.helm_id_to_node:
            raise KeyError(
                f"Node {node_id} not found in graph (total nodes: {len(self.nodes)})"
            )
        return self.helm_id_to_node[node_id].out_edges

    def get_incoming_edges(self, node_id: int) -> list[HelmEdge]:
        if node_id not in self.helm_id_to_node:
            raise KeyError(
                f"Node {node_id} not found in graph (total nodes: {len(self.nodes)})"
            )
        return self.helm_id_to_node[node_id].in_edges

    def summary(self) -> None:
        logger.info("HelmGraph Summary")
        logger.info("Nodes: %d", len(self.nodes))
        logger.info("Edges: %d", len(self.edges))
        logger.info("Inputs: %s", self.input_node_ids)
        logger.info("Outputs: %s", self.output_node_ids)
