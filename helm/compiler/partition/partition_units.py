import logging
from dataclasses import dataclass, field
from typing import List, Optional

from ..IR.graph import HelmGraph

logger = logging.getLogger(__name__)


@dataclass
class PartitionUnit:
    unit_id: int
    unit_type: str

    layer_start: Optional[int]
    layer_end: Optional[int]

    node_ids: List[int] = field(default_factory=list)

    # aggregated cost properties
    flops_prefill: int = 0
    flops_decode: int = 0

    param_bytes: int = 0
    activation_bytes: int = 0

    kv_bytes_per_token: int = 0

    contains_attention: bool = False
    contains_mlp: bool = False
    contains_norm: bool = False
    contains_kv_projection: bool = False


class PartitionUnitBuilder:
    def __init__(self, helm_graph: HelmGraph) -> None:
        self.graph = helm_graph
        self.units: List[PartitionUnit] = []

    def build(self) -> List[PartitionUnit]:
        self.units = []
        embed_nodes, output_nodes = self._scan_special_nodes()
        self._build_embedding_unit(embed_nodes)
        self._build_layer_units()
        self._build_output_unit(output_nodes)
        return self.units

    def _scan_special_nodes(self) -> tuple:
        """Single pass over graph.nodes to collect embedding and output head IDs."""
        embed_nodes: List[int] = []
        output_nodes: List[int] = []
        for node in self.graph.nodes:
            if node.is_embedding:
                embed_nodes.append(node.id)
            if node.is_output_head:
                output_nodes.append(node.id)
        return embed_nodes, output_nodes

    def _build_embedding_unit(self, embed_nodes: List[int]) -> None:
        if not embed_nodes:
            return

        unit = PartitionUnit(
            unit_id=len(self.units),
            unit_type="embedding",
            layer_start=None,
            layer_end=None,
            node_ids=list(embed_nodes),
        )
        self._aggregate_cost(unit)
        self.units.append(unit)

    def _build_layer_units(self) -> None:
        layer_ids = sorted(self.graph.layer_to_node_ids.keys())

        for layer_id in layer_ids:
            node_ids = self.graph.layer_to_node_ids[layer_id]
            if not node_ids:
                logger.warning(
                    "PartitionUnitBuilder: layer %d has no nodes — skipping", layer_id
                )
                continue

            unit = PartitionUnit(
                unit_id=len(self.units),
                unit_type="transformer_block",
                layer_start=layer_id,
                layer_end=layer_id,
                node_ids=list(node_ids),
            )

            self._aggregate_cost(unit)
            self.units.append(unit)

    def _build_output_unit(self, output_nodes: List[int]) -> None:
        if not output_nodes:
            return

        unit = PartitionUnit(
            unit_id=len(self.units),
            unit_type="output",
            layer_start=None,
            layer_end=None,
            node_ids=list(output_nodes),
        )

        self._aggregate_cost(unit)
        self.units.append(unit)

    def _aggregate_cost(self, unit: PartitionUnit) -> None:
        for nid in unit.node_ids:
            node = self.graph.helm_id_to_node.get(nid)
            if node is None:
                raise RuntimeError(
                    f"PartitionUnitBuilder: node id {nid} in unit {unit.unit_id} "
                    f"({unit.unit_type}) not found in helm_id_to_node. "
                    f"The graph may have been modified after units were built."
                )

            unit.flops_prefill += node.flops_prefill
            unit.flops_decode += node.flops_decode

            unit.param_bytes += node.param_bytes
            unit.activation_bytes += node.activation_bytes

            unit.kv_bytes_per_token += node.kv_bytes_per_token

            if node.is_attention:
                unit.contains_attention = True

            if node.is_mlp:
                unit.contains_mlp = True

            if node.is_norm:
                unit.contains_norm = True

            if node.is_kv_projection:
                unit.contains_kv_projection = True

    def print_units(self) -> None:
        logger.info("[PartitionUnitBuilder] Units")
        for unit in self.units:
            logger.info(
                "Unit %d | type=%s | layers=%s-%s | nodes=%d | flops_prefill=%d | flops_decode=%d",
                unit.unit_id,
                unit.unit_type,
                unit.layer_start,
                unit.layer_end,
                len(unit.node_ids),
                unit.flops_prefill,
                unit.flops_decode,
            )
