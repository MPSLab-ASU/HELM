import torch.fx as fx
from collections import defaultdict

from helm.compiler.partition.partition_plan import PartitionPlan
from helm.runtime.stage import Stage

class StageFXBuilder:
    def __init__(self, gm: fx.GraphModule, helm_graph, units, partition_plan: PartitionPlan):
        self.gm = gm
        self.helm_graph = helm_graph
        self.units = units
        self.plan = partition_plan
        self.fx_to_stage = {}

    def build(self) -> list[Stage]:
        stages, stage_devices = self._compute_stages()

        runtime_stages = []

        for stage_id in sorted(stages.keys()):
            stage_nodes = stages[stage_id]
            stage_graph = self._build_stage(stage_id, stage_nodes)
            device = stage_devices[stage_id]

            runtime_stages.append(Stage(
                stage_id=stage_id,
                device=device,
                module=stage_graph
            ))

        return runtime_stages

    def _compute_stages(self):
        unit_to_stage = {}
        stage_devices = {}

        # Map units to stages from the actual PartitionPlan
        for stage in self.plan.stages:
            stage_devices[stage.stage_id] = stage.device_id
            for u in stage.units:
                unit_to_stage[u.unit_id] = stage.stage_id

        # Map node_id -> stage_id
        node_id_to_stage = {}
        for u in self.units:
            if u.unit_id in unit_to_stage:
                for nid in u.node_ids:
                    node_id_to_stage[nid] = unit_to_stage[u.unit_id]

        fx_nodes = list(self.gm.graph.nodes)

        # Map fx nodes explicitly associated with units
        for node in self.helm_graph.nodes:
            if node.id in node_id_to_stage:
                self.fx_to_stage[node.fx_node] = node_id_to_stage[node.id]

        # Explicitly force lm_head to the final stage from the plan.  Stage IDs
        # can come from plan overrides, so do not assume dense 0..N-1 numbering.
        final_stage_id = self.plan.stages[-1].stage_id
        for node in fx_nodes:
            if node.op == "call_module" and "lm_head" in str(node.target):
                self.fx_to_stage[node] = final_stage_id

        # Unassigned nodes (like get_attr, inputs, missing ops)
        def get_node_stage(n):
            if isinstance(n, fx.Node):
                return self.fx_to_stage.get(n)
            if isinstance(n, (list, tuple)):
                for item in n:
                    s = get_node_stage(item)
                    if s is not None:
                        return s
            if isinstance(n, dict):
                for v in n.values():
                    s = get_node_stage(v)
                    if s is not None:
                        return s
            return None

        # Pass 1: inherit from consumers (backward traversal)
        for node in reversed(fx_nodes):
            if node not in self.fx_to_stage:
                s = get_node_stage(list(node.users.keys()))
                if s is not None:
                    self.fx_to_stage[node] = s

        # Pass 2: inherit from upstream (forward traversal)
        for node in fx_nodes:
            if node not in self.fx_to_stage:
                s = get_node_stage(node.args) or get_node_stage(node.kwargs)
                if s is not None:
                    self.fx_to_stage[node] = s

        unassigned = [node for node in fx_nodes if node not in self.fx_to_stage]
        if unassigned:
            details = ", ".join(
                f"{node.name}(op={node.op}, target={node.target})"
                for node in unassigned
            )
            raise RuntimeError(
                "StageFXBuilder could not infer stage for FX node(s): "
                f"{details}. Ensure PartitionUnitBuilder covers the node or "
                "that it is connected to a staged producer/consumer."
            )

        # Collect nodes per stage
        stage_nodes = defaultdict(list)
        for node in fx_nodes:
            stage_nodes[self.fx_to_stage[node]].append(node)

        return stage_nodes, stage_devices

    def _build_stage(self, stage_id, nodes):
        new_graph = fx.Graph()
        env = {}

        def get_or_create(n):
            if n in env:
                return env[n]
            if self.fx_to_stage[n] != stage_id:
                # Cross-stage input! Create placeholder.
                placeholder = new_graph.placeholder(n.name)
                env[n] = placeholder
                placeholder.meta = n.meta.copy()
                return placeholder
            raise RuntimeError(f"Node '{n.name}' missing in env despite being in stage {stage_id}. Execution order incorrect?")

        is_final_stage = any(n.op == 'output' for n in nodes)
        stage_outputs = []

        for node in nodes:
            if node.op == 'output':
                continue # Handled at the end

            if node.op == 'placeholder':
                new_node = new_graph.placeholder(node.name)
                env[node] = new_node
                new_node.meta = node.meta.copy()
            else:
                new_node = new_graph.node_copy(node, get_or_create)

                env[node] = new_node

        if is_final_stage:
            # We preserve the global output node verbatim
            global_output = next(n for n in nodes if n.op == 'output')
            new_graph.node_copy(global_output, get_or_create)
        else:
            # Mark values that later stages will consume as outputs of this stage
            for node in nodes:
                if node.op == 'output':
                    continue

                is_cross_stage = False
                for user in node.users:
                    if self.fx_to_stage[user] != stage_id:
                        is_cross_stage = True
                        break

                if is_cross_stage:
                    stage_outputs.append(node)

            # Return a dict mapping original node name to the value so that the next stage can cleanly kwarg it
            output_dict = {n.name: env[n] for n in stage_outputs}
            new_graph.output(output_dict)

        new_graph.lint()
        return fx.GraphModule(self.gm, new_graph)
