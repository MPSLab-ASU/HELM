from dataclasses import dataclass


@dataclass
class ExecutionSchedule:
    mode: str
    num_stages: int
    microbatches: int


class Scheduler:
    """
    Simple scheduler.

    Currently supports:
        sequential execution
    """

    def build(self, partition_plan):

        num_stages = len(partition_plan.stages)

        return ExecutionSchedule(
            mode="sequential",
            num_stages=num_stages,
            microbatches=1,
        )
