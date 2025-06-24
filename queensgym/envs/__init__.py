from dataclasses import dataclass
from dataclasses import field
from typing import Any


@dataclass(frozen=True, slots=True)
class Step:
    action: Any
    reward: float
    next_state: Any
    terminal: bool


@dataclass
class Episode:
    initial: Any
    steps: list[Step] = field(init=False, default_factory=list)

    def is_complete(self) -> bool:
        return self.steps[-1].terminal if self.steps else False

    def get_total_reward(self, t: int | None = None) -> float:
        if t is None:
            return sum(step.reward for step in self.steps)

        if 0 <= t <= len(self.steps):
            return sum(step.reward for step in self.steps[:t])

        raise IndexError(f"Invalid step index: {t}. Must be in range [1, {len(self.steps)}].")

    def add_step(self, step: Step) -> None:
        if not isinstance(step, Step):
            raise TypeError(f"Expected Step instance, got {type(step)}.")

        if self.is_complete():
            raise ValueError("Cannot add step to a completed episode.")

        self.steps.append(step)
