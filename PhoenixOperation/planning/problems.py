from __future__ import annotations

from planning.pddl import Problem
from planning.domain import DOMAIN
from world.rescue_layout import RescueLayout
from world.rescue_rules import build_initial_state


class SimpleRescueProblem(Problem):
    def __init__(self, layout: RescueLayout) -> None:
        initial_state, objects = build_initial_state(layout)

        ### Your code here ###
        goal = frozenset(("Rescued", patient) for patient in objects["patients"][:1])
        ### End of your code ###

        super().__init__(initial_state, goal, DOMAIN, objects)
        self.layout = layout


class MultiRescueProblem(Problem):
    def __init__(self, layout: RescueLayout) -> None:
        initial_state, objects = build_initial_state(layout)

        ### Your code here ###
        goal = frozenset(("Rescued", patient) for patient in objects["patients"])
        ### End of your code ###

        super().__init__(initial_state, goal, DOMAIN, objects)
        self.layout = layout
