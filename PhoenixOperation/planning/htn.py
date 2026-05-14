from __future__ import annotations

from collections import deque
from itertools import permutations

from planning.pddl import Action, Problem, apply_action, is_applicable
from planning.domain import PICKUP, PUTDOWN, RESCUE, SETUP_SUPPLIES, MOVE
from planning.utils import Queue


# ---------------------------------------------------------------------------
# HTN Infrastructure
# ---------------------------------------------------------------------------


class HLA:
    """
    A High-Level Action (HLA) in HTN planning.

    An HLA is an abstract task that can be refined into sequences of
    more primitive actions (or other HLAs). Each refinement is a list
    of HLA or Action objects.

    name:        Human-readable name for display
    refinements: List of possible refinements, each a list of HLA/Action objects
    """

    def __init__(self, name: str, refinements: list[list] | None = None) -> None:
        self.name = name
        self.refinements = refinements or []

    def __repr__(self) -> str:
        return f"HLA({self.name})"


def is_primitive(action: Action | HLA) -> bool:
    """Return True if action is a primitive (grounded Action), False if it is an HLA."""
    return isinstance(action, Action)


def is_plan_primitive(plan: list[Action | HLA]) -> bool:
    """Return True if every step in the plan is a primitive action."""
    return all(is_primitive(step) for step in plan)


# ---------------------------------------------------------------------------
# Punto 5a – hierarchicalSearch
# ---------------------------------------------------------------------------


def hierarchicalSearch(problem: Problem, hlas: list[HLA]) -> list[Action]:
    """
    HTN planning via BFS over hierarchical plan refinements.

    Start with an initial plan containing a single top-level HLA.
    At each step, find the first non-primitive step in the plan and
    replace it with one of its refinements. Continue until the plan
    is fully primitive and achieves the goal when executed from the
    initial state.

    Returns a list of primitive Action objects, or [] if no plan found.

    Tip: The search space consists of (partial plan, current plan index) pairs.
         Use a Queue (BFS) to explore all refinement choices fairly.
         A plan is a solution when:
           1. It contains only primitive actions (is_plan_primitive), AND
           2. Executing it from the initial state reaches a goal state.
         To simulate execution, apply each action in order using apply_action().
    """
    def execute(plan: list[Action]) -> tuple[bool, frozenset]:
        state = problem.initial_state
        for action in plan:
            if not is_applicable(state, action):
                return False, state
            state = apply_action(state, action)
        return True, state

    def primitive_prefix_is_valid(plan: list[Action | HLA]) -> bool:
        state = problem.initial_state
        for step in plan:
            if not isinstance(step, Action):
                return True
            if not is_applicable(state, step):
                return False
            state = apply_action(state, step)
        return True

    if not hlas:
        return []

    frontier = Queue()
    for hla in hlas:
        frontier.push([hla])

    visited: set[tuple[str, ...]] = set()

    while not frontier.isEmpty():
        plan = frontier.pop()
        key = tuple(step.name for step in plan)
        if key in visited:
            continue
        visited.add(key)
        if not primitive_prefix_is_valid(plan):
            continue

        if is_plan_primitive(plan):
            primitive_plan = [step for step in plan if isinstance(step, Action)]
            valid, final_state = execute(primitive_plan)
            if valid and problem.isGoalState(final_state):
                return primitive_plan
            continue

        hla_index = next(i for i, step in enumerate(plan) if not is_primitive(step))
        hla = plan[hla_index]
        for refinement in hla.refinements:
            frontier.push(plan[:hla_index] + refinement + plan[hla_index + 1 :])

    return []


# ---------------------------------------------------------------------------
# Punto 5b – HLA Definitions
# ---------------------------------------------------------------------------


def build_htn_hierarchy(problem: Problem) -> list[HLA]:
    """
    Build HTN HLAs for the rescue domain.

    The hierarchy defines four HLA types:
      - Navigate(from, to):       Move the robot step by step from one cell to another
      - PrepareSupplies(s, m):    Collect supplies and set them up at the medical post
      - ExtractPatient(p, m):     Pick up the patient and bring them to the medical post
      - FullRescueMission(s,p,m): Complete one rescue: prepare supplies + extract + rescue

    Refinements are built from the ground state to generate concrete Action objects.

    Tip: Refinements for Navigate are all single-step Move sequences between
         adjacent cells. PrepareSupplies and ExtractPatient chain Navigate HLAs
         with primitive PickUp, SetupSupplies, PutDown, and Rescue actions.
    """
    cells = problem.objects["cells"]
    patients = problem.objects["patients"]
    supplies = problem.objects["supplies"]
    medical_posts = problem.objects["medical_posts"]
    if not patients or not supplies or not medical_posts:
        return []

    robot = problem.objects["robots"][0]
    medical_post = medical_posts[0]

    at_fluents = [fluent for fluent in problem.initial_state if fluent[0] == "At"]
    locations = {entity: loc for _, entity, loc in at_fluents}

    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {cell: [] for cell in cells}
    for fluent in problem.initial_state:
        if fluent[0] == "Adjacent":
            adjacency[fluent[1]].append(fluent[2])

    def ground(schema, binding: dict[str, object]) -> Action:
        return schema.ground(binding)

    def move_action(origin, destination) -> Action:
        return ground(
            MOVE,
            {"r": robot, "from_cell": origin, "to_cell": destination},
        )

    def find_paths(start, goal, limit: int = 3) -> list[list[tuple[int, int]]]:
        if start == goal:
            return [[start]]

        paths: list[list[tuple[int, int]]] = []
        frontier = deque([[start]])
        shortest_len: int | None = None
        max_extra_steps = 2

        while frontier and len(paths) < limit:
            path = frontier.popleft()
            if shortest_len is not None and len(path) > shortest_len + max_extra_steps:
                continue

            current = path[-1]
            for neighbor in adjacency.get(current, []):
                if neighbor in path:
                    continue
                next_path = path + [neighbor]
                if neighbor == goal:
                    shortest_len = len(next_path) if shortest_len is None else shortest_len
                    paths.append(next_path)
                elif shortest_len is None or len(next_path) < shortest_len + max_extra_steps:
                    frontier.append(next_path)

        return paths

    navigate_cache: dict[tuple[tuple[int, int], tuple[int, int]], HLA] = {}

    def navigate(start, goal) -> HLA:
        key = (start, goal)
        if key not in navigate_cache:
            refinements = []
            for path in find_paths(start, goal):
                moves = [
                    move_action(path[i], path[i + 1])
                    for i in range(len(path) - 1)
                ]
                refinements.append(moves)
            navigate_cache[key] = HLA(f"Navigate({start}->{goal})", refinements)
        return navigate_cache[key]

    def prepare_supplies(supply: str, start, post) -> HLA:
        supply_loc = locations[supply]
        setup = ground(
            SETUP_SUPPLIES,
            {"r": robot, "s": supply, "loc": post},
        )
        pickup = ground(
            PICKUP,
            {"r": robot, "obj": supply, "loc": supply_loc},
        )
        return HLA(
            f"PrepareSupplies({supply},{post})",
            [[navigate(start, supply_loc), pickup, navigate(supply_loc, post), setup]],
        )

    def extract_patient(patient: str, start, post) -> HLA:
        patient_loc = locations[patient]
        pickup = ground(
            PICKUP,
            {"r": robot, "obj": patient, "loc": patient_loc},
        )
        putdown = ground(
            PUTDOWN,
            {"r": robot, "obj": patient, "loc": post},
        )
        rescue = ground(
            RESCUE,
            {"r": robot, "p": patient, "loc": post},
        )
        return HLA(
            f"ExtractPatient({patient},{post})",
            [[navigate(start, patient_loc), pickup, navigate(patient_loc, post), putdown, rescue]],
        )

    def full_rescue_mission(
        supply: str | None,
        patient: str,
        start,
        post,
        prepare: bool = True,
    ) -> HLA:
        if not prepare or supply is None:
            return HLA(
                f"FullRescueMission({patient},{post})",
                [[extract_patient(patient, start, post)]],
            )

        return HLA(
            f"FullRescueMission({supply},{patient},{post})",
            [[
                prepare_supplies(supply, start, post),
                extract_patient(patient, post, post),
            ]],
        )

    initial_robot_pos = locations[robot]
    patient_orders = list(permutations(patients))

    root_refinements = []
    for patient_order in patient_orders:
        for first_supply in supplies:
            mission_start = initial_robot_pos
            missions = []
            for index, patient in enumerate(patient_order):
                missions.append(
                    full_rescue_mission(
                        first_supply if index == 0 else None,
                        patient,
                        mission_start,
                        medical_post,
                        prepare=index == 0,
                    )
                )
                mission_start = medical_post
            root_refinements.append(missions)

    return [HLA("FullRescueMissionRoot", root_refinements)]
