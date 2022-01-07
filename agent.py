from mesa import Agent as Agent_mesa
from random import randrange, randint, random
from enum import IntEnum


class State(IntEnum):
    Suspectible = 0
    Exposed = 1
    Infected = 2
    Recovered = 3
    Death = 4


class AgeGroup(IntEnum):
    _60 = 0
    _60_69 = 1
    _70_79 = 2
    _80 = 3


class Agent(Agent_mesa):
    def __init__(self, id: int, model, pos=None, state=State.Suspectible, age=AgeGroup._60):
        super().__init__(id, model)
        self.pos = pos if pos is not None else (randrange(0, self.model.w), randrange(0, self.model.h))
        self.state = state
        self.age = age

    def step(self):
        if random() < 0.5:
            x, y = self.pos
            x += randint(-1, 1)
            y += randint(-1, 1)
            self.model.grid.move_agent(self, (x, y))
        self.dispatch_state()

    def advance(self):
        pass

    def dispatch_state(self):
        if self.state == State.Suspectible:
            self.is_suspectible()
        elif self.state == State.Exposed:
            self.is_exposed()
        elif self.state == State.Infected:
            self.is_infected()
        elif self.state == State.Recovered:
            self.is_recovered()
        elif self.state == State.Death:
            self.is_death()
        else:
            raise RuntimeError('Wrong agent state!')

    def is_suspectible(self):
        n = self.model.grid.iter_neighbors(self.pos, moore=True)
        for agent in n:
            if agent.state == State.Infected:
                if random() < self.model.params[0]:  # contact_rate
                    self.state = State.Exposed
                    self.model.new_exposed += 1
                    break

    def is_exposed(self):
        if random() < self.model.params[1]:  # days_to_symptoms_inv
            self.state = State.Infected
            self.prev_state = State.Exposed
            self.model.new_infected += 1

    def is_infected(self):
        if random() < self.model.params[2]:  # days_to_recovery_inv
            if random() < self.model.agent_mortality(self):
                self.state = State.Death
                self.model.new_deaths += 1
            else:
                self.state = State.Recovered
                self.model.new_recovered += 1

    def is_recovered(self):
        pass

    def is_death(self):
        pass
