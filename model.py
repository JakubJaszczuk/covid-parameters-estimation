from mesa import Model as Model_mesa
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from random import sample
try:
    from .agent import Agent, State, AgeGroup
except ImportError:
    from agent import Agent, State, AgeGroup


class Model(Model_mesa):
    def __init__(self, population=400, exposed=2, infected=0, params=(0.1, 0.1, 0.1), w=32, h=32, max_iter=100):
        self.running = True
        self.params = params
        self.max_iter = max_iter
        self.iter = 0
        self.population = population
        self.w = w
        self.h = h
        self.grid = MultiGrid(w, h, torus=True)
        self.agents = []
        self.schedule = RandomActivation(self)
        # FIX
        self.new_exposed = 0
        self.new_infected = 0
        self.new_recovered = 0
        self.new_deaths = 0
        # Mortality
        self.mortality_60 = 0.0032
        self.mortality_60_69 = 0.035
        self.mortality_70_79 = 0.0861
        self.mortality_80 = 0.14
        # Age groups
        #self.pop_60_69 = int(self.population * 0.1357)
        #self.pop_70_79 = int(self.population * 0.0697)
        #self.pop_80 = int(self.population * 0.0437)
        self.pop_60_69 = int(self.population * 0.1117)
        self.pop_70_79 = int(self.population * 0.088)
        self.pop_80 = int(self.population * 0.046)
        # Make agents
        self.make_agents(exposed, infected)

    def make_agents(self, exposed_count: int, infected_count: int):
        for i in range(self.population):
            agent = Agent(i, self, state=State.Suspectible)
            self.grid.place_agent(agent, agent.pos)
            self.schedule.add(agent)
            self.agents.append(agent)
        smp = sample(self.agents, exposed_count)
        for a in smp:
            a.state = State.Exposed
        smp = sample(self.agents, infected_count)
        for a in smp:
            a.state = State.Infected
        # Age groups
        smp = sample(self.agents, self.pop_60_69)
        for a in smp:
            a.age = AgeGroup._60_69
        f = list(e for e in self.agents if e.age == AgeGroup._60)
        smp = sample(f, self.pop_70_79)
        for a in smp:
            a.age = AgeGroup._70_79
        f = list(e for e in self.agents if e.age == AgeGroup._60)
        smp = sample(f, self.pop_80)
        for a in smp:
            a.age = AgeGroup._80

    def step(self):
        self.clear_counts()
        self.schedule.step()
        self.iter += 1
        # Warunek stopu
        if self.iter >= self.max_iter:
            self.running = False

    def count(self):
        suspectible = 0
        exposed = 0
        infected = 0
        recovered = 0
        deaths = 0
        for a in self.agents:
            if a.state == State.Suspectible:
                suspectible += 1
            elif a.state == State.Exposed:
                exposed += 1
            elif a.state == State.Infected:
                infected += 1
            elif a.state == State.Recovered:
                recovered += 1
            elif a.state == State.Death:
                deaths += 1
        return exposed, infected, recovered, deaths, suspectible

    def clear_counts(self):
        self.new_exposed = 0
        self.new_infected = 0
        self.new_recovered = 0
        self.new_deaths = 0

    def make_new_counts(self):
        return self.new_exposed, self.new_infected, self.new_recovered, self.new_deaths

    def agent_mortality(self, agent):
        if agent.age == AgeGroup._60:
            return self.mortality_60
        elif agent.age == AgeGroup._60_69:
            return self.mortality_60_69
        elif agent.age == AgeGroup._70_79:
            return self.mortality_70_79
        elif agent.age == AgeGroup._80:
            return self.mortality_80
        else:
            return 0
