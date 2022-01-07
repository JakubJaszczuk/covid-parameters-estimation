class ModelRunner:
    def __init__(self, model):
        self.model = model
        self.exposed = []
        self.infected = []
        self.recovered = []
        self.deaths = []

    def run(self):
        while self.model.running:
            self.model.step()
            counts = self.model.make_new_counts()
            self.exposed.append(counts[0])
            self.infected.append(counts[1])
            self.recovered.append(counts[2])
            self.deaths.append(counts[3])

    def __getitem__(self, key: str):
        if key == 'infected':
            return self.infected
        elif key == 'recovered':
            return self.recovered
        elif key == 'exposed':
            return self.exposed
        elif key == 'deaths':
            return self.deaths
        else:
            raise KeyError
