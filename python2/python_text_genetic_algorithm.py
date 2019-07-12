from datetime import datetime
import random
import string
from tqdm import tqdm

pop_size = 512
generation_count = 10000
mutation_rate = 0.01
alphabet = string.ascii_lowercase + " " + string.digits


def rand_char():
    return random.choice(alphabet)


text_sequence = list(rand_char() * 100)


class Gene:
    def __init__(self):
        self.text = []
        self.score = 0
        self.fitness = 0

    def crossover(self, other):
        child1 = Gene()
        child2 = Gene()
        point = random.randint(0, len(text_sequence))
        child1.text = self.text[:point] + other.text[point:]
        child2.text = other.text[:point] + self.text[point:]

        return child1, child2

    def mutate(self):
        for i in range(len(text_sequence)):
            if random.random() < mutation_rate:
                self.text[i] = rand_char()

    @classmethod
    def initialize(cls):
        new_gene = cls()
        for i in range(len(text_sequence)):
            new_gene.text.append(rand_char())

        return new_gene

    def calc_score(self):
        same = 0
        for i in range(len(text_sequence)):
            if self.text[i] == text_sequence[i]:
                same += 1

        return same


start = datetime.now()

pop = []

for i in range(pop_size):
    pop.append(Gene.initialize())

top_gene = Gene()
with tqdm(total=generation_count, disable=False) as t:
    while top_gene.score != len(text_sequence):  # each generation
        t.set_postfix(str="score:{}/{}".format(str(top_gene.score), str(len(text_sequence))))
        t.update()

        score_sum = 0
        for gene in pop:  # each gene
            gene.score = gene.calc_score()
            score_sum += gene.score

        top_gene = pop[0]
        for gene in pop:  # calc fitness
            gene.fitness = gene.score / score_sum
            if gene.fitness > top_gene.fitness:
                top_gene = gene

        new_pop = []
        for k in range(int(len(pop) / 2)):  # each gene
            childs = random.choices(population=pop, k=4)
            child1 = max(childs[:2], key=lambda x: x.fitness)
            child2 = max(childs[2:], key=lambda x: x.fitness)
            # for i in range(2):  # 2 times
            #     rand = random.random()
            #     sum_up = 0
            #     for j in range(pop_size):
            #         sum_up += pop[j].fitness
            #         if sum_up > rand:
            #             childs.append(pop[j])
            #             break
            child1, child2 = child1.crossover(child2)
            child1.mutate()
            child2.mutate()
            new_pop.append(child1)
            new_pop.append(child2)

        pop[random.randint(0, len(pop) - 1)] = top_gene
        pop = new_pop

end = datetime.now()
print("{}".format(str((end - start).total_seconds() * 1000)))