from .particle_swarm_optimization import ParticleSwarmOptimization
from .plant_propagation_algorithm import PlantPropagationAlgorithm
from .genetic_algorithm import GeneticAlgorithm
from .simulated_annealing import SimulatedAnnealing
from .differential_evolution import DifferentialEvolution
from .evolution_strategy import EvolutionStrategy
from .one_plus_one_es import OnePlusOneES
from .unbounded_integer_ea import UnboundedIntegerEA, test_discrete_bbob
from .gsemo import GSEMO, gsemo_onemax_zeromax

real = (
    ParticleSwarmOptimization, 
    SimulatedAnnealing,
    EvolutionStrategy,
    OnePlusOneES,
    DifferentialEvolution,
)

binary = (
    GeneticAlgorithm,
)

other = (
    PlantPropagationAlgorithm,
    UnboundedIntegerEA, 
)

multi = (
    GSEMO, 
)


if __name__ == '__main__':
    test_discrete_bbob(1, 1, 100, .0001) 
    
