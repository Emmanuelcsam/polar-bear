#!/usr/bin/env python3
"""
Neural Network Parameter Tuning System
Provides intelligent parameter optimization for all connected nodes
"""

import os
import sys
import json
import time
import random
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    import optuna
except ImportError:
    optuna = None
try:
    from scipy import optimize
except ImportError:
    optimize = None
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel
except ImportError:
    GaussianProcessRegressor = None
    RBF = None
    ConstantKernel = None

from .logger import logger, LogChannel, TimedOperation
from .node_base import BaseNode, NodeMetrics


class TuningStrategy(Enum):
    """Parameter tuning strategies"""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    GENETIC = "genetic"
    GRADIENT = "gradient"
    REINFORCEMENT = "reinforcement"
    ADAPTIVE = "adaptive"


class ParameterType(Enum):
    """Types of tunable parameters"""
    CONTINUOUS = "continuous"
    INTEGER = "integer"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"


@dataclass
class TunableParameter:
    """Definition of a tunable parameter"""
    name: str
    type: ParameterType
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    choices: Optional[List[Any]] = None
    current_value: Any = None
    best_value: Any = None
    description: str = ""
    node_id: str = ""
    importance: float = 1.0
    
    def sample(self, strategy: TuningStrategy = TuningStrategy.RANDOM_SEARCH) -> Any:
        """Sample a new parameter value"""
        if self.type == ParameterType.CONTINUOUS:
            if strategy == TuningStrategy.RANDOM_SEARCH:
                return random.uniform(self.min_value, self.max_value)
            else:
                # For other strategies, return current value (will be handled by strategy)
                return self.current_value
        
        elif self.type == ParameterType.INTEGER:
            if strategy == TuningStrategy.RANDOM_SEARCH:
                return random.randint(self.min_value, self.max_value)
            else:
                return self.current_value
        
        elif self.type == ParameterType.CATEGORICAL:
            if strategy == TuningStrategy.RANDOM_SEARCH:
                return random.choice(self.choices)
            else:
                return self.current_value
        
        elif self.type == ParameterType.BOOLEAN:
            if strategy == TuningStrategy.RANDOM_SEARCH:
                return random.choice([True, False])
            else:
                return self.current_value


@dataclass
class TuningResult:
    """Result of a parameter tuning run"""
    parameters: Dict[str, Any]
    objective_value: float
    metrics: Dict[str, float]
    timestamp: float = field(default_factory=time.time)
    iteration: int = 0
    
    def is_better_than(self, other: 'TuningResult', minimize: bool = True) -> bool:
        """Check if this result is better than another"""
        if minimize:
            return self.objective_value < other.objective_value
        else:
            return self.objective_value > other.objective_value


@dataclass
class TuningObjective:
    """Objective function for parameter tuning"""
    name: str
    function: Callable[[Dict[str, BaseNode], Dict[str, Any]], float]
    minimize: bool = True
    weight: float = 1.0
    constraints: List[Callable[[Dict[str, Any]], bool]] = field(default_factory=list)


class ParameterTuner:
    """Main parameter tuning system"""
    
    def __init__(self, max_workers: int = 4):
        """Initialize parameter tuner"""
        self.nodes: Dict[str, BaseNode] = {}
        self.parameters: Dict[str, TunableParameter] = {}
        self.objectives: List[TuningObjective] = []
        self.results: List[TuningResult] = []
        self.best_result: Optional[TuningResult] = None
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        
        # Tuning state
        self.current_iteration = 0
        self.max_iterations = 100
        self.convergence_threshold = 0.001
        self.patience = 10
        self.no_improvement_count = 0
        
        logger.info(LogChannel.NEURAL, "Parameter tuner initialized")
    
    def register_node(self, node: BaseNode):
        """Register a node for parameter tuning"""
        with self._lock:
            self.nodes[node.metadata.id] = node
            
            # Extract tunable parameters
            for param_name, param_info in node.tunable_params.items():
                full_name = f"{node.metadata.id}.{param_name}"
                
                # Determine parameter type
                if param_info['type'] == int:
                    param_type = ParameterType.INTEGER
                elif param_info['type'] == float:
                    param_type = ParameterType.CONTINUOUS
                elif param_info['type'] == bool:
                    param_type = ParameterType.BOOLEAN
                else:
                    param_type = ParameterType.CATEGORICAL
                
                # Create tunable parameter
                param = TunableParameter(
                    name=full_name,
                    type=param_type,
                    min_value=param_info.get('min'),
                    max_value=param_info.get('max'),
                    current_value=param_info.get('current_value'),
                    description=param_info.get('description', ''),
                    node_id=node.metadata.id
                )
                
                self.parameters[full_name] = param
                
            logger.info(LogChannel.NEURAL, f"Registered node for tuning: {node.metadata.name}",
                       parameters=len(node.tunable_params))
    
    def add_objective(self, objective: TuningObjective):
        """Add an optimization objective"""
        self.objectives.append(objective)
        logger.info(LogChannel.NEURAL, f"Added tuning objective: {objective.name}")
    
    def _evaluate_objective(self, params: Dict[str, Any]) -> float:
        """Evaluate the objective function with given parameters"""
        # Apply parameters to nodes
        with self._lock:
            for param_name, value in params.items():
                if '.' in param_name:
                    node_id, param_key = param_name.rsplit('.', 1)
                    if node_id in self.nodes:
                        self.nodes[node_id].tune_parameter(param_key, value)
        
        # Evaluate all objectives
        total_objective = 0.0
        metrics = {}
        
        for objective in self.objectives:
            try:
                value = objective.function(self.nodes, params)
                weighted_value = value * objective.weight
                
                if not objective.minimize:
                    weighted_value = -weighted_value
                
                total_objective += weighted_value
                metrics[objective.name] = value
                
            except Exception as e:
                logger.error(LogChannel.NEURAL, f"Objective evaluation failed: {objective.name}",
                           error=str(e))
                return float('inf') if objective.minimize else float('-inf')
        
        return total_objective
    
    def tune(self, strategy: TuningStrategy = TuningStrategy.BAYESIAN,
             max_iterations: int = 100, **kwargs) -> TuningResult:
        """Run parameter tuning"""
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self.no_improvement_count = 0
        
        logger.info(LogChannel.NEURAL, f"Starting parameter tuning with {strategy.value}",
                   parameters=len(self.parameters), iterations=max_iterations)
        
        try:
            if strategy == TuningStrategy.GRID_SEARCH:
                return self._tune_grid_search(**kwargs)
            elif strategy == TuningStrategy.RANDOM_SEARCH:
                return self._tune_random_search(**kwargs)
            elif strategy == TuningStrategy.BAYESIAN:
                return self._tune_bayesian(**kwargs)
            elif strategy == TuningStrategy.GENETIC:
                return self._tune_genetic(**kwargs)
            elif strategy == TuningStrategy.GRADIENT:
                return self._tune_gradient(**kwargs)
            elif strategy == TuningStrategy.ADAPTIVE:
                return self._tune_adaptive(**kwargs)
            else:
                raise ValueError(f"Unknown tuning strategy: {strategy}")
            
        finally:
            logger.info(LogChannel.NEURAL, "Parameter tuning completed",
                       best_objective=self.best_result.objective_value if self.best_result else None)
    
    def _tune_random_search(self, n_samples: int = 100, **kwargs) -> TuningResult:
        """Random search parameter tuning"""
        best_result = None
        
        for i in range(min(n_samples, self.max_iterations)):
            if self._stop_event.is_set():
                break
            
            # Sample random parameters
            params = {}
            for param_name, param in self.parameters.items():
                params[param_name] = param.sample(TuningStrategy.RANDOM_SEARCH)
            
            # Evaluate
            with TimedOperation(f"random_search_iteration_{i}"):
                objective_value = self._evaluate_objective(params)
            
            # Create result
            result = TuningResult(
                parameters=params.copy(),
                objective_value=objective_value,
                metrics={},
                iteration=i
            )
            
            self.results.append(result)
            
            # Update best
            if best_result is None or result.is_better_than(best_result, minimize=True):
                best_result = result
                self.best_result = result
                self.no_improvement_count = 0
                logger.info(LogChannel.NEURAL, f"New best found at iteration {i}",
                           objective=objective_value)
            else:
                self.no_improvement_count += 1
            
            # Early stopping
            if self.no_improvement_count >= self.patience:
                logger.info(LogChannel.NEURAL, f"Early stopping at iteration {i}")
                break
            
            self.current_iteration = i
        
        return best_result
    
    def _tune_bayesian(self, n_trials: int = 100, **kwargs) -> TuningResult:
        """Bayesian optimization using Optuna"""
        if not optuna:
            logger.warning(LogChannel.NEURAL, "Optuna not available, falling back to random search")
            return self._tune_random_search(n_samples=n_trials, **kwargs)
        
        def objective(trial):
            params = {}
            
            # Sample parameters using Optuna
            for param_name, param in self.parameters.items():
                if param.type == ParameterType.CONTINUOUS:
                    value = trial.suggest_float(param_name, param.min_value, param.max_value)
                elif param.type == ParameterType.INTEGER:
                    value = trial.suggest_int(param_name, param.min_value, param.max_value)
                elif param.type == ParameterType.CATEGORICAL:
                    value = trial.suggest_categorical(param_name, param.choices)
                elif param.type == ParameterType.BOOLEAN:
                    value = trial.suggest_categorical(param_name, [True, False])
                
                params[param_name] = value
            
            # Evaluate objective
            return self._evaluate_objective(params)
        
        # Create Optuna study
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Optimize
        study.optimize(
            objective,
            n_trials=min(n_trials, self.max_iterations),
            n_jobs=1,  # We handle parallelism ourselves
            callbacks=[lambda study, trial: self._optuna_callback(study, trial)]
        )
        
        # Get best result
        best_trial = study.best_trial
        best_result = TuningResult(
            parameters=best_trial.params,
            objective_value=best_trial.value,
            metrics={},
            iteration=best_trial.number
        )
        
        self.best_result = best_result
        return best_result
    
    def _optuna_callback(self, study, trial):
        """Callback for Optuna optimization"""
        self.current_iteration = trial.number
        
        # Create result
        result = TuningResult(
            parameters=trial.params,
            objective_value=trial.value,
            metrics={},
            iteration=trial.number
        )
        self.results.append(result)
        
        # Check for early stopping
        if self._stop_event.is_set():
            study.stop()
    
    def _tune_genetic(self, population_size: int = 50, generations: int = 100,
                     mutation_rate: float = 0.1, crossover_rate: float = 0.8,
                     **kwargs) -> TuningResult:
        """Genetic algorithm parameter tuning"""
        
        # Initialize population
        population = []
        for _ in range(population_size):
            individual = {}
            for param_name, param in self.parameters.items():
                individual[param_name] = param.sample(TuningStrategy.RANDOM_SEARCH)
            population.append(individual)
        
        best_result = None
        
        for generation in range(min(generations, self.max_iterations)):
            if self._stop_event.is_set():
                break
            
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                score = self._evaluate_objective(individual)
                fitness_scores.append(-score)  # Negative because we want to maximize fitness
            
            # Create results
            for i, (individual, fitness) in enumerate(zip(population, fitness_scores)):
                result = TuningResult(
                    parameters=individual.copy(),
                    objective_value=-fitness,
                    metrics={},
                    iteration=generation * population_size + i
                )
                self.results.append(result)
                
                if best_result is None or result.is_better_than(best_result, minimize=True):
                    best_result = result
                    self.best_result = result
            
            # Selection (tournament selection)
            new_population = []
            for _ in range(population_size):
                tournament = random.sample(list(zip(population, fitness_scores)), 3)
                winner = max(tournament, key=lambda x: x[1])[0]
                new_population.append(winner.copy())
            
            # Crossover
            for i in range(0, population_size - 1, 2):
                if random.random() < crossover_rate:
                    parent1, parent2 = new_population[i], new_population[i + 1]
                    child1, child2 = self._crossover(parent1, parent2)
                    new_population[i], new_population[i + 1] = child1, child2
            
            # Mutation
            for individual in new_population:
                if random.random() < mutation_rate:
                    self._mutate(individual)
            
            population = new_population
            self.current_iteration = generation
            
            logger.debug(LogChannel.NEURAL, f"Generation {generation} complete",
                        best_fitness=-max(fitness_scores))
        
        return best_result
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Perform crossover between two individuals"""
        child1, child2 = parent1.copy(), parent2.copy()
        
        for param_name in self.parameters:
            if random.random() < 0.5:
                child1[param_name], child2[param_name] = child2[param_name], child1[param_name]
        
        return child1, child2
    
    def _mutate(self, individual: Dict[str, Any]):
        """Mutate an individual"""
        for param_name, param in self.parameters.items():
            if random.random() < 0.1:  # 10% chance to mutate each parameter
                if param.type == ParameterType.CONTINUOUS:
                    # Gaussian mutation
                    std = (param.max_value - param.min_value) * 0.1
                    new_value = individual[param_name] + random.gauss(0, std)
                    new_value = max(param.min_value, min(param.max_value, new_value))
                    individual[param_name] = new_value
                else:
                    # Random mutation
                    individual[param_name] = param.sample(TuningStrategy.RANDOM_SEARCH)
    
    def _tune_adaptive(self, **kwargs) -> TuningResult:
        """Adaptive tuning that switches strategies based on performance"""
        strategies = [
            TuningStrategy.RANDOM_SEARCH,
            TuningStrategy.BAYESIAN,
            TuningStrategy.GENETIC
        ]
        
        strategy_performance = {s: [] for s in strategies}
        current_strategy_idx = 0
        switch_interval = 20
        
        best_result = None
        
        for i in range(self.max_iterations):
            if self._stop_event.is_set():
                break
            
            # Get current strategy
            current_strategy = strategies[current_strategy_idx]
            
            # Run one iteration of current strategy
            if current_strategy == TuningStrategy.RANDOM_SEARCH:
                result = self._tune_random_search(n_samples=1)
            elif current_strategy == TuningStrategy.BAYESIAN:
                result = self._tune_bayesian(n_trials=1)
            elif current_strategy == TuningStrategy.GENETIC:
                result = self._tune_genetic(population_size=10, generations=1)
            
            if result:
                strategy_performance[current_strategy].append(result.objective_value)
                
                if best_result is None or result.is_better_than(best_result, minimize=True):
                    best_result = result
                    self.best_result = result
            
            # Switch strategy every switch_interval iterations
            if (i + 1) % switch_interval == 0:
                # Calculate average performance for each strategy
                avg_performance = {}
                for s, scores in strategy_performance.items():
                    if scores:
                        avg_performance[s] = np.mean(scores[-10:])  # Last 10 scores
                    else:
                        avg_performance[s] = float('inf')
                
                # Select best performing strategy
                best_strategy = min(avg_performance, key=avg_performance.get)
                current_strategy_idx = strategies.index(best_strategy)
                
                logger.info(LogChannel.NEURAL, f"Switching to {best_strategy.value}",
                           performance=avg_performance)
            
            self.current_iteration = i
        
        return best_result
    
    def apply_best_parameters(self):
        """Apply the best found parameters to all nodes"""
        if not self.best_result:
            logger.warning(LogChannel.NEURAL, "No best parameters to apply")
            return
        
        with self._lock:
            for param_name, value in self.best_result.parameters.items():
                if '.' in param_name:
                    node_id, param_key = param_name.rsplit('.', 1)
                    if node_id in self.nodes:
                        self.nodes[node_id].tune_parameter(param_key, value)
                        logger.info(LogChannel.NEURAL, f"Applied best parameter: {param_name} = {value}")
    
    def get_parameter_importance(self) -> Dict[str, float]:
        """Analyze parameter importance based on tuning results"""
        if len(self.results) < 10:
            return {}
        
        importance = {}
        
        # Calculate correlation between each parameter and objective
        for param_name in self.parameters:
            values = []
            objectives = []
            
            for result in self.results:
                if param_name in result.parameters:
                    param_value = result.parameters[param_name]
                    
                    # Convert categorical to numeric
                    if isinstance(param_value, (bool, str)):
                        param_value = hash(str(param_value)) % 1000
                    
                    values.append(param_value)
                    objectives.append(result.objective_value)
            
            if len(values) > 1:
                # Calculate correlation
                correlation = abs(np.corrcoef(values, objectives)[0, 1])
                importance[param_name] = correlation
        
        # Normalize
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}
        
        return importance
    
    def save_results(self, filepath: str):
        """Save tuning results to file"""
        data = {
            'best_result': {
                'parameters': self.best_result.parameters,
                'objective_value': self.best_result.objective_value,
                'metrics': self.best_result.metrics,
                'iteration': self.best_result.iteration
            } if self.best_result else None,
            'all_results': [{
                'parameters': r.parameters,
                'objective_value': r.objective_value,
                'metrics': r.metrics,
                'iteration': r.iteration,
                'timestamp': r.timestamp
            } for r in self.results],
            'parameter_importance': self.get_parameter_importance()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(LogChannel.NEURAL, f"Tuning results saved to {filepath}")
    
    def stop(self):
        """Stop the tuning process"""
        self._stop_event.set()
        self.executor.shutdown(wait=True)


# Example objective functions
def create_performance_objective(test_data: Any) -> TuningObjective:
    """Create an objective that maximizes performance"""
    
    def performance_objective(nodes: Dict[str, BaseNode], params: Dict[str, Any]) -> float:
        # Run test data through nodes and measure performance
        total_time = 0.0
        error_count = 0
        
        for node in nodes.values():
            metrics = node.get_metrics()
            total_time += metrics['metrics']['average_processing_time']
            error_count += metrics['metrics']['failed_calls']
        
        # Combine time and errors into single objective
        return total_time + (error_count * 10.0)  # Heavily penalize errors
    
    return TuningObjective(
        name="performance",
        function=performance_objective,
        minimize=True,
        weight=1.0
    )


def create_accuracy_objective(validation_data: Any, expected_outputs: Any) -> TuningObjective:
    """Create an objective that maximizes accuracy"""
    
    def accuracy_objective(nodes: Dict[str, BaseNode], params: Dict[str, Any]) -> float:
        # This would run validation data through nodes and compare outputs
        # For now, return a placeholder
        return random.random()  # Replace with actual accuracy calculation
    
    return TuningObjective(
        name="accuracy",
        function=accuracy_objective,
        minimize=False,  # We want to maximize accuracy
        weight=2.0  # Higher weight for accuracy
    )


if __name__ == "__main__":
    # Demo the parameter tuner
    print("🎯 Parameter Tuner Demo")
    print("=" * 50)
    
    # Create some example nodes with tunable parameters
    from .node_base import AtomicNode, NodeMetadata
    
    def process_data(x, threshold=0.5, scale=1.0):
        return x * scale if x > threshold else 0
    
    # Create node
    node = AtomicNode(process_data, NodeMetadata(name="processor"))
    node.initialize()
    
    # Register tunable parameters
    node.register_tunable_parameter("threshold", float, 0.0, 1.0, "Detection threshold")
    node.register_tunable_parameter("scale", float, 0.1, 10.0, "Output scale factor")
    
    # Create tuner
    tuner = ParameterTuner()
    tuner.register_node(node)
    
    # Add objectives
    tuner.add_objective(create_performance_objective(None))
    
    print(f"\nRegistered {len(tuner.parameters)} tunable parameters:")
    for name, param in tuner.parameters.items():
        print(f"  - {name}: {param.type.value} [{param.min_value}, {param.max_value}]")
    
    # Run tuning
    print("\nRunning parameter tuning...")
    best_result = tuner.tune(TuningStrategy.RANDOM_SEARCH, max_iterations=20)
    
    print(f"\nBest parameters found:")
    for param, value in best_result.parameters.items():
        print(f"  - {param}: {value}")
    print(f"  Objective value: {best_result.objective_value:.4f}")
    
    # Apply best parameters
    tuner.apply_best_parameters()
    
    # Save results
    tuner.save_results("tuning_results.json")
    
    print("\n✅ Parameter tuning demo complete!")