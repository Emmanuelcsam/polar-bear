#!/usr/bin/env python3
"""
Hybrid Optimizer for Fiber Optics Neural Network
Combines Gradient Descent with Evolution Strategies
Optimizes both network weights and equation coefficients
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any
from datetime import datetime
import copy
from collections import deque
import multiprocessing as mp
from functools import partial

from fiber_config import get_config
from fiber_logger import get_logger
from fiber_advanced_optimizers import SAMWithLookahead


class EvolutionStrategy:
    """
    Evolution Strategy (ES) optimizer for equation coefficients
    Uses Natural Evolution Strategies with fitness shaping
    
    Mathematical formulation:
    θ_t+1 = θ_t + α * ∇_θ E_ε~N(0,σ²I)[F(θ + ε)]
    
    Where F is fitness function (similarity score)
    """
    
    def __init__(self, 
                 num_params: int,
                 population_size: int = 50,
                 sigma: float = 0.1,
                 learning_rate: float = 0.01,
                 elite_ratio: float = 0.2,
                 mutation_rate: float = 0.1):
        """
        Args:
            num_params: Number of parameters to optimize
            population_size: Size of population
            sigma: Standard deviation for Gaussian noise
            learning_rate: Learning rate for parameter updates
            elite_ratio: Fraction of population to keep as elite
            mutation_rate: Probability of random mutation
        """
        print(f"[{datetime.now()}] Initializing EvolutionStrategy")
        
        self.logger = get_logger("EvolutionStrategy")
        self.logger.log_class_init("EvolutionStrategy", 
                                 num_params=num_params,
                                 population_size=population_size)
        
        self.num_params = num_params
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.elite_size = int(population_size * elite_ratio)
        self.mutation_rate = mutation_rate
        
        # Initialize parameters
        self.theta = np.random.randn(num_params) * 0.1
        
        # History tracking
        self.fitness_history = deque(maxlen=100)
        self.parameter_history = deque(maxlen=20)
        self.best_fitness = -np.inf
        self.best_params = self.theta.copy()
        
        # Adaptive sigma
        self.sigma_adaptation_rate = 0.1
        self.target_success_rate = 0.2
        
        self.logger.info("EvolutionStrategy initialized")
    
    def ask(self) -> List[np.ndarray]:
        """
        Generate population of candidate solutions
        Returns perturbed parameters for evaluation
        """
        self.epsilons = []
        self.population = []
        
        for i in range(self.population_size):
            if i < self.elite_size and len(self.parameter_history) > 0:
                # Include elite members from previous generation
                epsilon = self.parameter_history[-1] - self.theta
            else:
                # Generate new perturbation
                epsilon = np.random.randn(self.num_params) * self.sigma
                
                # Apply mutation
                if np.random.rand() < self.mutation_rate:
                    mutation_idx = np.random.randint(0, self.num_params)
                    epsilon[mutation_idx] += np.random.randn() * self.sigma * 2
            
            self.epsilons.append(epsilon)
            candidate = self.theta + epsilon
            
            # Ensure parameters stay in reasonable range
            candidate = np.clip(candidate, -2.0, 2.0)
            self.population.append(candidate)
        
        return self.population
    
    def tell(self, fitness_values: List[float]):
        """
        Update parameters based on fitness evaluations
        Uses Natural Evolution Strategy gradient estimation
        
        Args:
            fitness_values: Fitness scores for each candidate
        """
        fitness_values = np.array(fitness_values)
        self.fitness_history.extend(fitness_values)
        
        # Track best
        best_idx = np.argmax(fitness_values)
        if fitness_values[best_idx] > self.best_fitness:
            self.best_fitness = fitness_values[best_idx]
            self.best_params = self.population[best_idx].copy()
            self.logger.info(f"New best fitness: {self.best_fitness:.4f}")
        
        # Fitness shaping (rank-based)
        ranks = np.argsort(np.argsort(-fitness_values))
        shaped_fitness = np.maximum(0, (self.population_size - ranks) / self.population_size)
        
        # Normalize
        shaped_fitness = (shaped_fitness - shaped_fitness.mean()) / (shaped_fitness.std() + 1e-8)
        
        # Compute natural gradient
        gradient = np.zeros(self.num_params)
        for i, (epsilon, fitness) in enumerate(zip(self.epsilons, shaped_fitness)):
            gradient += fitness * epsilon
        
        gradient /= (self.population_size * self.sigma)
        
        # Update parameters
        self.theta += self.learning_rate * gradient
        self.theta = np.clip(self.theta, -2.0, 2.0)
        
        # Store in history
        self.parameter_history.append(self.theta.copy())
        
        # Adapt sigma based on success rate
        success_rate = np.mean(fitness_values > np.median(self.fitness_history))
        if success_rate > self.target_success_rate:
            self.sigma *= (1 + self.sigma_adaptation_rate)
        else:
            self.sigma *= (1 - self.sigma_adaptation_rate)
        self.sigma = np.clip(self.sigma, 0.01, 1.0)
        
        # Log statistics
        self.logger.info(f"ES update - Mean fitness: {fitness_values.mean():.4f}, "
                        f"Best: {fitness_values.max():.4f}, "
                        f"Sigma: {self.sigma:.4f}")
    
    def get_parameters(self) -> np.ndarray:
        """Get current parameters"""
        return self.theta
    
    def get_best_parameters(self) -> np.ndarray:
        """Get best parameters found so far"""
        return self.best_params


class HybridOptimizer:
    """
    Hybrid optimizer combining gradient descent and evolution strategies
    - Gradient descent (SAM + Lookahead) for neural network weights
    - Evolution strategy for equation coefficients
    
    "Combine both for optimal results"
    """
    
    def __init__(self, 
                 model: nn.Module,
                 config: Dict,
                 fitness_fn: Optional[Callable] = None):
        """
        Args:
            model: Neural network model
            config: Configuration dictionary
            fitness_fn: Fitness function for evolution strategy
        """
        print(f"[{datetime.now()}] Initializing HybridOptimizer")
        
        self.logger = get_logger("HybridOptimizer")
        self.logger.log_class_init("HybridOptimizer")
        
        self.model = model
        self.config = config
        self.fitness_fn = fitness_fn
        
        # Gradient-based optimizer for network weights
        self.gradient_optimizer = SAMWithLookahead(
            model.parameters(),
            lr=config.get('LEARNING_RATE', 0.001),
            rho=config.get('SAM_RHO', 0.05),
            k=config.get('LOOKAHEAD_K', 5),
            alpha=config.get('LOOKAHEAD_ALPHA', 0.5)
        )
        
        # Evolution strategy for equation coefficients
        # "I=Ax1+Bx2+Cx3... =S(R)" - 5 coefficients
        self.evolution_optimizer = EvolutionStrategy(
            num_params=5,
            population_size=config.get('ES_POPULATION_SIZE', 20),
            sigma=config.get('ES_SIGMA', 0.1),
            learning_rate=config.get('ES_LEARNING_RATE', 0.01)
        )
        
        # Hybrid parameters
        self.es_interval = config.get('ES_INTERVAL', 100)  # Apply ES every N steps
        self.step_count = 0
        
        # Parallel evaluation for ES
        self.use_parallel = config.get('USE_PARALLEL_ES', True)
        if self.use_parallel:
            self.num_workers = min(mp.cpu_count(), 4)
        
        self.logger.info("HybridOptimizer initialized")
    
    def step(self, closure: Callable) -> float:
        """
        Perform optimization step
        
        Args:
            closure: Function that computes loss
            
        Returns:
            Loss value
        """
        self.step_count += 1
        
        # Check if it's time for evolution step
        if self.step_count % self.es_interval == 0 and self.fitness_fn is not None:
            self.logger.info(f"Performing evolution step at iteration {self.step_count}")
            self._evolution_step()
        
        # Regular gradient step
        loss = closure()
        
        # Gradient-based update for network weights
        self.gradient_optimizer.step(closure)
        
        return loss
    
    def _evolution_step(self):
        """
        Perform evolution strategy step for equation coefficients
        "Evolution step for global exploration"
        """
        # Get current equation coefficients
        if hasattr(self.model, 'get_equation_coefficients'):
            current_coeffs = self.model.get_equation_coefficients()
        else:
            current_coeffs = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        
        # Set ES parameters to current values
        self.evolution_optimizer.theta = current_coeffs
        
        # Generate population
        population = self.evolution_optimizer.ask()
        
        # Evaluate fitness for each candidate
        if self.use_parallel:
            fitness_values = self._parallel_evaluate(population)
        else:
            fitness_values = self._sequential_evaluate(population)
        
        # Update parameters
        self.evolution_optimizer.tell(fitness_values)
        
        # Apply best parameters to model
        best_coeffs = self.evolution_optimizer.get_best_parameters()
        if hasattr(self.model, 'set_equation_coefficients'):
            self.model.set_equation_coefficients(best_coeffs)
            
        # Log coefficient updates
        coeff_names = ['A', 'B', 'C', 'D', 'E']
        coeff_str = ", ".join([f"{name}: {val:.4f}" 
                              for name, val in zip(coeff_names, best_coeffs)])
        self.logger.info(f"Updated equation coefficients: {coeff_str}")
    
    def _sequential_evaluate(self, population: List[np.ndarray]) -> List[float]:
        """Evaluate population sequentially"""
        fitness_values = []
        
        for coeffs in population:
            # Set coefficients
            if hasattr(self.model, 'set_equation_coefficients'):
                self.model.set_equation_coefficients(coeffs)
            
            # Evaluate fitness
            with torch.no_grad():
                fitness = self.fitness_fn(self.model)
            
            fitness_values.append(fitness)
        
        return fitness_values
    
    def _parallel_evaluate(self, population: List[np.ndarray]) -> List[float]:
        """Evaluate population in parallel"""
        # Create a partial function with fixed model
        eval_fn = partial(self._evaluate_single, 
                         model_state=self.model.state_dict(),
                         fitness_fn=self.fitness_fn)
        
        # Use multiprocessing pool
        with mp.Pool(self.num_workers) as pool:
            fitness_values = pool.map(eval_fn, population)
        
        return fitness_values
    
    @staticmethod
    def _evaluate_single(coeffs: np.ndarray, 
                        model_state: Dict, 
                        fitness_fn: Callable) -> float:
        """Evaluate single set of coefficients (for parallel execution)"""
        # Note: This is a simplified version
        # In practice, you'd need to properly recreate the model
        return fitness_fn(coeffs)
    
    def state_dict(self) -> Dict:
        """Get optimizer state"""
        return {
            'gradient_optimizer': self.gradient_optimizer.state_dict(),
            'evolution_params': self.evolution_optimizer.get_parameters(),
            'evolution_best': self.evolution_optimizer.get_best_parameters(),
            'step_count': self.step_count
        }
    
    def load_state_dict(self, state_dict: Dict):
        """Load optimizer state"""
        self.gradient_optimizer.load_state_dict(state_dict['gradient_optimizer'])
        self.evolution_optimizer.theta = state_dict['evolution_params']
        self.evolution_optimizer.best_params = state_dict['evolution_best']
        self.step_count = state_dict['step_count']


class AdaptiveHybridOptimizer(HybridOptimizer):
    """
    Advanced hybrid optimizer with adaptive switching between strategies
    Automatically decides when to use gradient descent vs evolution
    """
    
    def __init__(self, model: nn.Module, config: Dict, fitness_fn: Optional[Callable] = None):
        """Initialize adaptive hybrid optimizer"""
        super().__init__(model, config, fitness_fn)
        
        self.logger = get_logger("AdaptiveHybridOptimizer")
        
        # Adaptive parameters
        self.loss_history = deque(maxlen=100)
        self.gradient_norm_history = deque(maxlen=100)
        self.improvement_threshold = 0.001
        self.stagnation_counter = 0
        self.max_stagnation = 20
        
        # Strategy weights (learned)
        self.strategy_weights = {
            'gradient': 0.8,
            'evolution': 0.2
        }
        
        self.logger.info("AdaptiveHybridOptimizer initialized")
    
    def step(self, closure: Callable) -> float:
        """
        Adaptive optimization step
        Dynamically chooses between gradient and evolution based on progress
        """
        self.step_count += 1
        
        # Compute loss and gradient norm
        loss = closure()
        grad_norm = self._compute_gradient_norm()
        
        # Update histories
        self.loss_history.append(loss.item())
        self.gradient_norm_history.append(grad_norm)
        
        # Check for stagnation
        if len(self.loss_history) > 10:
            recent_improvement = (self.loss_history[-10] - self.loss_history[-1]) / (abs(self.loss_history[-10]) + 1e-8)
            
            if recent_improvement < self.improvement_threshold:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
        
        # Adaptive strategy selection
        use_evolution = False
        
        if self.stagnation_counter > self.max_stagnation:
            # Force evolution when stuck
            use_evolution = True
            self.logger.info("Switching to evolution due to stagnation")
            self.stagnation_counter = 0
        elif grad_norm < 1e-4:
            # Use evolution when gradients are very small
            use_evolution = True
            self.logger.info("Using evolution due to small gradients")
        elif np.random.rand() < self.strategy_weights['evolution']:
            # Probabilistic selection
            use_evolution = True
        
        # Apply selected strategy
        if use_evolution and self.fitness_fn is not None:
            self._evolution_step()
            # Update strategy weights based on improvement
            self._update_strategy_weights('evolution', loss)
        else:
            # Gradient step
            self.gradient_optimizer.step(closure)
            self._update_strategy_weights('gradient', loss)
        
        return loss
    
    def _compute_gradient_norm(self) -> float:
        """Compute total gradient norm across all parameters"""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return np.sqrt(total_norm)
    
    def _update_strategy_weights(self, strategy: str, loss: torch.Tensor):
        """Update strategy weights based on performance"""
        if len(self.loss_history) < 2:
            return
        
        # Compute improvement
        improvement = (self.loss_history[-2] - loss.item()) / (abs(self.loss_history[-2]) + 1e-8)
        
        # Update weights using exponential moving average
        alpha = 0.1
        if improvement > 0:
            # Increase weight for successful strategy
            self.strategy_weights[strategy] = min(0.9, 
                self.strategy_weights[strategy] * (1 + alpha))
        else:
            # Decrease weight for unsuccessful strategy
            self.strategy_weights[strategy] = max(0.1,
                self.strategy_weights[strategy] * (1 - alpha))
        
        # Normalize weights
        total = sum(self.strategy_weights.values())
        for key in self.strategy_weights:
            self.strategy_weights[key] /= total


def create_hybrid_optimizer(model: nn.Module, 
                           config: Dict,
                           fitness_fn: Optional[Callable] = None,
                           adaptive: bool = True) -> HybridOptimizer:
    """
    Factory function to create hybrid optimizer
    
    Args:
        model: Neural network model
        config: Configuration dictionary
        fitness_fn: Fitness function for evolution strategy
        adaptive: Whether to use adaptive version
        
    Returns:
        HybridOptimizer instance
    """
    logger = get_logger("HybridOptimizerFactory")
    logger.log_function_entry("create_hybrid_optimizer")
    
    if adaptive:
        optimizer = AdaptiveHybridOptimizer(model, config, fitness_fn)
    else:
        optimizer = HybridOptimizer(model, config, fitness_fn)
    
    logger.info(f"Created {'Adaptive' if adaptive else 'Standard'} HybridOptimizer")
    logger.log_function_exit("create_hybrid_optimizer")
    
    return optimizer


# Test the hybrid optimizer
if __name__ == "__main__":
    print(f"[{datetime.now()}] Testing hybrid optimizer")
    
    # Create test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 10)
            self.coefficients = nn.Parameter(torch.ones(5))
        
        def forward(self, x):
            return self.fc(x)
        
        def get_equation_coefficients(self):
            return self.coefficients.detach().cpu().numpy()
        
        def set_equation_coefficients(self, coeffs):
            self.coefficients.data = torch.tensor(coeffs, dtype=torch.float32)
    
    model = TestModel()
    
    # Test fitness function
    def test_fitness(model):
        # Simple fitness: sum of coefficients (maximize)
        return model.get_equation_coefficients().sum()
    
    # Test standard hybrid optimizer
    print("\nTesting HybridOptimizer...")
    config = {
        'LEARNING_RATE': 0.001,
        'ES_POPULATION_SIZE': 10,
        'ES_INTERVAL': 5
    }
    optimizer = HybridOptimizer(model, config, test_fitness)
    
    # Test adaptive hybrid optimizer
    print("\nTesting AdaptiveHybridOptimizer...")
    adaptive_optimizer = AdaptiveHybridOptimizer(model, config, test_fitness)
    
    # Simulate training steps
    for i in range(20):
        def closure():
            loss = torch.sum(model.fc.weight ** 2)
            loss.backward()
            return loss
        
        loss = adaptive_optimizer.step(closure)
        if i % 5 == 0:
            print(f"Step {i}: Loss = {loss.item():.4f}, "
                  f"Coeffs = {model.get_equation_coefficients()}")
    
    print(f"[{datetime.now()}] Hybrid optimizer test completed")
    print(f"[{datetime.now()}] Next script: fiber_advanced_similarity.py")