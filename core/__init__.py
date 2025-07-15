"""
Core components of the Neural Network Integration System
"""

from .node_base import (
    BaseNode, AtomicNode, CompositeNode, 
    NodeMetadata, NodeInput, NodeOutput,
    NodeState, NodeType, NodeMetrics
)

from .synapse import (
    BaseSynapse, SynchronousSynapse, AsynchronousSynapse,
    StreamingSynapse, BroadcastSynapse, SynapticRouter,
    ConnectionType, ConnectionState, Message
)

from .logger import (
    logger, LogLevel, LogChannel, LogEntry,
    log_info, log_debug, log_warning, log_error,
    log_success, log_critical, TimedOperation
)

from .config_manager import (
    config_manager, ConfigurationManager,
    ConfigType, ConfigLevel, ConfigOption, ConfigProfile
)

from .dependency_manager import (
    DependencyManager, interactive_setup as dependency_setup
)

from .module_analyzer import (
    ModuleAnalyzer, NodeFactory,
    FunctionInfo, ClassInfo, ModuleInfo
)

from .parameter_tuner import (
    ParameterTuner, TuningStrategy, ParameterType,
    TunableParameter, TuningResult, TuningObjective,
    create_performance_objective, create_accuracy_objective
)

__all__ = [
    # Node base
    "BaseNode", "AtomicNode", "CompositeNode",
    "NodeMetadata", "NodeInput", "NodeOutput",
    "NodeState", "NodeType", "NodeMetrics",
    
    # Synapse
    "BaseSynapse", "SynchronousSynapse", "AsynchronousSynapse",
    "StreamingSynapse", "BroadcastSynapse", "SynapticRouter",
    "ConnectionType", "ConnectionState", "Message",
    
    # Logger
    "logger", "LogLevel", "LogChannel", "LogEntry",
    "log_info", "log_debug", "log_warning", "log_error",
    "log_success", "log_critical", "TimedOperation",
    
    # Config
    "config_manager", "ConfigurationManager",
    "ConfigType", "ConfigLevel", "ConfigOption", "ConfigProfile",
    
    # Dependencies
    "DependencyManager", "dependency_setup",
    
    # Module analyzer
    "ModuleAnalyzer", "NodeFactory",
    "FunctionInfo", "ClassInfo", "ModuleInfo",
    
    # Parameter tuner
    "ParameterTuner", "TuningStrategy", "ParameterType",
    "TunableParameter", "TuningResult", "TuningObjective",
    "create_performance_objective", "create_accuracy_objective"
]