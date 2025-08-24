# Monitoring Module

This directory contains monitoring systems. Currently, it includes the `performance_monitor` to track the performance, overall health, and stability of the simulation.

## Core Components

*   `performance_monitor.py`: A real-time performance monitor for the GameEngine, simulators, and workers. It includes FPS measurement, frame timings, worker statistics, and advanced profiling.
*   `predictive_monitoring.py`: An advanced predictive monitor for Crisalida. It monitors key simulation metrics (memory, entities, CPU) and generates predictive alerts.
*   `performance_decorators.py`: Decorators for measuring and optimizing performance. It includes decorators for measuring execution time, CPU usage, memory, and advanced profiling.