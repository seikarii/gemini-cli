# Tools Module

This directory contains definitions and implementations of various tools that the agent can use to interact with the environment, such as file system operations, shell command execution, and web interactions.

## Structure

-   **Tool Definitions**: Each file in the root of this directory defines one or more tools. Each tool inherits from the `BaseTool` class defined in `base.py`.
-   **Demos**: The `demos/` directory contains standalone scripts that demonstrate the functionality of each tool. To run a demo, execute the corresponding script in the `demos/` directory.
-   **Registration**: The `registration.py` file is responsible for registering all the tools in the `ToolRegistry`.
-   **Base**: The `base.py` file contains the `BaseTool` class, the `ToolRegistry` class, and other base components for the tool system.

## Core Components

*   `base.py`: Provides the foundational structure for all tool implementations.
*   `data_visualization.py`: Generates charts, tables, and advanced statistical analysis for structured data.
*   `default_tools.py`: Registers all essential tools (filesystem, memory, shell, web) in the `ToolRegistry`.
*   `dialectical_oracle_tool.py`: A tool for dialectical debates and advanced philosophical reasoning.
*   `external_api.py`: A generic tool for accessing external APIs with authentication, error handling, and advanced support for multiple formats.
*   `file_system.py`: Provides robust file reading, safe writing, directory and glob listing, regex-based content search, and advanced replacement.
*   `interactive_debugging.py`: An advanced tool for interactive and programmatic debugging of Python code.
*   `llm_management_tool.py`: Manages local LLM connectors (brain/cerebellum) for development and operation.
*   `memory_tools.py`: Provides enhanced memory recall with semantic search and advanced filtering capabilities.
*   `realtime_monitoring.py`: Allows subscribing to system events (file changes, logs, processes) and reacting in real-time.
*   `registration_manager.py`: Manages the registration of all Crisalida tools.
*   `registration.py`: Registers all default tools.
*   `semantic_search.py`: An advanced tool for semantic search and contextual analysis of Python codebases.
*   `shell_and_web.py`: Provides advanced tools for secure shell command execution and web processing.
*   `sub_llm_agent.py`: A tool for intelligent delegation to specialized LLM sub-models.
*   `subchat_tool.py`: An advanced asynchronous communication and feedback system.
*   `validation_tools.py`: Includes tools for structured linting with Ruff, test execution with Pytest, and type checking with MyPy.
*   `visual_ui_interaction.py`: Allows programmatic interaction with graphical interfaces (web, apps).

## Usage

To use a tool, you need to get it from the `ToolRegistry` and then call its `execute` method. For example:

```python
from crisalida_lib.tools.base import ToolRegistry

registry = ToolRegistry()
registry.auto_discover_tools()

result = await registry.execute_tool("my_tool", {"param1": "value1"})
```

## Demos

To run the demo for a specific tool, you can run the corresponding script in the `demos/` directory. For example, to run the demo for the `shell_and_web` tools, you can run the following command:

```bash
python -m crisalida_lib.tools.demos.shell_and_web_demos
```