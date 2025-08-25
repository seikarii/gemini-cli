import logging

from crisalida_lib.ASTRAL_TOOLS.data_visualization import DataVisualizationTool

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import matplotlib.style as mplstyle

    mplstyle.use("fast")
    MATPLOTLIB_AVAILABLE = bool(plt)  # Use plt to avoid unused import warning
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available - chart generation will be limited")

try:
    import pandas as pd

    PANDAS_AVAILABLE = bool(pd)  # Use pd to avoid unused import warning
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("pandas not available - advanced data analysis will be limited")


async def demo_data_visualization():
    """Demo the data visualization tool"""
    print("📊 DATA VISUALIZATION TOOL DEMO")
    print("=" * 40)
    tool = DataVisualizationTool()
    sample_data = {
        "month": ["Jan", "Feb", "Mar", "Apr", "May"],
        "sales": [100, 120, 140, 110, 160],
        "profit": [20, 25, 30, 22, 35],
    }
    result = await tool.execute(action="statistics", data=sample_data)
    print(f"Statistics: {result.success}")
    print(result.output[:200] + "..." if len(result.output) > 200 else result.output)
    result = await tool.execute(
        action="format_table", data=sample_data, title="Monthly Sales Data"
    )
    print(f"\nTable: {result.success}")
    print(result.output)
    print("\n✅ Data visualization demo completed!")
