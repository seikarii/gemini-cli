#!/usr/bin/env python3
"""
Data Visualization and Analysis Tool for Gemini CLI
===================================================
Genera gráficos, tablas y análisis estadístico avanzado para datos estructurados.

Características:
- Generación de gráficos (línea, barra, dispersión, histograma, pastel, caja, mapa de calor)
- Análisis estadístico y correlacional
- Formateo de tablas y exportación a CSV/JSON/HTML
- Procesamiento eficiente de datos grandes
- Exportación a PNG, SVG, HTML, CSV
- Integración con pandas y matplotlib si están disponibles
- Soporte para análisis de tendencias y series temporales
"""

import asyncio
import json
import logging
import tempfile
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from crisalida_lib.ASTRAL_TOOLS.base import BaseTool, ToolCallResult

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import matplotlib.style as mplstyle

    mplstyle.use("fast")
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available - chart generation will be limited")

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("pandas not available - advanced data analysis will be limited")


class DataVisualizationParameters(BaseModel):
    action: Literal[
        "create_chart",
        "analyze_data",
        "format_table",
        "load_data",
        "statistics",
        "correlation_analysis",
        "trend_analysis",
        "export_csv",
    ] = Field(..., description="Type of visualization/analysis to perform")

    data: list[list[Any]] | dict[str, list[Any]] | str = Field(
        ..., description="Input data as list of lists, dict of lists, or JSON string"
    )
    chart_type: (
        Literal["line", "bar", "scatter", "histogram", "pie", "box", "heatmap"] | None
    ) = Field(None, description="Type of chart to generate")
    title: str | None = Field(None, description="Chart/table title")
    x_label: str | None = Field(None, description="X-axis label")
    y_label: str | None = Field(None, description="Y-axis label")
    columns: list[str] | None = Field(None, description="Column names for data")
    target_column: str | None = Field(None, description="Target column for analysis")
    output_format: Literal["png", "svg", "html", "text", "csv"] = Field(
        "text", description="Output format"
    )
    output_path: str | None = Field(None, description="Output file path")
    width: int = Field(10, description="Chart width in inches")
    height: int = Field(6, description="Chart height in inches")
    style: str = Field("default", description="Chart style/theme")
    preview_rows: int = Field(10, description="Number of rows to preview in analysis")


class DataVisualizationTool(BaseTool):
    """Herramienta avanzada para visualización y análisis estadístico de datos."""

    def _get_name(self) -> str:
        return "data_visualization"

    def _get_description(self) -> str:
        return (
            "Genera gráficos, tablas y análisis estadístico avanzado sobre datos estructurados. "
            "Soporta exportación a PNG, SVG, HTML, CSV y análisis de tendencias/correlaciones."
        )

    def _get_category(self) -> str:
        return "data_analysis"

    def _get_pydantic_schema(self) -> type[BaseModel]:
        return DataVisualizationParameters

    def _parse_data(
        self, data: list[list[Any]] | dict[str, list[Any]] | str
    ) -> dict[str, Any]:
        """Convierte datos de entrada en formato dict estándar."""
        if isinstance(data, str):
            try:
                parsed = json.loads(data)
                return self._parse_data(parsed)
            except json.JSONDecodeError as e:
                raise ValueError("Invalid JSON string provided") from e
        if isinstance(data, list):
            if not data:
                raise ValueError("Empty data provided")
            if all(isinstance(item, str) for item in data[0]):
                headers = data[0]
                rows = data[1:]
            else:
                headers = [f"col_{i}" for i in range(len(data[0]))]
                rows = data
            result = {}
            for i, header in enumerate(headers):
                result[header] = [row[i] if i < len(row) else None for row in rows]
            return result
        if isinstance(data, dict):
            return data
        raise ValueError(f"Unsupported data type: {type(data)}")

    def _calculate_statistics(self, data: dict[str, list[Any]]) -> dict[str, Any]:
        """Calcula estadísticas básicas para columnas numéricas."""
        stats = {}
        for column, values in data.items():
            numeric_values = []
            for val in values:
                try:
                    if val is not None:
                        numeric_values.append(float(val))
                except (ValueError, TypeError):
                    continue
            if numeric_values:
                np_arr = np.array(numeric_values)
                stats[column] = {
                    "count": len(numeric_values),
                    "mean": float(np.mean(np_arr)),
                    "median": float(np.median(np_arr)),
                    "std": float(np.std(np_arr)),
                    "min": float(np.min(np_arr)),
                    "max": float(np.max(np_arr)),
                    "q25": float(np.percentile(np_arr, 25)),
                    "q75": float(np.percentile(np_arr, 75)),
                }
        return stats

    def _create_chart(
        self, data: dict[str, list[Any]], params: DataVisualizationParameters
    ) -> str:
        """Genera un gráfico usando matplotlib."""
        if not MATPLOTLIB_AVAILABLE:
            return "matplotlib not available - cannot generate charts"
        plt.figure(figsize=(params.width, params.height))
        chart_type = params.chart_type or "line"
        numeric_cols = []
        for col, values in data.items():
            try:
                numeric_count = sum(
                    1
                    for v in values
                    if v is not None
                    and str(v).replace(".", "").replace("-", "").isdigit()
                )
                if numeric_count > len(values) * 0.5:
                    numeric_cols.append(col)
            except (ValueError, TypeError):
                continue
        if len(numeric_cols) < 1:
            return "No numeric columns found for plotting"
        try:
            if chart_type == "line":
                if len(numeric_cols) >= 2:
                    x_data = [float(v) for v in data[numeric_cols[0]] if v is not None]
                    y_data = [float(v) for v in data[numeric_cols[1]] if v is not None]
                    plt.plot(x_data[: len(y_data)], y_data)
                else:
                    y_data = [float(v) for v in data[numeric_cols[0]] if v is not None]
                    plt.plot(y_data)
            elif chart_type == "bar":
                if len(numeric_cols) >= 2:
                    labels = [str(v) for v in data[numeric_cols[0]] if v is not None]
                    values = [float(v) for v in data[numeric_cols[1]] if v is not None]
                    plt.bar(labels[: len(values)], values)
                else:
                    values = [float(v) for v in data[numeric_cols[0]] if v is not None]
                    plt.bar(range(len(values)), values)
            elif chart_type == "scatter":
                if len(numeric_cols) >= 2:
                    x_data = [float(v) for v in data[numeric_cols[0]] if v is not None]
                    y_data = [float(v) for v in data[numeric_cols[1]] if v is not None]
                    plt.scatter(x_data[: len(y_data)], y_data)
                else:
                    return "Scatter plot requires at least 2 numeric columns"
            elif chart_type == "histogram":
                values = [float(v) for v in data[numeric_cols[0]] if v is not None]
                plt.hist(values, bins=20)
            elif chart_type == "pie":
                if len(data) >= 2:
                    cols = list(data.keys())
                    labels = [str(v) for v in data[cols[0]] if v is not None]
                    try:
                        values = [float(v) for v in data[cols[1]] if v is not None]
                        plt.pie(values[: len(labels)], labels=labels, autopct="%1.1f%%")
                    except (ValueError, TypeError):
                        return "Pie chart requires numeric values in second column"
                else:
                    return "Pie chart requires at least 2 columns"
            elif chart_type == "box":
                box_data = [
                    [float(v) for v in data[col] if v is not None]
                    for col in numeric_cols
                ]
                plt.boxplot(box_data, label=numeric_cols)
            elif chart_type == "heatmap":
                if len(numeric_cols) >= 2:
                    arr = np.array(
                        [
                            [float(v) if v is not None else np.nan for v in data[col]]
                            for col in numeric_cols
                        ]
                    )
                    plt.imshow(arr, aspect="auto", cmap="viridis")
                    plt.colorbar()
                else:
                    return "Heatmap requires at least 2 numeric columns"
            if params.title:
                plt.title(params.title)
            if params.x_label:
                plt.xlabel(params.x_label)
            if params.y_label:
                plt.ylabel(params.y_label)
            if params.output_path:
                plt.savefig(
                    params.output_path,
                    format=params.output_format,
                    dpi=150,
                    bbox_inches="tight",
                )
                result = f"Chart saved to {params.output_path}"
            else:
                with tempfile.NamedTemporaryFile(
                    suffix=f".{params.output_format}", delete=False
                ) as tmp:
                    plt.savefig(
                        tmp.name,
                        format=params.output_format,
                        dpi=150,
                        bbox_inches="tight",
                    )
                    result = f"Chart saved to {tmp.name}"
            plt.close()
            return result
        except Exception as e:
            plt.close()
            return f"Error creating chart: {str(e)}"

    def _format_table(
        self,
        data: dict[str, list[Any]],
        title: str | None = None,
        preview_rows: int = 10,
    ) -> str:
        """Formatea datos como tabla de texto."""
        if not data:
            return "No data to display"
        columns = list(data.keys())
        max_rows = min(max(len(values) for values in data.values()), preview_rows)
        col_widths = {}
        for col in columns:
            max_width = len(col)
            for val in data[col][:max_rows]:
                max_width = max(max_width, len(str(val)) if val is not None else 4)
            col_widths[col] = min(max_width, 20)
        result = []
        if title:
            result.append(f"\n{title}")
            result.append("=" * len(title))
        header = " | ".join(col.ljust(col_widths[col]) for col in columns)
        result.append(header)
        result.append("-" * len(header))
        for i in range(max_rows):
            row_values = []
            for col in columns:
                if i < len(data[col]):
                    val = data[col][i]
                    val_str = str(val) if val is not None else "None"
                    if len(val_str) > col_widths[col]:
                        val_str = val_str[: col_widths[col] - 3] + "..."
                    row_values.append(val_str.ljust(col_widths[col]))
                else:
                    row_values.append("".ljust(col_widths[col]))
            result.append(" | ".join(row_values))
        return "\n".join(result)

    def _export_csv(
        self, data: dict[str, list[Any]], output_path: str | None = None
    ) -> str:
        """Exporta datos a CSV usando pandas si está disponible."""
        if not PANDAS_AVAILABLE:
            return "pandas not available - cannot export CSV"
        try:
            df = pd.DataFrame(data)
            if output_path:
                df.to_csv(output_path, index=False)
                return f"CSV exported to {output_path}"
            else:
                with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                    df.to_csv(tmp.name, index=False)
                    return f"CSV exported to {tmp.name}"
        except Exception as e:
            return f"Error exporting CSV: {str(e)}"

    async def execute(self, **kwargs) -> ToolCallResult:
        start_time = asyncio.get_event_loop().time()
        try:
            params = DataVisualizationParameters(**kwargs)
            parsed_data = self._parse_data(params.data)
            if params.action == "load_data":
                output = f"Data loaded successfully with columns: {list(parsed_data.keys())}\n"
                output += f"Rows: {max(len(values) for values in parsed_data.values())}"
            elif params.action == "statistics":
                stats = self._calculate_statistics(parsed_data)
                output = "Statistical Summary:\n\n"
                for col, col_stats in stats.items():
                    output += f"{col}:\n"
                    for stat_name, value in col_stats.items():
                        output += f"  {stat_name}: {value:.4f}\n"
                    output += "\n"
            elif params.action == "format_table":
                output = self._format_table(
                    parsed_data, params.title, params.preview_rows
                )
            elif params.action == "create_chart":
                if not params.chart_type:
                    return ToolCallResult(
                        command="data_visualization",
                        success=False,
                        output="",
                        error_message="chart_type is required for create_chart action",
                        execution_time=asyncio.get_event_loop().time() - start_time,
                    )
                output = self._create_chart(parsed_data, params)
            elif params.action == "analyze_data":
                stats = self._calculate_statistics(parsed_data)
                preview_data = {
                    col: values[: params.preview_rows]
                    for col, values in parsed_data.items()
                }
                output = "Data Analysis Report:\n\n"
                output += f"Dataset Shape: {len(list(parsed_data.keys()))} columns, {max(len(v) for v in parsed_data.values())} rows\n\n"
                output += "Statistical Summary:\n"
                for col, col_stats in stats.items():
                    output += f"\n{col}: mean={col_stats['mean']:.2f}, std={col_stats['std']:.2f}, range=[{col_stats['min']:.2f}, {col_stats['max']:.2f}]"
                output += "\n\n"
                output += self._format_table(
                    preview_data, "Data Preview", params.preview_rows
                )
            elif params.action == "correlation_analysis":
                numeric_data = {}
                for col, values in parsed_data.items():
                    numeric_values = []
                    for val in values:
                        try:
                            if val is not None:
                                numeric_values.append(float(val))
                            else:
                                numeric_values.append(np.nan)
                        except (ValueError, TypeError):
                            numeric_values.append(np.nan)
                    if len([v for v in numeric_values if not np.isnan(v)]) > 0:
                        numeric_data[col] = numeric_values
                if len(numeric_data) < 2:
                    output = "Need at least 2 numeric columns for correlation analysis"
                else:
                    output = "Correlation Analysis:\n\n"
                    cols = list(numeric_data.keys())
                    for i, col1 in enumerate(cols):
                        for col2 in cols[i + 1 :]:
                            arr1 = np.array(numeric_data[col1])
                            arr2 = np.array(numeric_data[col2])
                            mask = ~(np.isnan(arr1) | np.isnan(arr2))
                            if np.sum(mask) > 1:
                                corr = np.corrcoef(arr1[mask], arr2[mask])[0, 1]
                                output += f"{col1} vs {col2}: {corr:.4f}\n"
            elif params.action == "trend_analysis":
                output = "Trend Analysis:\n\n"
                for col, values in parsed_data.items():
                    numeric_values = []
                    for val in values:
                        try:
                            if val is not None:
                                numeric_values.append(float(val))
                        except (ValueError, TypeError):
                            continue
                    if len(numeric_values) > 2:
                        x = np.arange(len(numeric_values))
                        y = np.array(numeric_values)
                        slope = np.polyfit(x, y, 1)[0]
                        trend_desc = (
                            "increasing"
                            if slope > 0
                            else "decreasing" if slope < 0 else "stable"
                        )
                        output += f"{col}: {trend_desc} trend (slope: {slope:.4f})\n"
            elif params.action == "export_csv":
                output = self._export_csv(parsed_data, params.output_path)
            else:
                return ToolCallResult(
                    command="data_visualization",
                    success=False,
                    output="",
                    error_message=f"Unknown action: {params.action}",
                )
            execution_time = asyncio.get_event_loop().time() - start_time
            return ToolCallResult(
                command=f"data_visualization({params.action})",
                success=True,
                output=output,
                error_message=None,
                execution_time=execution_time,
                metadata={
                    "action": params.action,
                    "data_shape": f"{len(parsed_data)} columns",
                    "chart_type": params.chart_type,
                },
            )
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"Data visualization error: {e}")
            return ToolCallResult(
                command="data_visualization",
                success=False,
                output="",
                error_message=str(e),
                execution_time=execution_time,
            )

    async def demo(self):
        """Demonstrate DataVisualizationTool functionality."""
        print("📊 DATA VISUALIZATION TOOL DEMO")
        print("=" * 50)

        print("This tool creates charts and visualizations from data")
        print("Supports: tables, line charts, bar charts, pie charts, histograms")

        # Simple demo without creating actual charts
        print("✅ Tool initialized and ready for data visualization")
        print("📈 Use with data (JSON/CSV) and chart_type parameters")

        return ToolCallResult(
            command="data_visualization_demo",
            success=True,
            output="DataVisualizationTool demo completed - tool ready for use",
            execution_time=0.1,
            error_message=None,
        )
