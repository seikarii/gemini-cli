import asyncio

from crisalida_lib.ASTRAL_TOOLS.base import ToolCallResult
from crisalida_lib.ASTRAL_TOOLS.llm_management_tool import LLMManagementTool


class DummyLLMGateway:
    async def process_prompt(self, prompt: str) -> str:
        return f"Dummy response for: {prompt}"

    async def list_models(self):
        return [{"name": "dummy-model", "size": 100}]

    async def check_model_availability(self, model_name: str):
        return True

    async def generate_async(self, model_name: str, prompt: str):
        return f"Dummy generated text for {model_name}: {prompt}"

    async def enable_llm_mode(self):
        pass

    async def enable_offline_mode(self):
        pass

    @property
    def is_llm_mode_enabled(self):
        return True

    @property
    def health_monitor(self):
        class DummyHealthMonitor:
            async def check_all(self):
                return {
                    "individual_status": {},
                    "any_available": True,
                    "all_available": True,
                }

            def get_health_summary(self):  # Added this method
                return {"overall_health": "Good", "availability_rates": {}}

        return DummyHealthMonitor()


async def demonstrate_llm_management():
    """Demonstrates the LLMManagementTool's functionality."""
    print("--- Demonstrating LLMManagementTool ---")
    llm_gateway = DummyLLMGateway()
    tool = LLMManagementTool(llm_gateway=llm_gateway)

    # Example: Get the status of the LLM gateway
    status_result = await tool.execute(action="status")
    print(f"Status: {status_result.output}")

    return ToolCallResult(
        command="demo_llm_management",
        success=True,
        output="LLMManagementTool demo completed.",
        execution_time=0.1,
    )


if __name__ == "__main__":
    asyncio.run(demonstrate_llm_management())
