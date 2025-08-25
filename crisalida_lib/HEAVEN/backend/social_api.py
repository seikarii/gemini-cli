"""
Janus Social API - FastAPI app exposing social network data from the Metacosmos simulation.
Provides endpoints for network statistics, entity profiles, group dynamics, and cultural artifacts.
Integrates with JanusCore, EmpatheticBondingSystem, and SocialDynamics for real-time simulation data.
"""

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# Define data models for API requests and responses
class EntityProfile(BaseModel):
    entity_id: str
    bonds_count: int
    partners: list[dict]
    average_bond_strength: float
    reproduction_attempts: int
    altruistic_acts: int
    qualia_state: dict[str, float] = {}
    last_interaction: float = 0.0


class Group(BaseModel):
    group_id: str
    members: list[str]
    group_qualia: dict[str, float]
    cultural_artifacts: list[str]
    group_cohesion: float = 0.0
    creation_time: float = 0.0


class CreateGroupRequest(BaseModel):
    group_id: str
    member_ids: list[str]


def create_social_api(janus_core_instance):
    from backend.janus_core import JanusCore

    app = FastAPI(
        title="Janus Social API",
        description="API para datos sociales y culturales del Metacosmos Janus (EVA)",
        version="1.1.0",
    )

    # Dependency to get the JanusCore instance
    def get_janus_core():
        return janus_core_instance

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allows all origins
        allow_credentials=True,
        allow_methods=["*"],  # Allows all methods
        allow_headers=["*"],  # Allows all headers
    )

    @app.get("/", tags=["root"])
    def read_root():
        return {
            "message": "Welcome to the Janus Social API (EVA)",
            "endpoints": [
                "/network_state",
                "/entity/{entity_id}",
                "/groups",
                "/groups/{group_id}",
                "/eva/entity/{entity_id}/experiences",
                "/eva/group/{group_id}/experiences",
                "/eva/phase",
            ],
        }

    @app.get("/network_state", tags=["network"])
    def get_network_state(janus_core: JanusCore = Depends(get_janus_core)):
        state = janus_core.empathetic_bonding_system.get_network_state()
        state["group_count"] = len(janus_core.social_dynamics.get_all_groups())
        state["cultural_artifact_count"] = (
            janus_core.social_dynamics.get_cultural_artifact_count()
        )
        # EVA: Añadir métricas agregadas de memoria viviente
        if hasattr(janus_core, "eva_metrics_calculator"):
            state["eva_metrics"] = (
                janus_core.eva_metrics_calculator.calculate_eva_aggregated_metrics(
                    phase=getattr(janus_core, "eva_phase", "default")
                )
            )
        return state

    @app.get("/entity/{entity_id}", response_model=EntityProfile, tags=["entity"])
    def get_entity_profile(
        entity_id: str, janus_core: JanusCore = Depends(get_janus_core)
    ):
        profile = janus_core.empathetic_bonding_system.get_entity_social_profile(
            entity_id
        )
        if not profile:
            raise HTTPException(status_code=404, detail="Entity not found")
        qualia = janus_core.storage_manager.get_entity_qualia_state(entity_id)
        last_interaction = janus_core.storage_manager.get_last_interaction_time(
            entity_id
        )
        profile["qualia_state"] = qualia or {}
        profile["last_interaction"] = last_interaction or 0.0
        # EVA: Añadir experiencias vivientes y fase actual
        eva_experiences = []
        eva_phase = getattr(janus_core, "eva_phase", "default")
        eva_memory_store = getattr(janus_core, "eva_memory_store", {})
        for exp_id, reality_bytecode in eva_memory_store.items():
            if hasattr(reality_bytecode, "to_dict"):
                exp_dict = reality_bytecode.to_dict()
            else:
                exp_dict = {
                    "bytecode_id": getattr(reality_bytecode, "bytecode_id", exp_id),
                    "phase": getattr(reality_bytecode, "phase", eva_phase),
                    "qualia_state": getattr(reality_bytecode, "qualia_state", None),
                    "timestamp": getattr(reality_bytecode, "timestamp", None),
                }
            if exp_dict.get("experience", {}).get("entity_id") == entity_id:
                eva_experiences.append(exp_dict)
        profile["eva_experiences"] = eva_experiences
        profile["eva_phase"] = eva_phase
        return profile

    @app.get("/eva/entity/{entity_id}/experiences", tags=["eva"])
    def get_entity_eva_experiences(
        entity_id: str, janus_core: JanusCore = Depends(get_janus_core)
    ):
        eva_memory_store = getattr(janus_core, "eva_memory_store", {})
        eva_phase = getattr(janus_core, "eva_phase", "default")
        experiences = []
        for _exp_id, reality_bytecode in eva_memory_store.items():
            exp_dict = (
                reality_bytecode.to_dict()
                if hasattr(reality_bytecode, "to_dict")
                else {}
            )
            if exp_dict.get("experience", {}).get("entity_id") == entity_id:
                experiences.append(exp_dict)
        return {
            "entity_id": entity_id,
            "eva_phase": eva_phase,
            "eva_experiences": experiences,
        }

    @app.get("/groups/{group_id}", response_model=Group, tags=["groups"])
    def get_group(group_id: str, janus_core: JanusCore = Depends(get_janus_core)):
        group = janus_core.social_dynamics.get_group(group_id)
        if not group:
            raise HTTPException(status_code=404, detail="Group not found")
        group["group_cohesion"] = janus_core.social_dynamics.get_group_cohesion(
            group_id
        )
        group["creation_time"] = janus_core.social_dynamics.get_group_creation_time(
            group_id
        )
        # EVA: Añadir experiencias vivientes asociadas al grupo
        eva_experiences = []
        eva_phase = getattr(janus_core, "eva_phase", "default")
        eva_memory_store = getattr(janus_core, "eva_memory_store", {})
        for _exp_id, reality_bytecode in eva_memory_store.items():
            exp_dict = (
                reality_bytecode.to_dict()
                if hasattr(reality_bytecode, "to_dict")
                else {}
            )
            if exp_dict.get("experience", {}).get("group_id") == group_id:
                eva_experiences.append(exp_dict)
        group["eva_experiences"] = eva_experiences
        group["eva_phase"] = eva_phase
        return group

    @app.get("/eva/group/{group_id}/experiences", tags=["eva"])
    def get_group_eva_experiences(
        group_id: str, janus_core: JanusCore = Depends(get_janus_core)
    ):
        eva_memory_store = getattr(janus_core, "eva_memory_store", {})
        eva_phase = getattr(janus_core, "eva_phase", "default")
        experiences = []
        for _exp_id, reality_bytecode in eva_memory_store.items():
            exp_dict = (
                reality_bytecode.to_dict()
                if hasattr(reality_bytecode, "to_dict")
                else {}
            )
            if exp_dict.get("experience", {}).get("group_id") == group_id:
                experiences.append(exp_dict)
        return {
            "group_id": group_id,
            "eva_phase": eva_phase,
            "eva_experiences": experiences,
        }

    @app.get("/eva/phase", tags=["eva"])
    def get_eva_phase(janus_core: JanusCore = Depends(get_janus_core)):
        return {"eva_phase": getattr(janus_core, "eva_phase", "default")}

    @app.post("/groups", tags=["groups"])
    def create_group(
        request: CreateGroupRequest, janus_core: JanusCore = Depends(get_janus_core)
    ):
        group = janus_core.social_dynamics.create_group(
            request.group_id, request.member_ids
        )
        if not group:
            raise HTTPException(status_code=400, detail="Failed to create group")
        return group

    @app.get("/groups", tags=["groups"])
    def get_all_groups(janus_core: JanusCore = Depends(get_janus_core)):
        groups = janus_core.social_dynamics.get_all_groups()
        for group in groups:
            group["group_cohesion"] = janus_core.social_dynamics.get_group_cohesion(
                group["group_id"]
            )
            group["cultural_artifacts"] = (
                janus_core.social_dynamics.get_group_artifacts(group["group_id"])
            )
            # EVA: Añadir experiencias vivientes asociadas al grupo
            eva_experiences = []
            eva_memory_store = getattr(janus_core, "eva_memory_store", {})
            for _exp_id, reality_bytecode in eva_memory_store.items():
                exp_dict = (
                    reality_bytecode.to_dict()
                    if hasattr(reality_bytecode, "to_dict")
                    else {}
                )
                if exp_dict.get("experience", {}).get("group_id") == group["group_id"]:
                    eva_experiences.append(exp_dict)
            group["eva_experiences"] = eva_experiences
            group["eva_phase"] = getattr(janus_core, "eva_phase", "default")
        return groups

    @app.get("/cultural_artifacts", tags=["culture"])
    def get_cultural_artifacts(janus_core: JanusCore = Depends(get_janus_core)):
        return janus_core.social_dynamics.get_all_cultural_artifacts()

    @app.get("/entity/{entity_id}/bonds", tags=["entity"])
    def get_entity_bonds(
        entity_id: str, janus_core: JanusCore = Depends(get_janus_core)
    ):
        return janus_core.empathetic_bonding_system.get_entity_bonds(entity_id)

    @app.get("/entity/{entity_id}/groups", tags=["entity"])
    def get_entity_groups(
        entity_id: str, janus_core: JanusCore = Depends(get_janus_core)
    ):
        return janus_core.social_dynamics.get_entity_groups(entity_id)

    # EVA: Endpoint para benchmarking y métricas agregadas
    @app.get("/eva/metrics", tags=["eva"])
    def get_eva_metrics(janus_core: JanusCore = Depends(get_janus_core)):
        if hasattr(janus_core, "eva_metrics_calculator"):
            return janus_core.eva_metrics_calculator.calculate_eva_aggregated_metrics(
                phase=getattr(janus_core, "eva_phase", "default")
            )
        return {}

    return app


# This part is for standalone testing if needed
if __name__ == "__main__":
    from unittest.mock import MagicMock

    import uvicorn

    # Mock JanusCore for standalone testing
    mock_janus_core = MagicMock()
    mock_janus_core.empathetic_bonding_system.get_network_state.return_value = {
        "total_bonds": 10,
        "average_strength": 0.75,
        "group_count": 1,
        "cultural_artifact_count": 1,
    }
    mock_janus_core.empathetic_bonding_system.get_entity_social_profile.return_value = {
        "entity_id": "entity_1",
        "bonds_count": 2,
        "partners": [{"id": "entity_2", "strength": 0.8}],
        "average_bond_strength": 0.8,
        "reproduction_attempts": 1,
        "altruistic_acts": 5,
        "qualia_state": {"order": 0.6, "chaos": 0.4},
        "last_interaction": 1680000000.0,
    }
    mock_janus_core.social_dynamics.create_group.return_value = {
        "group_id": "group_1",
        "members": ["entity_1", "entity_2"],
        "group_qualia": {"order": 0.6, "chaos": 0.4},
        "cultural_artifacts": ["artifact_1"],
        "group_cohesion": 0.8,
        "creation_time": 1680000000.0,
    }
    mock_janus_core.social_dynamics.get_group.return_value = {
        "group_id": "group_1",
        "members": ["entity_1", "entity_2"],
        "group_qualia": {"order": 0.6, "chaos": 0.4},
        "cultural_artifacts": ["artifact_1"],
        "group_cohesion": 0.8,
        "creation_time": 1680000000.0,
    }
    mock_janus_core.social_dynamics.get_all_groups.return_value = [
        {
            "group_id": "group_1",
            "members": ["entity_1", "entity_2"],
            "group_qualia": {"order": 0.6, "chaos": 0.4},
            "cultural_artifacts": ["artifact_1"],
            "group_cohesion": 0.8,
            "creation_time": 1680000000.0,
        }
    ]
    mock_janus_core.social_dynamics.get_cultural_artifact_count.return_value = 1
    mock_janus_core.social_dynamics.get_all_cultural_artifacts.return_value = [
        "artifact_1"
    ]
    mock_janus_core.social_dynamics.get_group_cohesion.return_value = 0.8
    mock_janus_core.social_dynamics.get_group_creation_time.return_value = 1680000000.0
    mock_janus_core.social_dynamics.get_group_artifacts.return_value = ["artifact_1"]
    mock_janus_core.empathetic_bonding_system.get_entity_bonds.return_value = [
        {"id": "entity_2", "strength": 0.8}
    ]
    mock_janus_core.social_dynamics.get_entity_groups.return_value = ["group_1"]
    mock_janus_core.storage_manager.get_entity_qualia_state.return_value = {
        "order": 0.6,
        "chaos": 0.4,
    }
    mock_janus_core.storage_manager.get_last_interaction_time.return_value = (
        1680000000.0
    )

    app = create_social_api(mock_janus_core)
    uvicorn.run(app, host="0.0.0.0", port=8000)
