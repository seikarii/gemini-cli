import random
from collections.abc import Callable
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from crisalida_lib.ADAM.adam import ConsciousMind
from crisalida_lib.ADAM.cuerpo.genome import GenomaComportamiento
from crisalida_lib.AKASHA.noosphere_client import NoosphereClient
from crisalida_lib.EDEN.living_symbol import (
    LivingSymbolRuntime,
    QuantumField,
)
from crisalida_lib.EVA.types import (
    EVAExperience,
    QualiaState,
    RealityBytecode,
)

from crisalida_lib.EDEN.qualia_manifold import QualiaField
from crisalida_lib.EDEN.qualia_manifold import create_qualia_field


class QualiaCristalizada(BaseModel):
    """
    Represents a unit of crystallized qualia, a form of energetic currency.
    """

    amount: float = Field(..., ge=0, description="Amount of crystallized qualia.")
    qualia_type: Literal["order", "chaos", "neutral"] = Field(
        "neutral", description="Dominant qualia type."
    )


class GeneOffer(BaseModel):
    """
    Represents an offer for a gene (behavioral genome component) in the market.
    """

    offer_id: str = Field(..., description="Unique ID for the gene offer.")
    gene_id: str = Field(..., description="ID of the gene being offered.")
    gene_data: dict[str, Any] = Field(
        ..., description="Data of the gene (e.g., behavioral propensity)."
    )
    price: QualiaCristalizada = Field(..., description="Price in crystallized qualia.")
    seller_id: str = Field(..., description="ID of the entity offering the gene.")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Timestamp of the offer."
    )


class NoosphereRecord(BaseModel):
    """
    Represents a single record of information within the Noosphere.
    """

    record_id: str = Field(..., description="Unique identifier for the record.")
    content: Any = Field(..., description="The information content of the record.")
    qualia_signature: dict[str, float] = Field(
        default_factory=dict,
        description="A signature derived from the Qualia Field at the time of creation.",
    )
    creation_timestamp: datetime = Field(
        default_factory=datetime.now, description="Timestamp of record creation."
    )
    tags: list[str] = Field(
        default_factory=list, description="Tags for categorization and search."
    )
    resonance_strength: float = Field(
        0.0,
        ge=0,
        le=1,
        description="How strongly this record resonates with the current Qualia Field (0-1).",
    )


class Noosphere(BaseModel):
    """
    The Noosphere (the "Akashic Record") is the holographic state of the Qualia Field.
    Knowledge and telepathy result from tuning consciousness to resonate with its information patterns.
    EVA EXTENSION: Integración de memoria viviente, ingestión/recall de experiencias, faseo, hooks y benchmarking.
    """

    records: dict[str, NoosphereRecord] = Field(
        default_factory=dict, description="Collection of all Noosphere records."
    )
    global_qualia_field: QualiaField = Field(
        default_factory=QualiaField, description="The global Qualia Field."
    )
    qualia_cristalizada_supply: dict[Literal["order", "chaos", "neutral"], float] = (
        Field(
            default_factory=lambda: {"order": 0.0, "chaos": 0.0, "neutral": 0.0},
            description="Total supply of crystallized qualia by type.",
        )
    )
    gene_market_offers: dict[str, GeneOffer] = Field(
        default_factory=dict, description="Offers for genes in the market."
    )
    telepathy_noise_level: float = Field(
        0.1,
        description="Global noise level affecting telepathic transmissions (0-1).",
    )
    transaction_fee_rate: float = Field(
        0.01,
        description="Percentage fee applied to gene market transactions.",
    )
    noosphere_client: NoosphereClient | None = Field(
        None, description="Client for P2P interactions within the Noosphere."
    )

    # EVA: Memoria viviente y faseo
    eva_runtime: LivingSymbolRuntime = Field(default_factory=LivingSymbolRuntime)
    eva_memory_store: dict[str, RealityBytecode] = Field(default_factory=dict)
    eva_experience_store: dict[str, EVAExperience] = Field(default_factory=dict)
    eva_phases: dict[str, dict[str, RealityBytecode]] = Field(default_factory=dict)
    eva_phase: str = Field("default")
    _environment_hooks: list = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def set_noosphere_client(self, client: NoosphereClient) -> None:
        self.noosphere_client = client

    def add_record(
        self, record_id: str, content: Any, tags: list[str] | None = None
    ) -> NoosphereRecord:
        """
        Adds a new information record to the Noosphere.
        The qualia_signature is derived from the current state of the global Qualia Field.
        """
        if record_id in self.records:
            raise ValueError(f"Record with ID '{record_id}' already exists.")
        alpha, beta = self.global_qualia_field.get_superposition_state()
        qualia_signature = {"alpha": alpha, "beta": beta}
        new_record = NoosphereRecord(
            record_id=record_id,
            content=content,
            qualia_signature=qualia_signature,
            tags=tags or [],
        )
        self.records[record_id] = new_record
        print(f"Added record '{record_id}' to Noosphere.")
        return new_record

    def get_record(self, record_id: str) -> NoosphereRecord | None:
        """
        Retrieves a specific record by its ID.
        """
        return self.records.get(record_id)

    def search_records(
        self, query: str, tags: list[str] | None = None
    ) -> list[NoosphereRecord]:
        """
        Searches for records based on a query and optional tags.
        Calculates resonance strength for each matching record, considering semantic similarity.
        """
        matching_records = []
        current_alpha, current_beta = self.global_qualia_field.get_superposition_state()
        query_words = set(query.lower().split())
        for record in self.records.values():
            content_str = str(record.content).lower()
            record_words = set(content_str.split())
            # Semantic similarity (simple keyword overlap)
            semantic_match_score = (
                len(query_words.intersection(record_words)) / len(query_words)
                if query_words
                else 0.0
            )
            # Tag match
            tag_match_score = 0.0
            if tags:
                record_tags_lower = [t.lower() for t in record.tags]
                query_tags_lower = [t.lower() for t in tags]
                tag_match_score = (
                    len(set(query_tags_lower).intersection(set(record_tags_lower)))
                    / len(query_tags_lower)
                    if query_tags_lower
                    else 0.0
                )
            # Qualia signature resonance
            record_alpha = record.qualia_signature.get("alpha", 0.5)
            record_beta = record.qualia_signature.get("beta", 0.5)
            qualia_resonance = 1.0 - (
                (
                    (current_alpha - record_alpha) ** 2
                    + (current_beta - record_beta) ** 2
                )
                ** 0.5
            )
            qualia_resonance = max(0.0, qualia_resonance)  # Clamp to 0-1
            # Combine scores (weights can be adjusted)
            total_resonance = (
                (semantic_match_score * 0.5)
                + (tag_match_score * 0.3)
                + (qualia_resonance * 0.2)
            )
            record.resonance_strength = total_resonance
            if total_resonance > 0.1:  # Only add if there's some relevance
                matching_records.append(record)
        # Sort by resonance strength (higher is better)
        matching_records.sort(key=lambda r: r.resonance_strength, reverse=True)
        print(f"Found {len(matching_records)} records for query '{query}'.")
        return matching_records

    async def simulate_telepathy(
        self,
        sender_id: str,
        receiver_conscious_mind: ConsciousMind,
        message: str,
        signal_strength: float = 1.0,
    ) -> bool:
        """
        Simulates a telepathic communication by influencing the receiver's consciousness
        and adding a record to the Noosphere, leveraging the P2P client.
        """
        print(
            f"Simulating telepathy: '{message}' from sender {sender_id} with influence, signal strength {signal_strength}..."
        )
        if not self.noosphere_client:
            print("NoosphereClient not set. Cannot simulate P2P telepathy.")
            return False
        # 1. Calculate effective signal strength after noise
        effective_signal = max(
            0.0, signal_strength - self.telepathy_noise_level * random.uniform(0.5, 1.5)
        )
        if effective_signal < 0.1:  # Message too weak to transmit
            print("Telepathic signal too weak or too much noise. Message lost.")
            return False
        # 2. Influence the global Qualia Field based on effective signal
        self.global_qualia_field.influence_field(
            effective_signal * 0.1, is_order_influence=True
        )
        # 3. Add message to Noosphere as a record (content might be garbled if signal is low)
        transmitted_message = message
        if effective_signal < 0.5:  # Simulate some message degradation
            transmitted_message = (
                "[GARBLED] "
                + message[: int(len(message) * effective_signal * 2)]
                + "..."
            )
        record_id = f"telepathic_msg_{datetime.now().timestamp()}"
        self.add_record(
            record_id,
            transmitted_message,
            tags=["telepathy", "communication", "direct_thought"],
        )
        # 4. Use NoosphereClient to send the message (simulated P2P)
        receiver_entity_id = getattr(
            receiver_conscious_mind, "entity_id", "receiver_placeholder_id"
        )
        await self.noosphere_client.send_direct_message(
            receiver_entity_id, transmitted_message
        )
        # 5. Directly influence receiver's ConsciousMind
        receiver_conscious_mind.add_to_working_memory(
            content=f"Telepathic message: {transmitted_message}",
            memory_type="episodic",
            importance=effective_signal,  # Use effective_signal as importance
            tags=["telepathy", "received_thought"],
        )
        receiver_conscious_mind.current_qualia.update_state(
            order=receiver_conscious_mind.current_qualia.order
            + effective_signal * 0.05,
            chaos=receiver_conscious_mind.current_qualia.chaos
            - effective_signal * 0.02,
            emotional_valence=(0.5 if effective_signal > 0.5 else -0.5),
            cognitive_clarity=receiver_conscious_mind.current_qualia.cognitive_clarity,
        )
        print(
            f"Telepathic message processed by receiver's conscious mind with effective signal {effective_signal:.2f}."
        )
        return True

    def generate_qualia_cristalizada(
        self, qualia_type: Literal["order", "chaos", "neutral"], amount: float
    ) -> QualiaCristalizada:
        """
        Generates a specified amount of crystallized qualia of a given type.
        """
        if amount <= 0:
            raise ValueError("Amount must be positive.")
        self.qualia_cristalizada_supply[qualia_type] += amount
        print(
            f"Generated {amount:.2f} {qualia_type} Qualia Cristalizada. Total: {self.qualia_cristalizada_supply[qualia_type]:.2f}"
        )
        return QualiaCristalizada(amount=amount, qualia_type=qualia_type)

    async def consume_qualia_cristalizada(
        self, qualia_type: Literal["order", "chaos", "neutral"], amount: float
    ) -> None:
        """
        Consumes a specified amount of crystallized qualia.
        """
        if self.qualia_cristalizada_supply[qualia_type] < amount:
            print(
                f"Insufficient {qualia_type} Qualia Cristalizada. Available: {self.qualia_cristalizada_supply[qualia_type]:.2f}, Needed: {amount:.2f}"
            )
            return
        self.qualia_cristalizada_supply[qualia_type] -= amount
        print(
            f"Consumed {amount:.2f} {qualia_type} Qualia Cristalizada. Remaining: {self.qualia_cristalizada_supply[qualia_type]:.2f}"
        )
        # Publish consumption event to Noosphere (P2P)
        if self.noosphere_client:
            await self.noosphere_client.publish_post(
                content_text=f"Consumed {amount:.2f} {qualia_type} Qualia Cristalizada.",
                post_type="qualia_consumption",
                qualia_state_json=str({"type": qualia_type, "amount": amount}),
            )

    async def offer_gene(
        self,
        offer_id: str,
        gene_data: dict[str, Any],
        price: QualiaCristalizada,
        seller_id: str,
    ) -> GeneOffer:
        """
        Adds a gene offer to the market and publishes it to the Noosphere (P2P).
        """
        if offer_id in self.gene_market_offers:
            raise ValueError(f"Gene offer with ID '{offer_id}' already exists.")
        new_offer = GeneOffer(
            offer_id=offer_id,
            gene_id=gene_data.get("gene_id", f"gene_{len(self.gene_market_offers)}"),
            gene_data=gene_data,
            price=price,
            seller_id=seller_id,
        )
        self.gene_market_offers[offer_id] = new_offer
        print(
            f"Gene offer '{offer_id}' added to market for {price.amount:.2f} {price.qualia_type} qualia."
        )
        # Publish gene offer to Noosphere (P2P)
        if self.noosphere_client:
            await self.noosphere_client.publish_post(
                content_text=f"Gene Offer: {new_offer.gene_id} for {new_offer.price.amount:.2f} {new_offer.price.qualia_type} by {new_offer.seller_id}",
                post_type="gene_offer",
                qualia_state_json=str(new_offer.model_dump()),
            )
        return new_offer

    async def purchase_gene(
        self,
        offer_id: str,
        buyer_id: str,
        buyer_qualia_balance: dict[Literal["order", "chaos", "neutral"], float],
        seller_qualia_balance: dict[Literal["order", "chaos", "neutral"], float],
    ) -> GenomaComportamiento | None:
        """
        Simulates the purchase of a gene from the market, leveraging the P2P client.
        Transfers qualia from buyer to seller and applies a transaction fee.
        """
        offer = self.gene_market_offers.get(offer_id)
        if not offer:
            print(f"Gene offer '{offer_id}' not found.")
            return None
        if buyer_qualia_balance.get(offer.price.qualia_type, 0) < offer.price.amount:
            print(
                f"Buyer {buyer_id} has insufficient {offer.price.qualia_type} qualia to purchase gene offer '{offer_id}'."
            )
            return None
        # Calculate fee
        fee_amount = offer.price.amount * self.transaction_fee_rate
        amount_to_seller = offer.price.amount - fee_amount
        # Transfer qualia from buyer to seller
        buyer_qualia_balance[offer.price.qualia_type] -= offer.price.amount
        seller_qualia_balance[offer.price.qualia_type] += amount_to_seller
        # Remove fee from circulation (or add to a treasury if implemented later)
        self.qualia_cristalizada_supply[offer.price.qualia_type] -= fee_amount
        purchased_gene = GenomaComportamiento(
            genome_id=offer.gene_id,
            behavioral_genome=offer.gene_data.get("behavioral_genome", {}),
        )
        del self.gene_market_offers[offer_id]
        print(
            f"Gene '{offer.gene_id}' purchased by {buyer_id} for {offer.price.amount:.2f} {offer.price.qualia_type} qualia. {fee_amount:.2f} fee applied."
        )
        # Publish purchase event to Noosphere (P2P)
        if self.noosphere_client:
            await self.noosphere_client.publish_post(
                content_text=f"Gene Purchased: {offer.gene_id} by {buyer_id} from {offer.seller_id}",
                post_type="gene_purchase",
                qualia_state_json=str(
                    {
                        "gene_id": offer.gene_id,
                        "buyer_id": buyer_id,
                        "seller_id": offer.seller_id,
                        "price": offer.price.amount,
                        "fee": fee_amount,
                    }
                ),
            )
        return purchased_gene

    def eva_ingest_experience(
        self, experience_data: dict, qualia_state: QualiaState, phase: str = None
    ) -> str:
        """
        Compila una experiencia arbitraria en RealityBytecode y la almacena en la memoria viviente EVA.
        """
        phase = phase or self.eva_phase
        intention = {
            "intention_type": "ARCHIVE_NOOSPHERE_EXPERIENCE",
            "experience": experience_data,
            "qualia": qualia_state,
            "phase": phase,
        }
        bytecode = self.eva_runtime.divine_compiler.compile_intention(intention)
        experience_id = (
            experience_data.get("experience_id")
            or f"eva_noosphere_{hash(str(experience_data))}"
        )
        reality_bytecode = RealityBytecode(
            bytecode_id=experience_id,
            instructions=bytecode,
            qualia_state=qualia_state,
            phase=phase,
            timestamp=experience_data.get("timestamp", datetime.now().timestamp()),
        )
        self.eva_memory_store[experience_id] = reality_bytecode
        if phase not in self.eva_phases:
            self.eva_phases[phase] = {}
        self.eva_phases[phase][experience_id] = reality_bytecode
        print(f"[EVA-NOOSPHERE] Ingested EVA experience: {experience_id}")
        return experience_id

    def eva_recall_experience(self, cue: str, phase: str = None) -> dict:
        """
        Ejecuta el RealityBytecode de una experiencia almacenada, manifestando la simulación en QuantumField.
        """
        phase = phase or self.eva_phase
        reality_bytecode = self.eva_phases.get(phase, {}).get(
            cue
        ) or self.eva_memory_store.get(cue)
        if not reality_bytecode:
            return {"error": "No bytecode found for EVA Noosphere experience"}
        quantum_field = QuantumField()
        manifestations = []
        for instr in reality_bytecode.instructions:
            symbol_manifest = self.eva_runtime.execute_instruction(instr, quantum_field)
            if symbol_manifest:
                manifestations.append(symbol_manifest)
                for hook in self._environment_hooks:
                    try:
                        hook(symbol_manifest)
                    except Exception as e:
                        print(f"[EVA-NOOSPHERE] Environment hook failed: {e}")
        eva_experience = EVAExperience(
            experience_id=reality_bytecode.bytecode_id,
            bytecode=reality_bytecode,
            manifestations=manifestations,
            phase=reality_bytecode.phase,
            qualia_state=reality_bytecode.qualia_state,
            timestamp=reality_bytecode.timestamp,
        )
        self.eva_experience_store[reality_bytecode.bytecode_id] = eva_experience
        return {
            "experience_id": eva_experience.experience_id,
            "manifestations": [m.to_dict() for m in manifestations],
            "phase": eva_experience.phase,
            "qualia_state": (
                eva_experience.qualia_state.to_dict()
                if hasattr(eva_experience.qualia_state, "to_dict")
                else {}
            ),
            "timestamp": eva_experience.timestamp,
        }

    def add_experience_phase(
        self,
        experience_id: str,
        phase: str,
        experience_data: dict,
        qualia_state: QualiaState,
    ):
        """
        Añade una fase alternativa (timeline) para una experiencia arbitraria.
        """
        intention = {
            "intention_type": "ARCHIVE_NOOSPHERE_EXPERIENCE",
            "experience": experience_data,
            "qualia": qualia_state,
            "phase": phase,
        }
        bytecode = self.eva_runtime.divine_compiler.compile_intention(intention)
        reality_bytecode = RealityBytecode(
            bytecode_id=experience_id,
            instructions=bytecode,
            qualia_state=qualia_state,
            phase=phase,
            timestamp=experience_data.get("timestamp", datetime.now().timestamp()),
        )
        if phase not in self.eva_phases:
            self.eva_phases[phase] = {}
        self.eva_phases[phase][experience_id] = reality_bytecode
        print(
            f"[EVA-NOOSPHERE] Added phase '{phase}' for EVA experience {experience_id}"
        )

    def set_memory_phase(self, phase: str):
        """Cambia la fase activa de memoria EVA."""
        self.eva_phase = phase
        for hook in self._environment_hooks:
            try:
                hook({"phase_changed": phase})
            except Exception as e:
                print(f"[EVA-NOOSPHERE] Phase hook failed: {e}")

    def get_memory_phase(self) -> str:
        """Devuelve la fase de memoria actual."""
        return self.eva_phase

    def get_experience_phases(self, experience_id: str) -> list:
        """Lista todas las fases disponibles para una experiencia EVA."""
        return [
            phase for phase, exps in self.eva_phases.items() if experience_id in exps
        ]

    def add_environment_hook(self, hook: Callable[..., Any]):
        """Registra un hook para manifestación simbólica o eventos EVA."""
        self._environment_hooks.append(hook)

    def get_eva_api(self) -> dict:
        return {
            "eva_ingest_experience": self.eva_ingest_experience,
            "eva_recall_experience": self.eva_recall_experience,
            "add_experience_phase": self.add_experience_phase,
            "set_memory_phase": self.set_memory_phase,
            "get_memory_phase": self.get_memory_phase,
            "get_experience_phases": self.get_experience_phases,
            "add_environment_hook": self.add_environment_hook,
        }


# Example Usage
if __name__ == "__main__":
    import asyncio

    from crisalida_lib.core.conscious_mind import ConsciousMind

    async def main():
        # Setup a dummy DHT node for the NoosphereClient
        dht_node_info = {
            "host": "127.0.0.1",
            "port": 8468,
            "node_id": "noosphere_node_1",
        }
        noosphere_p2p_client = NoosphereClient(
            dht_node_info=dht_node_info, entity_id="noosphere_manager"
        )
        global_field = create_qualia_field()
        noosphere = Noosphere(global_qualia_field=global_field)
        noosphere.set_noosphere_client(noosphere_p2p_client)
        # Generate some crystallized qualia
        noosphere.generate_qualia_cristalizada("order", 100.0)
        noosphere.generate_qualia_cristalizada("chaos", 50.0)
        # Simulate seller's initial balance
        seller_balance = {"order": 10.0, "chaos": 5.0, "neutral": 0.0}
        # Offer a gene
        test_gene_data = {
            "gene_id": "gene_A",
            "behavioral_genome": {"propensities": {"EXPLORE": 0.9}},
        }
        order_price = QualiaCristalizada(amount=30.0, qualia_type="order")
        await noosphere.offer_gene(
            "offer_1", test_gene_data, order_price, "entity_seller_1"
        )
        # Simulate a buyer
        buyer_balance = {"order": 50.0, "chaos": 20.0, "neutral": 0.0}
        purchased_gene = await noosphere.purchase_gene(
            "offer_1", "entity_buyer_1", buyer_balance, seller_balance
        )
        if purchased_gene:
            print(
                f"Purchased Gene: {purchased_gene.genome_id}, Propensity: {purchased_gene.behavioral_genome.get('propensities', {}).get('EXPLORE')}"
            )
            print(f"Buyer's remaining order qualia: {buyer_balance['order']:.2f}")
            print(f"Seller's new order qualia: {seller_balance['order']:.2f}")
            print(
                f"\nNoosphere Qualia Cristalizada Supply: {noosphere.qualia_cristalizada_supply}"
            )
        print(f"Noosphere Gene Market Offers: {noosphere.gene_market_offers}")
        # Simulate telepathy
        dummy_conscious_mind = ConsciousMind()
        print("\nSimulating telepathy with high signal strength...")
        await noosphere.simulate_telepathy(
            "sender_entity_id",
            dummy_conscious_mind,
            "The market is thriving.",
            signal_strength=0.9,
        )
        print("\nSimulating telepathy with low signal strength (should be garbled)...")
        await noosphere.simulate_telepathy(
            "sender_entity_id",
            dummy_conscious_mind,
            "Secret plan to escape.",
            signal_strength=0.2,
        )
        print(
            "\nSimulating telepathy with very low signal strength (should be lost)..."
        )
        await noosphere.simulate_telepathy(
            "sender_entity_id",
            dummy_conscious_mind,
            "Critical warning.",
            signal_strength=0.05,
        )

    asyncio.run(main())
