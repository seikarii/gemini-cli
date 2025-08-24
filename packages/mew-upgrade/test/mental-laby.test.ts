/**
 * @file Test file for the MentalLaby and its persistence.
 */

import { MentalLaby, MemoryNode } from '../src/mind/mental-laby';
import { StateSerializer } from '../src/persistence/state-serializer';
import { LocalStorageBackend } from '../src/persistence/storage-manager';
import { PersistenceService } from '../src/persistence/persistence-service';

async function runMentalLabyTests() {
  console.log("\n--- Running MentalLaby Tests ---");

  // --- Setup Persistence Service ---
  const testBasePath = '/tmp/mew-agent-test-state'; // Use a temporary directory for tests
  const serializer = new StateSerializer();
  const storage = new LocalStorageBackend();
  const persistenceService = new PersistenceService(testBasePath, storage, serializer);

  // --- Test 1: Basic Store and Recall ---
  console.log("\nTest 1: Basic Store and Recall");
  const laby1 = new MentalLaby();
  laby1.store({ concept: "apple", color: "red" });
  laby1.store({ concept: "banana", color: "yellow" });
  laby1.store({ concept: "fruit", taste: "sweet" });

  let recalledMemories = laby1.recall({ concept: "apple" });
  console.log("Recalled for 'apple':", recalledMemories.map(m => m.data));
  // Expected: should recall 'apple' and 'fruit'

  recalledMemories = laby1.recall({ concept: "yellow" });
  console.log("Recalled for 'yellow':", recalledMemories.map(m => m.data));
  // Expected: should recall 'banana'

  // --- Test 2: Reinforcement ---
  console.log("\nTest 2: Reinforcement");
  laby1.store({ concept: "apple", color: "red" }); // Store again to reinforce
  const appleNode = Array.from(laby1.exportState().nodes as any[]).find(n => n.data.concept === "apple");
  console.log("Apple node usage count after reinforcement:", appleNode ? appleNode.usageCount : 'N/A');
  // Expected: usageCount should be 2

  // --- Test 3: Persistence ---
  console.log("\nTest 3: Persistence");
  const laby2 = new MentalLaby();
  const testKey = 'test_memory_state';

  // Save state from laby1
  await persistenceService.save(laby1, testKey);
  console.log("State saved from laby1.");

  // Load state into laby2
  await persistenceService.load(laby2, testKey);
  console.log("State loaded into laby2.");

  // Verify loaded state
  recalledMemories = laby2.recall({ concept: "fruit" });
  console.log("Recalled from loaded laby2 for 'fruit':", recalledMemories.map(m => m.data));
  // Expected: should recall 'fruit' and 'apple'

  const initialNodeCount = Array.from(laby1.exportState().nodes as any[]).length;
  const loadedNodeCount = Array.from(laby2.exportState().nodes as any[]).length;
  console.log(`Node count: Original=${initialNodeCount}, Loaded=${loadedNodeCount}`);
  // Expected: counts should match

  console.log("\n--- MentalLaby Tests Complete ---");
}

runMentalLabyTests().catch(console.error);
