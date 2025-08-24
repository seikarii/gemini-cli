# Demiurge Module

This directory contains the interface and control systems for the 'Demiurge'. The Demiurge is the entity or interface that acts as the creator, with the ability to interact with and modify the simulation at a fundamental level.

## Core Components

*   `demiurge.py`: The main interface for the Demiurge's interaction. It allows initializing the Demiurge's state, compiling and executing ontological intentions, and manifesting matter in the symbolic universe using the VM and the quantum compiler.
*   `demiurge_avatar.py`: Defines the `DemiurgeAvatar` class, representing the player's manifestation within the Janus Metacosmos. It extends `ComplexBeing` and includes attributes and methods specific to the player's interaction with the simulation, such as power level, known words, and acquired abilities.
*   `demiurge_controls.py`: The Demiurge's control system, interpreting intentions from various inputs. It implements Layers 1 (Intuitive Control) and 2 (Divine Word) of the IMD.