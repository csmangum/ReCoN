#!/usr/bin/env python3
"""
FastAPI backend for the Syllable Template Learning App.
Provides API endpoints for ReCoN network operations.
"""

import sys
import os
sys.path.append('/workspace')

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import yaml
import json

from recon_core.compiler import compile_from_dict
from recon_core.engine import Engine
from recon_core.config import EngineConfig
from recon_core.enums import State, UnitType
from perception.audio_terminals import create_synthetic_audio_features

app = FastAPI(title="Syllable Template API", version="1.0.0")

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global network state
network_state = {
    "graph": None,
    "engine": None,
    "syllables": [],
    "terminals": [],
    "words": [],
    "initialized": False
}

class SyllableActivationRequest(BaseModel):
    syllable_id: str
    steps: int = 3

class SimulationRequest(BaseModel):
    steps: int = 10
    reset: bool = True

class UnitState(BaseModel):
    id: str
    state: str
    activation: float
    kind: str

class NetworkState(BaseModel):
    units: List[UnitState]
    step: int
    terminals: Dict[str, float]

def initialize_network():
    """Initialize the ReCoN network."""
    try:
        # Load network specification
        with open('/workspace/scripts/engage_active_perception.yaml', 'r') as f:
            network_spec = yaml.safe_load(f)
        
        # Compile network
        graph = compile_from_dict(network_spec)
        
        # Create engine
        config = EngineConfig()
        config.deterministic_order = True
        config.confirmation_ratio = 0.75
        config.sur_positive = 0.4
        config.ret_feedback_enabled = True
        
        engine = Engine(graph, config)
        
        # Extract units by type
        syllables = []
        terminals = []
        words = []
        
        for unit_id, unit in graph.units.items():
            if unit.kind == UnitType.SCRIPT:
                if 'phoneme' in unit_id:
                    syllables.append(unit_id)
                elif unit_id != 'u_phrase':
                    words.append(unit_id)
            elif unit.kind == UnitType.TERMINAL:
                terminals.append(unit_id)
        
        # Update global state
        network_state.update({
            "graph": graph,
            "engine": engine,
            "syllables": syllables,
            "terminals": terminals,
            "words": words,
            "initialized": True
        })
        
        return True
        
    except Exception as e:
        print(f"Failed to initialize network: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize the network on startup."""
    initialize_network()

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Syllable Template API", "status": "running"}

@app.get("/network/info")
async def get_network_info():
    """Get basic network information."""
    if not network_state["initialized"]:
        raise HTTPException(status_code=500, detail="Network not initialized")
    
    return {
        "syllables": [s.replace('u_', '').replace('_phoneme', '') for s in network_state["syllables"]],
        "terminals": network_state["terminals"],
        "words": [w.replace('u_', '') for w in network_state["words"]],
        "total_units": len(network_state["graph"].units)
    }

@app.get("/network/state")
async def get_network_state():
    """Get current network state."""
    if not network_state["initialized"]:
        raise HTTPException(status_code=500, detail="Network not initialized")
    
    graph = network_state["graph"]
    engine = network_state["engine"]
    
    # Get unit states
    units = []
    for unit_id, unit in graph.units.items():
        units.append(UnitState(
            id=unit_id,
            state=unit.state.name,
            activation=unit.a,
            kind=unit.kind.name
        ))
    
    # Get terminal activations
    terminal_activations = {}
    for terminal_id in network_state["terminals"]:
        if terminal_id in graph.units:
            unit = graph.units[terminal_id]
            terminal_activations[terminal_id] = unit.a
    
    return NetworkState(
        units=units,
        step=engine.t,
        terminals=terminal_activations
    )

@app.post("/network/reset")
async def reset_network():
    """Reset the network to initial state."""
    if not network_state["initialized"]:
        raise HTTPException(status_code=500, detail="Network not initialized")
    
    network_state["engine"].reset()
    return {"message": "Network reset", "step": 0}

@app.post("/syllable/activate")
async def activate_syllable(request: SyllableActivationRequest):
    """Activate a specific syllable."""
    if not network_state["initialized"]:
        raise HTTPException(status_code=500, detail="Network not initialized")
    
    graph = network_state["graph"]
    engine = network_state["engine"]
    
    # Reset network
    engine.reset()
    
    # Activate the syllable
    syllable_id = f"u_{request.syllable_id}_phoneme"
    if syllable_id not in graph.units:
        raise HTTPException(status_code=404, detail=f"Syllable {request.syllable_id} not found")
    
    unit = graph.units[syllable_id]
    unit.state = State.ACTIVE
    unit.a = 1.0
    
    # Run simulation steps
    for _ in range(request.steps):
        engine.step()
    
    # Return updated state
    return await get_network_state()

@app.post("/simulation/run")
async def run_simulation(request: SimulationRequest):
    """Run a simulation for the specified number of steps."""
    if not network_state["initialized"]:
        raise HTTPException(status_code=500, detail="Network not initialized")
    
    engine = network_state["engine"]
    
    if request.reset:
        engine.reset()
    
    # Run simulation
    for _ in range(request.steps):
        engine.step()
    
    return await get_network_state()

@app.get("/audio/features")
async def get_audio_features():
    """Get synthetic audio features."""
    features = create_synthetic_audio_features("engage active perception")
    return {"features": features}

@app.get("/syllables")
async def get_syllables():
    """Get list of available syllables with their display names."""
    if not network_state["initialized"]:
        raise HTTPException(status_code=500, detail="Network not initialized")
    
    syllables = []
    for syllable_id in network_state["syllables"]:
        display_name = syllable_id.replace('u_', '').replace('_phoneme', '')
        syllables.append({
            "id": syllable_id,
            "display_name": display_name
        })
    
    return {"syllables": syllables}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)