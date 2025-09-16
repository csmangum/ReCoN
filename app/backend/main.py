from __future__ import annotations

import asyncio
from typing import Literal

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

from .engine import InMemoryReconEngine, NodeState


app = FastAPI(title="ReCoN API", version="0.1.0")

# Allow local dev frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = InMemoryReconEngine()


class ControlRequest(BaseModel):
    cmd: Literal["step", "run", "pause", "reset"]


@app.get("/recon/graph")
async def get_graph():
    return JSONResponse(jsonable_encoder(engine.graph()))


@app.get("/recon/state")
async def get_state():
    return JSONResponse(jsonable_encoder(engine.state()))


@app.post("/recon/control")
async def post_control(body: ControlRequest):
    if body.cmd == "step":
        done = await engine.step()
        return {"ok": True, "done": done}
    if body.cmd == "run":
        await engine.run()
        return {"ok": True}
    if body.cmd == "pause":
        await engine.pause()
        return {"ok": True}
    if body.cmd == "reset":
        await engine.reset()
        return {"ok": True}
    return {"ok": False}


@app.websocket("/recon/stream")
async def ws_stream(ws: WebSocket):
    await ws.accept()

    # Send initial graph
    await ws.send_json({
        "type": "init",
        "graph": jsonable_encoder(engine.graph()),
    })

    q = engine.subscribe()
    try:
        # Immediately push current state to client
        await ws.send_json({
            "type": "state",
            "step": engine.state().step,
            "nodeStates": {k: v for k, v in engine.state().nodeStates.items()},
            "edgeStates": engine.state().edgeStates,
            "explanations": engine.state().explanations,
        })

        while True:
            try:
                st = await q.get()
            except asyncio.CancelledError:
                break

            await ws.send_json({
                "type": "state",
                "step": st.step,
                "nodeStates": st.nodeStates,
                "edgeStates": st.edgeStates,
                "explanations": st.explanations,
            })

            # Emit a done notification when we reach a fixpoint in this stub
            if engine.is_done():
                await ws.send_json({"type": "done", "reason": "fixpoint"})
    except WebSocketDisconnect:
        pass
    finally:
        engine.unsubscribe(q)


@app.get("/")
async def root():
    return {"service": "recon", "status": "ok"}

