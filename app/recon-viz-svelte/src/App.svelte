<script lang="ts">
  import { onMount } from "svelte";
  import GraphView from "./lib/GraphView.svelte";
  import Controls from "./lib/Controls.svelte";
  import type { ReconGraph, ReconState, StreamMsg } from "./lib/types";

  const WS_URL   = import.meta.env.VITE_WS_URL  ?? "ws://localhost:8000/recon/stream";
  const HTTP_BASE = import.meta.env.VITE_HTTP_BASE ?? "http://localhost:8000";

  let graph: ReconGraph | null = null;
  let state: ReconState | null = null;
  let connected = false;
  let log: string[] = [];

  let ws: WebSocket | null = null;

  function pushLog(line: string) { log = [`${new Date().toLocaleTimeString()} ${line}`, ...log]; }

  onMount(() => {
    ws = new WebSocket(WS_URL);
    ws.onopen = () => { connected = true; pushLog("[ws] connected"); };
    ws.onclose = () => { connected = false; pushLog("[ws] disconnected"); };
    ws.onerror = () => { connected = false; pushLog("[ws] error"); };
    ws.onmessage = (evt) => {
      const msg: StreamMsg = JSON.parse(evt.data);
      if (msg.type === "init"):
        graph = msg.graph;
        pushLog(`[init] graph received (${graph.nodes.length} nodes)`);
      else if (msg.type === "state"):
        state = {
          step: msg.step,
          nodeStates: msg.nodeStates,
          edgeStates: msg.edgeStates,
          explanations: msg.explanations
        };
        if (state.explanations?.length) state.explanations.forEach(e => pushLog(`[s${state!.step}] ${e}`));
      else if (msg.type === "done"):
        pushLog(`[done] ${msg.reason ?? "fixpoint"}`);
    };
    return () => ws?.close();
  });

  const canControl = () => !!graph;

  async function sendControl(cmd: "step"|"run"|"pause"|"reset") {
    await fetch(`${HTTP_BASE}/recon/control`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cmd })
    });
  }

  function exportJSON() {
    if (!graph || !state) return;
    const blob = new Blob([JSON.stringify({ graph, state }, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = `recon_step_${state.step}.json`; a.click();
    URL.revokeObjectURL(url);
  }
</script>

<style>
  :global(:root){ --bg:#0f1220; --panel:#14182b; --fg:#e7ebf6; --muted:#9aa3b2; --accent:#4c9eff; }
  :global(html,body,#app){ height:100%; margin:0; }
  :global(body){ background:var(--bg); color:var(--fg); font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto; }
  header{ display:flex; align-items:center; justify-content:space-between; padding:12px 16px; border-bottom:1px solid #202544; }
  .pill{ padding:4px 8px; border-radius:999px; font-size:12px; border:1px solid #2b335a; }
  .ok{ background:#183a24; border-color:#1d5a2f; color:#bff5c9; }
  .bad{ background:#3a1820; border-color:#5a1d2b; color:#f5bfc9; }
  .wrap{ display:grid; grid-template-columns: 320px 1fr; height: calc(100vh - 54px); }
  aside{ background:var(--panel); border-right:1px solid #202544; padding:12px; overflow:auto; }
  main{ position:relative; }
  .log{ margin-top:12px; }
  .log h3, .legend h3, .controls h3{ margin:8px 0 6px; font-size:13px; color:#cdd5e7; }
  .logBody{ border:1px solid #2b335a; border-radius:6px; padding:8px; background:#0f1220; height:180px; overflow:auto; font-size:12px; color:#c9d2ea; }
  .legend ul{ list-style:none; padding:0; margin:8px 0; display:grid; gap:6px; }
  .dot{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; vertical-align:middle; }
  .idle{ background:#D4D8E3 } .requested{ background:#F5C542 } .confirmed{ background:#51C878 } .failed{ background:#E05A4F } .inhibited{ background:#9B9EAB }
  .links .edge{ display:inline-block; width:24px; height:2px; background:#B4BAC8; margin-right:6px; vertical-align:middle; }
  .links{ grid-template-columns: 1fr 1fr; }
</style>

<header>
  <h1>ReCoN Visualizer</h1>
  <div class="pill {connected ? 'ok' : 'bad'}">{connected ? 'live' : 'offline'}</div>
  </header>

<div class="wrap">
  <aside>
    <Controls
      disabled={!canControl()}
      on:step={() => sendControl("step")}
      on:run={() => sendControl("run")}
      on:pause={() => sendControl("pause")}
      on:reset={() => sendControl("reset")}
      on:exportJSON={exportJSON}
    />

    <div class="legend">
      <h3>Legend</h3>
      <ul>
        <li><span class="dot idle"></span>Idle</li>
        <li><span class="dot requested"></span>Requested</li>
        <li><span class="dot confirmed"></span>Confirmed</li>
        <li><span class="dot failed"></span>Failed</li>
        <li><span class="dot inhibited"></span>Inhibited</li>
      </ul>
      <ul class="links">
        <li><span class="edge"></span>POR</li>
        <li><span class="edge"></span>RET</li>
        <li><span class="edge"></span>SUB</li>
        <li><span class="edge"></span>SUR</li>
        <li><span class="edge"></span>INH</li>
      </ul>
    </div>

    <div class="log">
      <h3>Event log</h3>
      <div class="logBody">
        {#each log as line, i}
          <div>{line}</div>
        {/each}
      </div>
    </div>
  </aside>

  <main>
    <GraphView {graph} {state} />
  </main>
</div>
