(() => {
  const toolbar = document.getElementById('toolbar');
  const searchInput = document.getElementById('searchInput');
  const addNodeBtn = document.getElementById('addNodeBtn');
  const addEdgeBtn = document.getElementById('addEdgeBtn');
  const deleteBtn = document.getElementById('deleteBtn');
  const resetViewBtn = document.getElementById('resetViewBtn');
  const importFile = document.getElementById('importFile');
  const exportBtn = document.getElementById('exportBtn');
  const inspectorForm = document.getElementById('inspector');
  const applyBtn = document.getElementById('applyBtn');

  // Toolbar quick actions
  toolbar.innerHTML = `
    <button id="fitBtn">Fit</button>
    <button id="selectAllBtn">Select All</button>
    <button id="clearSelBtn">Clear Selection</button>
    <button id="invertSelBtn">Invert</button>
  `;

  const elements = [
    // Sample nodes
    { data: { id: 'a', label: 'Alpha', color: '#22d3ee', size: 30, group: 'svc' } },
    { data: { id: 'b', label: 'Beta', color: '#a78bfa', size: 26, group: 'svc' } },
    { data: { id: 'c', label: 'Gamma', color: '#f472b6', size: 24, group: 'db' } },
    { data: { id: 'd', label: 'Delta', color: '#34d399', size: 28, group: 'cache' } },
    // Sample edges
    { data: { id: 'ab', source: 'a', target: 'b', weight: 1.0 } },
    { data: { id: 'bc', source: 'b', target: 'c', weight: 0.5 } },
    { data: { id: 'ac', source: 'a', target: 'c', weight: 0.8 } },
    { data: { id: 'ad', source: 'a', target: 'd', weight: 0.2 } },
  ];

  const cy = cytoscape({
    container: document.getElementById('cy'),
    elements,
    wheelSensitivity: 0.15,
    minZoom: 0.1,
    maxZoom: 3,
    style: [
      {
        selector: 'node',
        style: {
          'background-color': 'data(color)',
          'label': 'data(label)',
          'color': '#e5e7eb',
          'font-size': 12,
          'text-valign': 'center',
          'text-halign': 'center',
          'text-outline-color': '#0b1020',
          'text-outline-width': 2,
          'width': 'mapData(size, 10, 80, 16px, 64px)',
          'height': 'mapData(size, 10, 80, 16px, 64px)',
          'overlay-opacity': 0,
        }
      },
      {
        selector: 'node:selected',
        style: {
          'overlay-color': '#22d3ee',
          'overlay-opacity': 0.2,
          'border-color': '#22d3ee',
          'border-width': 2,
        }
      },
      {
        selector: 'edge',
        style: {
          'width': 'mapData(weight, 0.1, 3, 1px, 5px)',
          'line-color': '#293249',
          'target-arrow-color': '#293249',
          'curve-style': 'bezier',
          'target-arrow-shape': 'triangle',
        }
      },
      {
        selector: 'edge:selected',
        style: { 'line-color': '#22d3ee', 'target-arrow-color': '#22d3ee' }
      },
      { selector: '.faded', style: { 'opacity': 0.08, 'text-opacity': 0.2 } },
    ],
    layout: { name: 'cose-bilkent', animate: true, fit: true, randomize: false }
  });

  // Layout buttons
  document.querySelectorAll('button[data-layout]').forEach(btn => {
    btn.addEventListener('click', () => {
      const name = btn.getAttribute('data-layout');
      cy.layout({ name, animate: true, fit: true }).run();
    });
  });

  // Toolbar actions
  document.getElementById('fitBtn').addEventListener('click', () => cy.fit());
  document.getElementById('selectAllBtn').addEventListener('click', () => cy.elements().select());
  document.getElementById('clearSelBtn').addEventListener('click', () => cy.elements().unselect());
  document.getElementById('invertSelBtn').addEventListener('click', () => {
    const toSelect = cy.elements().difference(cy.elements(':selected'));
    cy.elements().unselect();
    toSelect.select();
  });

  // Search with soft highlight
  searchInput.addEventListener('input', () => {
    const q = searchInput.value.trim().toLowerCase();
    cy.elements().removeClass('faded');
    if (!q) return;
    const matched = cy.nodes().filter(n => (n.data('label') || '').toLowerCase().includes(q));
    cy.elements().addClass('faded');
    matched.neighborhood().add(matched).removeClass('faded');
  });

  // Selection sync to inspector
  function populateInspector(ele) {
    const idInput = inspectorForm.elements['id'];
    const labelInput = inspectorForm.elements['label'];
    const colorInput = inspectorForm.elements['color'];
    const sizeInput = inspectorForm.elements['size'];
    const groupInput = inspectorForm.elements['group'];
    const edgeWeight = inspectorForm.elements['edgeWeight'];

    if (!ele) {
      idInput.value = '';
      labelInput.value = '';
      colorInput.value = '#3b82f6';
      sizeInput.value = 28;
      groupInput.value = '';
      edgeWeight.value = '';
      return;
    }
    if (ele.isNode()) {
      idInput.value = ele.id();
      labelInput.value = ele.data('label') || '';
      colorInput.value = ele.data('color') || '#3b82f6';
      sizeInput.value = ele.data('size') || 28;
      groupInput.value = ele.data('group') || '';
      edgeWeight.value = '';
    } else {
      idInput.value = ele.id();
      labelInput.value = '';
      colorInput.value = '#3b82f6';
      sizeInput.value = 28;
      groupInput.value = '';
      edgeWeight.value = ele.data('weight') ?? '';
    }
  }

  cy.on('select unselect', () => {
    const sel = cy.elements(':selected');
    if (sel.length === 1) populateInspector(sel[0]);
    else populateInspector(null);
  });

  // Apply inspector changes
  applyBtn.addEventListener('click', () => {
    const sel = cy.elements(':selected');
    if (sel.length !== 1) return;
    const ele = sel[0];
    if (ele.isNode()) {
      ele.data({
        label: inspectorForm.elements['label'].value,
        color: inspectorForm.elements['color'].value,
        size: Number(inspectorForm.elements['size'].value) || 28,
        group: inspectorForm.elements['group'].value || undefined,
      });
    } else {
      const w = parseFloat(inspectorForm.elements['edgeWeight'].value);
      if (!Number.isNaN(w)) ele.data('weight', w);
    }
  });

  // Add node
  function addNodeAt(center) {
    const id = uniqueId('n');
    const pos = center || cy.extent().x1 ? cy.center() : undefined;
    const position = center || cy.renderedPosition({ x: window.innerWidth / 2, y: window.innerHeight / 2 });
    const node = cy.add({ group: 'nodes', data: { id, label: id, color: '#22d3ee', size: 28 }, position: cy.project(position) });
    cy.animate({ center: { eles: node }, duration: 250 });
    node.select();
  }
  addNodeBtn.addEventListener('click', () => addNodeAt());

  // Edge creation mode
  let connectMode = false;
  let firstNode = null;
  addEdgeBtn.addEventListener('click', () => {
    connectMode = !connectMode;
    addEdgeBtn.classList.toggle('active', connectMode);
    if (!connectMode) firstNode = null;
  });
  cy.on('tap', 'node', (evt) => {
    const node = evt.target;
    if (!connectMode) return;
    if (!firstNode) {
      firstNode = node;
      node.select();
    } else if (firstNode.id() !== node.id()) {
      const id = uniqueId('e');
      cy.add({ group: 'edges', data: { id, source: firstNode.id(), target: node.id(), weight: 1.0 } });
      firstNode.unselect();
      firstNode = null;
      connectMode = false;
      addEdgeBtn.classList.remove('active');
    }
  });

  // Delete
  deleteBtn.addEventListener('click', () => cy.$(':selected').remove());

  // Reset view
  resetViewBtn.addEventListener('click', () => cy.reset() || cy.fit());

  // Keyboard shortcuts
  document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
      e.preventDefault(); searchInput.focus(); searchInput.select(); return;
    }
    if (e.key === 'Delete' || e.key === 'Backspace') { e.preventDefault(); cy.$(':selected').remove(); }
    if (e.key.toLowerCase() === 'n') { e.preventDefault(); addNodeAt(); }
    if (e.key.toLowerCase() === 'e') { e.preventDefault(); addEdgeBtn.click(); }
    if (e.key === '0') { e.preventDefault(); cy.fit(); }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'f') { e.preventDefault(); searchInput.focus(); searchInput.select(); }
  });

  // Import / Export
  exportBtn.addEventListener('click', () => {
    const json = JSON.stringify(cy.json().elements, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'graph.json'; a.click();
    URL.revokeObjectURL(url);
  });
  importFile.addEventListener('change', async () => {
    const file = importFile.files?.[0];
    if (!file) return;
    const text = await file.text();
    try {
      const els = JSON.parse(text);
      cy.elements().remove();
      cy.add(els);
      cy.layout({ name: 'cose-bilkent', animate: true, fit: true }).run();
    } catch (err) {
      alert('Invalid JSON');
    } finally {
      importFile.value = '';
    }
  });

  // Utilities
  function uniqueId(prefix) {
    const r = Math.random().toString(36).slice(2, 8);
    const id = `${prefix}_${r}`;
    if (cy.getElementById(id).nonempty()) return uniqueId(prefix);
    return id;
  }
})();

