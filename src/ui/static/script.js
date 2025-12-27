
document.addEventListener('DOMContentLoaded', () => {

    // --- References ---
    const topKSlider = document.getElementById('top-k');
    const denseWSlider = document.getElementById('dense-weight');
    const bm25WSlider = document.getElementById('bm25-weight');
    const threshSlider = document.getElementById('threshold');

    // Sliders Listeners
    const bindVal = (sliderId, displayId) => {
        const slider = document.getElementById(sliderId);
        const display = document.getElementById(displayId);
        slider.addEventListener('input', (e) => display.textContent = e.target.value);
    };

    bindVal('top-k', 'top-k-val');
    bindVal('dense-weight', 'dense-w-val');
    bindVal('bm25-weight', 'bm25-w-val');
    bindVal('threshold', 'conf-val');

    // GNN Params Listeners
    bindVal('gnn-layers', 'gnn-layers-val');
    bindVal('gnn-heads', 'gnn-heads-val');
    bindVal('gnn-dropout', 'gnn-drop-val');

    document.getElementById('gnn-dim').addEventListener('change', (e) => {
        document.getElementById('gnn-dim-val').textContent = e.target.value;
    });

    // Retrieval Method Visibility Logic
    const methodSelect = document.getElementById('retrieval-method');
    const hybridControls = document.querySelectorAll('.hybrid-only');

    methodSelect.addEventListener('change', (e) => {
        const method = e.target.value;
        if (method === 'hybrid') {
            hybridControls.forEach(el => el.style.display = 'block');
        } else {
            hybridControls.forEach(el => el.style.display = 'none');
        }
    });

    // Verification Mode Visibility Logic
    const verRadios = document.querySelectorAll('input[name="ver-mode"]');
    const gnnParams = document.getElementById('gnn-params');
    const nliParams = document.getElementById('nli-params');

    verRadios.forEach(radio => {
        radio.addEventListener('change', (e) => {
            if (e.target.value === 'gnn' || e.target.value === 'ensemble') {
                gnnParams.style.display = 'block';
                nliParams.style.display = (e.target.value === 'ensemble') ? 'block' : 'none';
            } else {
                gnnParams.style.display = 'none';
                nliParams.style.display = 'block';
            }
        });
    });

    function drawGraph(graphData) {
        if (!graphData) {
            document.getElementById('graph-section').classList.add('hidden');
            return;
        }
        document.getElementById('graph-section').classList.remove('hidden');

        const canvas = document.getElementById('graphCanvas');
        const ctx = canvas.getContext('2d');
        const container = document.getElementById('graph-viz-container');
        const tooltip = document.getElementById('graph-tooltip');

        // Set canvas size
        canvas.width = container.clientWidth;
        canvas.height = container.clientHeight;

        const nodes = graphData.nodes;
        const edges = graphData.edges;
        const width = canvas.width;
        const height = canvas.height;
        const centerX = width / 2;
        const centerY = height / 2;
        const radius = Math.min(width, height) * 0.35;

        // Position nodes
        const positions = nodes.map((node, i) => {
            if (i === 0) return { x: centerX, y: centerY };
            const angle = (2 * Math.PI * (i - 1)) / (nodes.length - 1);
            return {
                x: centerX + radius * Math.cos(angle),
                y: centerY + radius * Math.sin(angle)
            };
        });

        const draw = (hoverIdx = -1) => {
            ctx.clearRect(0, 0, width, height);

            // 1. Draw Edges
            edges.forEach(edge => {
                const start = positions[edge.source];
                const end = positions[edge.target];
                if (!start || !end) return;

                ctx.beginPath();
                ctx.moveTo(start.x, start.y);
                ctx.lineTo(end.x, end.y);

                const isRelated = (edge.source === hoverIdx || edge.target === hoverIdx);
                const alpha = isRelated ? 0.8 : Math.max(0.1, edge.weight);
                ctx.strokeStyle = `rgba(99, 102, 241, ${alpha})`;
                ctx.lineWidth = (isRelated ? 3 : 1) + edge.weight * 8;
                ctx.stroke();
            });

            // 2. Draw Nodes
            positions.forEach((pos, i) => {
                const isHovered = (i === hoverIdx);
                const size = isHovered ? 16 : 12;

                // Glow for claim
                if (i === 0) {
                    const gradient = ctx.createRadialGradient(pos.x, pos.y, 0, pos.x, pos.y, size * 2);
                    gradient.addColorStop(0, 'rgba(99, 102, 241, 0.3)');
                    gradient.addColorStop(1, 'rgba(99, 102, 241, 0)');
                    ctx.fillStyle = gradient;
                    ctx.beginPath();
                    ctx.arc(pos.x, pos.y, size * 2, 0, 2 * Math.PI);
                    ctx.fill();
                }

                ctx.beginPath();
                ctx.arc(pos.x, pos.y, size, 0, 2 * Math.PI);

                if (nodes[i].label === 'Claim') {
                    ctx.fillStyle = '#6366f1';
                } else if (nodes[i].label === 'Entity') {
                    ctx.fillStyle = isHovered ? '#34d399' : '#10b981';
                } else {
                    ctx.fillStyle = isHovered ? '#8b5cf6' : '#1e293b';
                }

                ctx.fill();
                ctx.strokeStyle = '#f8fafc';
                ctx.lineWidth = isHovered ? 3 : 2;
                ctx.stroke();

                // Label
                ctx.fillStyle = isHovered ? '#fff' : '#94a3b8';
                ctx.font = isHovered ? 'bold 12px Outfit' : '10px Outfit';
                ctx.textAlign = 'center';

                let shortLabel = '';
                if (nodes[i].label === 'Claim') shortLabel = 'CLAIM';
                else if (nodes[i].label === 'Entity') shortLabel = 'ENT';
                else shortLabel = `E${i}`;

                ctx.fillText(shortLabel, pos.x, pos.y + size + 15);
            });
        };

        // Interaction
        canvas.onmousemove = (e) => {
            const rect = canvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;

            let found = -1;
            positions.forEach((pos, i) => {
                const dx = pos.x - mouseX;
                const dy = pos.y - mouseY;
                if (Math.sqrt(dx * dx + dy * dy) < 20) {
                    found = i;
                }
            });

            if (found !== -1) {
                draw(found);
                tooltip.style.display = 'block';
                tooltip.style.left = (mouseX + 15) + 'px';
                tooltip.style.top = (mouseY + 15) + 'px';

                const node = nodes[found];
                const text = node.text ? (node.text.length > 150 ? node.text.substring(0, 150) + '...' : node.text) : 'Evidence sentence';
                tooltip.innerHTML = `<strong>${node.label}</strong><br>${text}`;
            } else {
                draw();
                tooltip.style.display = 'none';
            }
        };

        canvas.onmouseleave = () => {
            draw();
            tooltip.style.display = 'none';
        };

        draw();

        // Handle resize
        window.addEventListener('resize', () => {
            canvas.width = container.clientWidth;
            canvas.height = container.clientHeight;
            draw();
        });
    }


    // Chart Instance
    let chartInstance = null;

    // --- Verify Action ---
    const verifyBtn = document.getElementById('verify-btn');
    const loader = document.querySelector('.loader');

    verifyBtn.addEventListener('click', async () => {
        const claim = document.getElementById('claim-input').value.trim();
        if (!claim) return alert("Please enter a claim.");

        // UI State
        verifyBtn.disabled = true;
        loader.style.display = 'inline-block';
        document.getElementById('results-area').classList.add('hidden');
        document.getElementById('results-area').classList.remove('visible');

        // Build Config Payload
        const payload = {
            claim: claim,
            retrieval: {
                method: document.getElementById('retrieval-method').value,
                top_k: parseInt(document.getElementById('top-k').value),
                bm25_weight: parseFloat(document.getElementById('bm25-weight').value),
                dense_weight: parseFloat(document.getElementById('dense-weight').value)
            },
            verification: {
                model_type: document.querySelector('input[name="ver-mode"]:checked').value,
                threshold: parseFloat(document.getElementById('threshold').value),
                gnn_layers: parseInt(document.getElementById('gnn-layers').value),
                gnn_heads: parseInt(document.getElementById('gnn-heads').value),
                gnn_hidden_dim: parseInt(document.getElementById('gnn-dim').value),
                gnn_dropout: parseFloat(document.getElementById('gnn-dropout').value),
                aggregation_strategy: document.getElementById('agg-strategy').value
            }
        };

        try {
            const response = await fetch('/verify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!response.ok) throw new Error("Verification failed");

            const result = await response.json();
            renderResults(result);

        } catch (err) {
            alert("Error: " + err.message);
        } finally {
            verifyBtn.disabled = false;
            loader.style.display = 'none';
        }
    });

    // --- Reset Defaults ---
    const resetBtn = document.getElementById('reset-btn');

    resetBtn.addEventListener('click', async () => {
        try {
            const response = await fetch('/defaults');
            if (!response.ok) throw new Error("Could not fetch defaults");
            const data = await response.json();

            // Apply Retrieval Settings
            methodSelect.value = data.retrieval.method;
            methodSelect.dispatchEvent(new Event('change')); // Trigger visibility logic

            const setSlider = (id, val, displayId) => {
                document.getElementById(id).value = val;
                document.getElementById(displayId).textContent = val;
            };

            setSlider('top-k', data.retrieval.top_k, 'top-k-val');
            setSlider('bm25-weight', data.retrieval.bm25_weight, 'bm25-w-val');
            setSlider('dense-weight', data.retrieval.dense_weight, 'dense-w-val');

            // Apply Verification Settings
            document.querySelector(`input[name="ver-mode"][value="${data.verification.model_type}"]`).checked = true;
            // Trigger radio change for GNN params visibility
            document.querySelector(`input[name="ver-mode"][value="${data.verification.model_type}"]`).dispatchEvent(new Event('change'));

            setSlider('threshold', data.verification.threshold, 'conf-val');

            // GNN Resets
            setSlider('gnn-layers', data.verification.gnn_layers, 'gnn-layers-val');
            setSlider('gnn-heads', data.verification.gnn_heads, 'gnn-heads-val');
            setSlider('gnn-dropout', data.verification.gnn_dropout, 'gnn-drop-val');
            document.getElementById('gnn-dim').value = data.verification.gnn_hidden_dim;
            document.getElementById('gnn-dim-val').textContent = data.verification.gnn_hidden_dim;

            // Aggregation Reset
            document.getElementById('agg-strategy').value = data.verification.aggregation_strategy;

        } catch (err) {
            console.error(err);
            alert("Failed to reset defaults");
        }
    });

    function renderResults(res) {
        const resultsArea = document.getElementById('results-area');
        const verdictCard = document.getElementById('verdict-card');

        // 1. Verdict
        const label = res.predicted_label;
        document.getElementById('prediction-text').textContent = label;

        verdictCard.className = 'verdict-card'; // Reset
        if (label === 'SUPPORTS') verdictCard.classList.add('supports');
        else if (label === 'REFUTES') verdictCard.classList.add('refutes');
        else verdictCard.classList.add('mixed');

        // 2. Confidence
        const confPerc = Math.round(res.confidence * 100);
        document.getElementById('conf-text').textContent = `${confPerc}% Confidence`;
        document.getElementById('conf-fill').style.width = `${confPerc}%`;

        // 3. Evidence
        const evList = document.getElementById('evidence-list');
        evList.innerHTML = '';
        if (res.evidence.length === 0) {
            evList.innerHTML = '<div class="evidence-item">No sufficient evidence found matching your criteria.</div>';
        } else {
            res.evidence.forEach(item => {
                const el = document.createElement('div');
                el.className = 'evidence-item';
                el.innerHTML = `
                   <div class="evidence-id">Doc ${item.doc_id} <span class="evidence-score">${item.score.toFixed(3)}</span></div>
                   <p>${item.text}</p>
                `;
                evList.appendChild(el);
            });
        }

        // 4. Debug Docs
        const docList = document.getElementById('docs-list');
        docList.innerHTML = '';
        res.retrieved_docs.forEach(doc => {
            const el = document.createElement('div');
            el.className = 'doc-item';
            el.innerHTML = `<strong>${doc.title}</strong> (Score: ${doc.score.toFixed(4)})`;
            docList.appendChild(el);
        });

        // 5. Render Chart
        renderChart(res.retrieved_docs);

        // Show
        resultsArea.classList.remove('hidden');
        // Small delay for transition
        setTimeout(() => resultsArea.classList.add('visible'), 50);

        // 6. Draw Graph Attention Map
        if (res.graph_data) {
            setTimeout(() => drawGraph(res.graph_data), 100);
        } else {
            document.getElementById('graph-section').classList.add('hidden');
        }

        // 7. Display LLM Explanation
        const explanationSection = document.getElementById('explanation-section');
        const explanationText = document.getElementById('explanation-text');
        if (res.explanation) {
            explanationText.textContent = res.explanation;
            explanationSection.classList.remove('hidden');
        } else {
            explanationSection.classList.add('hidden');
        }
    }

    function renderChart(docs) {
        const ctx = document.getElementById('scoresChart').getContext('2d');

        if (chartInstance) {
            chartInstance.destroy();
        }

        const labels = docs.map((d, i) => `Doc ${i + 1}`);
        const data = docs.map(d => d.score);

        chartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Retrieval Score',
                    data: data,
                    backgroundColor: 'rgba(99, 102, 241, 0.6)',
                    borderColor: 'rgba(99, 102, 241, 1)',
                    borderWidth: 1,
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: { color: 'rgba(255,255,255,0.1)' },
                        ticks: { color: '#94a3b8' }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: '#94a3b8' }
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }


});
