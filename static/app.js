// Ceres AI - Dashboard Client Script

let chart = null;
let currentRecordingState = false;
let lastDamagedCount = 0;

document.addEventListener("DOMContentLoaded", () => {
    // Inicializar Gráficos
    initChart();
    
    // Atualização periódica rápida (a cada 500ms para parecer tempo real)
    pollMetrics();
    setInterval(pollMetrics, 500);

    // Event Listeners dos Botões
    document.getElementById("btn-record").addEventListener("click", toggleRecording);
    document.getElementById("btn-mock-damaged").addEventListener("click", triggerManualEjection);
    document.getElementById("btn-clear-db").addEventListener("click", clearMetrics);
    
    // Configura grupo inicial no backend ao mudar o select
    document.getElementById("group-select").addEventListener("change", (e) => {
        setGroup(e.target.value);
    });
});

// Inicialização do Gráfico Chart.js
function initChart() {
    const ctx = document.getElementById('metricsChart').getContext('2d');
    chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: [], // Nomes dos Grupos/Lotes
            datasets: [
                {
                    label: 'Saudáveis',
                    backgroundColor: '#10b981',
                    borderColor: '#10b981',
                    borderWidth: 1,
                    data: []
                },
                {
                    label: 'Estragados',
                    backgroundColor: '#f43f5e',
                    borderColor: '#f43f5e',
                    borderWidth: 1,
                    data: []
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: {
                        color: '#9ca3af',
                        font: { family: 'Outfit' }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9ca3af', font: { family: 'Outfit' } }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9ca3af', font: { family: 'Outfit' } },
                    beginAtZero: true
                }
            }
        }
    });
}

// Busca as métricas e o histórico do backend
async function pollMetrics() {
    try {
        // Busca métricas gerais e de grupo
        const response = await fetch('/api/metrics');
        const data = await response.json();
        
        // Atualiza contadores principais
        document.getElementById("count-healthy").innerText = data.summary.healthy;
        document.getElementById("count-damaged").innerText = data.summary.damaged;
        document.getElementById("count-total").innerText = data.summary.total;
        
        // Calcula taxa de eficiência
        const efficiency = data.summary.total > 0 
            ? ((data.summary.healthy / data.summary.total) * 100).toFixed(1) + "%" 
            : "100%";
        document.getElementById("rate-efficiency").innerText = efficiency;

        // Animação da válvula se a contagem de estragados aumentou (indica ejeção recente)
        if (data.summary.damaged > lastDamagedCount) {
            flashValveIndicator();
        }
        lastDamagedCount = data.summary.damaged;

        // Atualiza estado de gravação no botão se mudou no servidor
        updateRecordingUI(data.recording);

        // Atualiza o gráfico de grupos
        updateChartData(data.groups);

        // Busca o histórico de log recente
        pollHistory();
        
    } catch (error) {
        console.error("Erro ao buscar métricas:", error);
    }
}

// Atualiza os dados do gráfico
function updateChartData(groupsData) {
    if (!chart) return;
    
    const labels = Object.keys(groupsData);
    const healthyData = [];
    const damagedData = [];
    
    labels.forEach(group => {
        healthyData.push(groupsData[group].healthy || 0);
        damagedData.push(groupsData[group].damaged || 0);
    });
    
    chart.data.labels = labels;
    chart.data.datasets[0].data = healthyData;
    chart.data.datasets[1].data = damagedData;
    chart.update();
}

// Busca o histórico recente de detecção
async function pollHistory() {
    try {
        const response = await fetch('/api/recent_history');
        const history = await response.json();
        
        const tbody = document.getElementById("log-table-body");
        if (history.length === 0) {
            tbody.innerHTML = `<tr><td colspan="5" class="empty-state">Nenhum grão avaliado ainda neste lote.</td></tr>`;
            return;
        }
        
        let html = "";
        history.forEach(item => {
            const timeFormatted = new Date(item.timestamp).toLocaleTimeString('pt-BR');
            const isDamaged = item.status === "damaged";
            const badgeClass = isDamaged ? "badge-status damaged" : "badge-status healthy";
            const labelText = isDamaged ? "ESTRAGADO" : "SAUDÁVEL";
            const actionText = isDamaged ? "💨 JATO DE AR ACIONADO" : "✔️ Aprovado";
            
            html += `
                <tr>
                    <td>${timeFormatted}</td>
                    <td>${item.group_name}</td>
                    <td><span class="${badgeClass}">${labelText}</span></td>
                    <td>${item.confidence}%</td>
                    <td><span class="badge-action">${actionText}</span></td>
                </tr>
            `;
        });
        tbody.innerHTML = html;
    } catch (error) {
        console.error("Erro ao buscar histórico:", error);
    }
}

// Configura o grupo ativo no backend
async function setGroup(groupName) {
    try {
        await fetch('/api/set_group', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ group_name: groupName })
        });
    } catch (error) {
        console.error("Erro ao definir grupo:", error);
    }
}

// Inicia/Para gravação de vídeo
async function toggleRecording() {
    const endpoint = currentRecordingState ? '/api/stop_recording' : '/api/start_recording';
    try {
        const response = await fetch(endpoint, { method: 'POST' });
        const data = await response.json();
        if (data.status === "success") {
            currentRecordingState = !currentRecordingState;
            updateRecordingUI(currentRecordingState);
        }
    } catch (error) {
        console.error("Erro ao alternar gravação:", error);
    }
}

// Atualiza o estado visual da gravação
function updateRecordingUI(isRecording) {
    currentRecordingState = isRecording;
    const btn = document.getElementById("btn-record");
    const badge = document.getElementById("recording-badge");
    
    if (isRecording) {
        btn.innerHTML = `<span class="btn-icon">⏹</span> Parar Gravação`;
        btn.classList.remove("btn-primary");
        btn.classList.add("btn-danger");
        badge.style.display = "flex";
    } else {
        btn.innerHTML = `<span class="btn-icon">⏺</span> Iniciar Gravação`;
        btn.classList.remove("btn-danger");
        btn.classList.add("btn-primary");
        badge.style.display = "none";
    }
}

// Simula ejeção manual para testes de maquinário
async function triggerManualEjection() {
    try {
        const group = document.getElementById("group-select").value;
        const response = await fetch('/api/trigger_ejection', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ group_name: group })
        });
        const data = await response.json();
        if (data.status === "success") {
            flashValveIndicator();
        }
    } catch (error) {
        console.error("Erro ao ejetar manualmente:", error);
    }
}

// Limpa métricas do banco de dados
async function clearMetrics() {
    if (!confirm("Tem certeza que deseja apagar todo o histórico de contagem e gráficos?")) return;
    try {
        await fetch('/api/clear_metrics', { method: 'POST' });
        pollMetrics();
    } catch (error) {
        console.error("Erro ao limpar métricas:", error);
    }
}

// Executa animações de acionamento do solenoide
function flashValveIndicator() {
    // 1. Overlay visual no vídeo
    const overlay = document.getElementById("valve-overlay");
    overlay.classList.add("active");
    
    // 2. Luz de status no painel de controle
    const light = document.getElementById("valve-status-light");
    light.classList.add("active");
    light.innerText = "ATIVO (JATO)";

    // Remove as classes de ativação após o pulso (ex: 250ms na tela para visibilidade)
    setTimeout(() => {
        overlay.classList.remove("active");
        light.classList.remove("active");
        light.innerText = "Inativo";
    }, 250);
}
