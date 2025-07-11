// Neural Hivemind Web Interface JavaScript

class NeuralHivemindUI {
    constructor() {
        this.socket = null;
        this.currentTab = 'dashboard';
        this.isInitialized = false;
        this.performanceChart = null;
        this.networkGraph = null;
        
        this.init();
    }
    
    init() {
        // Initialize Socket.IO connection
        this.initSocketConnection();
        
        // Setup event listeners
        this.setupEventListeners();
        
        // Initialize charts
        this.initCharts();
        
        // Load initial data
        this.loadStatus();
    }
    
    initSocketConnection() {
        this.socket = io();
        
        this.socket.on('connect', () => {
            console.log('Connected to server');
            this.updateConnectionStatus(true);
        });
        
        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
            this.updateConnectionStatus(false);
        });
        
        this.socket.on('log_update', (data) => {
            this.addLogEntry(data);
        });
        
        this.socket.on('progress_update', (data) => {
            this.updateProgress(data);
        });
        
        this.socket.on('initialization_complete', (data) => {
            this.handleInitializationComplete(data);
        });
        
        this.socket.on('optimization_complete', (data) => {
            this.handleOptimizationComplete(data);
        });
    }
    
    setupEventListeners() {
        // Navigation
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const tab = e.currentTarget.dataset.tab;
                this.switchTab(tab);
            });
        });
        
        // Initialize button
        document.getElementById('initialize-btn').addEventListener('click', () => {
            this.showInitModal();
        });
        
        // Init form
        document.getElementById('init-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.initializeHivemind();
        });
        
        document.getElementById('cancel-init').addEventListener('click', () => {
            this.hideModal('init-modal');
        });
        
        // File upload
        const uploadArea = document.getElementById('upload-area');
        const fileInput = document.getElementById('file-input');
        
        uploadArea.addEventListener('click', () => fileInput.click());
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('drag-over');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('drag-over');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('drag-over');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileUpload(files[0]);
            }
        });
        
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFileUpload(e.target.files[0]);
            }
        });
        
        // Analyze button
        document.getElementById('analyze-btn').addEventListener('click', () => {
            this.analyzeImage();
        });
        
        // Optimization
        document.getElementById('start-optimization').addEventListener('click', () => {
            this.startOptimization();
        });
        
        // Script search
        document.getElementById('script-search').addEventListener('input', (e) => {
            this.filterScripts(e.target.value);
        });
        
        // Network reset
        document.getElementById('reset-network').addEventListener('click', () => {
            if (this.networkGraph) {
                this.networkGraph.fit();
            }
        });
        
        // Clear logs
        document.getElementById('clear-logs').addEventListener('click', () => {
            document.getElementById('log-viewer').innerHTML = '';
        });
    }
    
    switchTab(tabName) {
        // Update nav
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
        
        // Update content
        document.querySelectorAll('.tab-content').forEach(tab => {
            tab.classList.remove('active');
        });
        document.getElementById(`${tabName}-tab`).classList.add('active');
        
        this.currentTab = tabName;
        
        // Load tab-specific data
        if (tabName === 'scripts' && this.isInitialized) {
            this.loadScripts();
        } else if (tabName === 'network' && this.isInitialized) {
            this.loadNetworkGraph();
        } else if (tabName === 'logs') {
            this.loadLogs();
        }
    }
    
    updateConnectionStatus(connected) {
        const indicator = document.getElementById('connection-status');
        const text = document.getElementById('connection-text');
        
        if (connected) {
            indicator.classList.add('connected');
            text.textContent = 'Connected';
        } else {
            indicator.classList.remove('connected');
            text.textContent = 'Disconnected';
        }
    }
    
    showInitModal() {
        document.getElementById('init-modal').classList.add('active');
    }
    
    hideModal(modalId) {
        document.getElementById(modalId).classList.remove('active');
    }
    
    showProgressModal(title) {
        document.getElementById('progress-title').textContent = title;
        document.getElementById('progress-modal').classList.add('active');
    }
    
    async initializeHivemind() {
        const formData = {
            focus_area: document.getElementById('focus-area').value,
            optimization: document.getElementById('optimization-type').value,
            max_threads: parseInt(document.getElementById('max-threads').value)
        };
        
        this.hideModal('init-modal');
        this.showProgressModal('Initializing Neural Hivemind...');
        
        try {
            const response = await fetch('/api/initialize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });
            
            const data = await response.json();
            
            if (data.status === 'initializing') {
                // Wait for socket events for progress
            }
        } catch (error) {
            console.error('Initialization error:', error);
            this.hideModal('progress-modal');
            alert('Failed to initialize hivemind');
        }
    }
    
    handleInitializationComplete(data) {
        this.hideModal('progress-modal');
        
        if (data.status === 'success') {
            this.isInitialized = true;
            document.getElementById('initialize-btn').textContent = 'Reinitialize';
            this.updateMetrics(data.metrics);
            this.addLogEntry({
                level: 'INFO',
                message: 'Neural Hivemind initialized successfully!'
            });
        } else {
            alert('Initialization failed: ' + data.error);
        }
    }
    
    async loadStatus() {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();
            
            if (data.status === 'ready') {
                this.isInitialized = true;
                document.getElementById('initialize-btn').textContent = 'Reinitialize';
                this.updateMetrics(data.metrics);
            }
        } catch (error) {
            console.error('Failed to load status:', error);
        }
    }
    
    updateMetrics(metrics) {
        document.getElementById('total-scripts').textContent = metrics.total_scripts || 0;
        document.getElementById('total-images').textContent = metrics.total_images || 0;
        document.getElementById('total-data').textContent = metrics.total_data_files || 0;
        document.getElementById('total-connections').textContent = metrics.total_connections || 0;
        
        // Update top combinations
        const combinationsList = document.getElementById('combinations-list');
        if (metrics.top_combinations && metrics.top_combinations.length > 0) {
            combinationsList.innerHTML = metrics.top_combinations.map(combo => `
                <div class="combination-item">
                    <div class="combination-scripts">
                        ${combo.scripts.map(script => `<span class="script-tag">${script}</span>`).join('')}
                    </div>
                    <div class="combination-score">${(combo.score * 100).toFixed(1)}%</div>
                </div>
            `).join('');
        }
        
        // Update performance chart
        if (metrics.optimization_runs > 0) {
            this.updatePerformanceChart(metrics);
        }
    }
    
    initCharts() {
        const ctx = document.getElementById('performance-chart').getContext('2d');
        this.performanceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Performance Score',
                    data: [],
                    borderColor: '#00d4ff',
                    backgroundColor: 'rgba(0, 212, 255, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1,
                        ticks: {
                            callback: function(value) {
                                return (value * 100) + '%';
                            }
                        }
                    }
                }
            }
        });
    }
    
    updatePerformanceChart(metrics) {
        // Add dummy data for demonstration
        const runs = metrics.optimization_runs || 0;
        const labels = Array.from({length: runs}, (_, i) => `Run ${i + 1}`);
        const data = Array.from({length: runs}, () => Math.random() * 0.3 + 0.6);
        
        this.performanceChart.data.labels = labels;
        this.performanceChart.data.datasets[0].data = data;
        this.performanceChart.update();
    }
    
    async loadScripts() {
        try {
            const response = await fetch('/api/scripts');
            const data = await response.json();
            
            const tbody = document.getElementById('scripts-tbody');
            if (data.scripts.length === 0) {
                tbody.innerHTML = '<tr><td colspan="7" class="empty-state">No scripts found</td></tr>';
                return;
            }
            
            tbody.innerHTML = data.scripts.map(script => `
                <tr>
                    <td>${script.name}</td>
                    <td>${script.directory}</td>
                    <td>${script.functions}</td>
                    <td>${script.classes}</td>
                    <td>${script.parameters}</td>
                    <td>
                        <span class="status-badge ${script.has_error ? 'error' : 'success'}">
                            ${script.has_error ? 'Error' : 'OK'}
                        </span>
                    </td>
                    <td>
                        <button class="btn btn-secondary btn-sm" onclick="app.viewScriptDetails('${script.path}')">
                            <i class="fas fa-eye"></i> View
                        </button>
                    </td>
                </tr>
            `).join('');
        } catch (error) {
            console.error('Failed to load scripts:', error);
        }
    }
    
    async viewScriptDetails(scriptPath) {
        try {
            const response = await fetch(`/api/script${scriptPath}`);
            const data = await response.json();
            
            console.log('Script details:', data);
            // TODO: Show script details in a modal
        } catch (error) {
            console.error('Failed to load script details:', error);
        }
    }
    
    filterScripts(searchTerm) {
        const rows = document.querySelectorAll('#scripts-tbody tr');
        const term = searchTerm.toLowerCase();
        
        rows.forEach(row => {
            const text = row.textContent.toLowerCase();
            row.style.display = text.includes(term) ? '' : 'none';
        });
    }
    
    async loadNetworkGraph() {
        try {
            const response = await fetch('/api/network-graph');
            const data = await response.json();
            
            const container = document.getElementById('network-visualization');
            
            const options = {
                nodes: {
                    shape: 'dot',
                    size: 16,
                    font: {
                        size: 12,
                        color: '#e0e0e0'
                    },
                    borderWidth: 2,
                    color: {
                        background: '#00d4ff',
                        border: '#0099cc',
                        highlight: {
                            background: '#00ff88',
                            border: '#00cc66'
                        }
                    }
                },
                edges: {
                    color: {
                        color: '#666',
                        highlight: '#00d4ff'
                    },
                    arrows: {
                        to: {
                            enabled: true,
                            scaleFactor: 0.5
                        }
                    }
                },
                physics: {
                    stabilization: {
                        iterations: 200
                    },
                    barnesHut: {
                        gravitationalConstant: -30000,
                        springConstant: 0.001,
                        damping: 0.09
                    }
                },
                interaction: {
                    hover: true,
                    tooltipDelay: 200
                }
            };
            
            // Create network
            this.networkGraph = new vis.Network(container, data, options);
            
            // Add click handler
            this.networkGraph.on('click', (params) => {
                if (params.nodes.length > 0) {
                    const nodeId = params.nodes[0];
                    this.viewScriptDetails(nodeId);
                }
            });
        } catch (error) {
            console.error('Failed to load network graph:', error);
        }
    }
    
    handleFileUpload(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file');
            return;
        }
        
        this.uploadedFile = file;
        document.getElementById('analyze-btn').disabled = false;
        
        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            // Could add image preview here
        };
        reader.readAsDataURL(file);
        
        this.addLogEntry({
            level: 'INFO',
            message: `Uploaded file: ${file.name}`
        });
    }
    
    async analyzeImage() {
        if (!this.uploadedFile) return;
        
        const formData = new FormData();
        formData.append('image', this.uploadedFile);
        formData.append('task', document.getElementById('analysis-task').value);
        
        document.getElementById('analyze-btn').disabled = true;
        
        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            const resultsDiv = document.getElementById('analysis-results');
            resultsDiv.innerHTML = `
                <h3>Analysis Results</h3>
                <div class="result-item">
                    <strong>Execution Plan:</strong>
                    <div class="combination-scripts">
                        ${data.execution_plan.map(script => `<span class="script-tag">${script}</span>`).join('')}
                    </div>
                </div>
                <div class="result-item">
                    <strong>Execution Time:</strong> ${data.results.execution_time.toFixed(2)}s
                </div>
                <div class="result-item">
                    <strong>Outputs Generated:</strong> ${data.results.outputs}
                </div>
                <div class="result-item">
                    <strong>Errors:</strong> ${data.results.errors}
                </div>
            `;
        } catch (error) {
            console.error('Analysis error:', error);
            alert('Failed to analyze image');
        } finally {
            document.getElementById('analyze-btn').disabled = false;
        }
    }
    
    async startOptimization() {
        const epochs = parseInt(document.getElementById('epochs-input').value);
        
        document.getElementById('start-optimization').disabled = true;
        document.getElementById('optimization-status').textContent = 'Optimizing...';
        
        try {
            const response = await fetch('/api/optimize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ epochs })
            });
            
            const data = await response.json();
            
            if (data.status === 'optimizing') {
                // Wait for socket events
            }
        } catch (error) {
            console.error('Optimization error:', error);
            alert('Failed to start optimization');
            document.getElementById('start-optimization').disabled = false;
        }
    }
    
    handleOptimizationComplete(data) {
        document.getElementById('start-optimization').disabled = false;
        
        if (data.status === 'success') {
            document.getElementById('optimization-status').textContent = 'Optimization complete!';
            this.updateMetrics(data.metrics);
            this.addLogEntry({
                level: 'INFO',
                message: 'Network optimization completed successfully!'
            });
        } else {
            document.getElementById('optimization-status').textContent = 'Optimization failed';
            alert('Optimization failed: ' + data.error);
        }
    }
    
    updateProgress(data) {
        const progressFill = document.getElementById('progress-fill');
        const progressText = document.getElementById('progress-text');
        
        if (progressFill) {
            progressFill.style.width = `${data.percentage}%`;
        }
        
        if (progressText) {
            progressText.textContent = `${data.task}: ${Math.round(data.percentage)}%`;
        }
        
        // Update optimization progress if applicable
        if (data.task === 'optimization') {
            const optProgressFill = document.getElementById('optimization-progress-fill');
            if (optProgressFill) {
                optProgressFill.style.width = `${data.percentage}%`;
            }
        }
    }
    
    addLogEntry(data) {
        const logViewer = document.getElementById('log-viewer');
        const timestamp = new Date(data.timestamp || Date.now()).toLocaleTimeString();
        
        const entry = document.createElement('div');
        entry.className = 'log-entry';
        entry.innerHTML = `
            <span class="log-time">${timestamp}</span>
            <span class="log-level ${data.level}">${data.level}</span>
            <span class="log-message">${data.message}</span>
        `;
        
        logViewer.appendChild(entry);
        logViewer.scrollTop = logViewer.scrollHeight;
    }
    
    async loadLogs() {
        try {
            const response = await fetch('/api/logs');
            const data = await response.json();
            
            const logViewer = document.getElementById('log-viewer');
            logViewer.innerHTML = '';
            
            data.logs.forEach(log => {
                // Parse log format
                const match = log.match(/(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - \w+ - (\w+) - (.+)/);
                if (match) {
                    this.addLogEntry({
                        timestamp: match[1],
                        level: match[2],
                        message: match[3]
                    });
                }
            });
        } catch (error) {
            console.error('Failed to load logs:', error);
        }
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new NeuralHivemindUI();
});