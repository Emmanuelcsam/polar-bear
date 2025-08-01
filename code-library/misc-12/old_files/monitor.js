const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const orchestrator = require('./orchestrator');
const path = require('path');

const app = express();
const server = http.createServer(app);
const io = socketIo(server);

// Serve static files
app.use(express.static('public'));
app.use(express.json());

// Store connected clients
const clients = new Set();

// WebSocket connection handling
io.on('connection', (socket) => {
    console.log('Client connected:', socket.id);
    clients.add(socket);
    
    // Send initial status
    socket.emit('status', {
        processes: orchestrator.getProcessStatus(),
        outputs: orchestrator.getAllProcessOutputs(),
        isRunning: orchestrator.isRunning,
        systemInfo: orchestrator.getSystemInfo()
    });
    
    socket.on('disconnect', () => {
        console.log('Client disconnected:', socket.id);
        clients.delete(socket);
    });
    
    // Handle control commands
    socket.on('startAll', async () => {
        try {
            await orchestrator.startAllProcesses();
            socket.emit('commandResult', { success: true, message: 'All processes started with Pylon Viewer integration' });
        } catch (error) {
            socket.emit('commandResult', { success: false, message: error.message });
        }
    });
    
    socket.on('stopAll', () => {
        orchestrator.stopAllProcesses();
        socket.emit('commandResult', { success: true, message: 'All processes stopped including Pylon Viewer' });
    });
    
    socket.on('restartProcess', (scriptName) => {
        orchestrator.restartProcess(scriptName);
        socket.emit('commandResult', { success: true, message: `${scriptName} restart initiated` });
    });
    
    socket.on('getStatus', () => {
        socket.emit('status', {
            processes: orchestrator.getProcessStatus(),
            outputs: orchestrator.getAllProcessOutputs(),
            isRunning: orchestrator.isRunning,
            systemInfo: orchestrator.getSystemInfo()
        });
    });
    
    // Handle circle overlay controls
    socket.on('circleControl', (command) => {
        // This would send commands to the main.py process
        console.log(`Circle control command: ${command}`);
        socket.emit('commandResult', { success: true, message: `Circle control: ${command}` });
    });
});

// Broadcast status updates to all connected clients
function broadcastStatus() {
    const status = {
        processes: orchestrator.getProcessStatus(),
        outputs: orchestrator.getAllProcessOutputs(),
        isRunning: orchestrator.isRunning,
        systemInfo: orchestrator.getSystemInfo(),
        timestamp: new Date().toISOString()
    };
    
    for (const client of clients) {
        client.emit('status', status);
    }
}

// Set up event listeners for orchestrator
orchestrator.on('processOutput', () => {
    broadcastStatus();
});

orchestrator.on('processClosed', () => {
    broadcastStatus();
});

orchestrator.on('allProcessesStarted', () => {
    broadcastStatus();
});

orchestrator.on('allProcessesStopped', () => {
    broadcastStatus();
});

// API endpoints
app.get('/api/status', (req, res) => {
    res.json({
        processes: orchestrator.getProcessStatus(),
        outputs: orchestrator.getAllProcessOutputs(),
        isRunning: orchestrator.isRunning,
        systemInfo: orchestrator.getSystemInfo()
    });
});

app.post('/api/start', async (req, res) => {
    try {
        await orchestrator.startAllProcesses();
        res.json({ success: true, message: 'All processes started with Pylon Viewer integration' });
    } catch (error) {
        res.status(500).json({ success: false, message: error.message });
    }
});

app.post('/api/stop', (req, res) => {
    orchestrator.stopAllProcesses();
    res.json({ success: true, message: 'All processes stopped including Pylon Viewer' });
});

app.post('/api/restart/:script', (req, res) => {
    const scriptName = req.params.script;
    orchestrator.restartProcess(scriptName);
    res.json({ success: true, message: `${scriptName} restart initiated` });
});

app.get('/api/system-info', (req, res) => {
    res.json(orchestrator.getSystemInfo());
});

// Serve the main HTML page
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Start the server
const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
    console.log(`🚀 Monitor server running on http://localhost:${PORT}`);
    console.log('📊 Open your browser to monitor the integrated core detection system');
    console.log('🎮 Circle overlay controls: WASD for movement, Q/E for resize');
    console.log('🔍 Pylon Viewer integration enabled');
});

module.exports = { app, server, io }; 