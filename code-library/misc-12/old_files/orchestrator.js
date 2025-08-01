const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const EventEmitter = require('events');

class CoreDetectionOrchestrator extends EventEmitter {
    constructor() {
        super();
        this.processes = new Map();
        this.pythonPath = this.findPythonPath();
        this.config = this.loadConfig();
        this.isRunning = false;
        this.processOutputs = new Map();
        this.pylonViewerProcess = null;
        this.integratedMode = true; // Enable integrated mode with circle overlay
    }

    findPythonPath() {
        // Try to find Python executable
        const possiblePaths = [
            'py',
            'python',
            'python3',
            '/c/Users/Saem1001/AppData/Local/Programs/Python/Python313/python.exe'
        ];

        for (const pythonPath of possiblePaths) {
            try {
                const result = require('child_process').spawnSync(pythonPath, ['--version'], { 
                    stdio: 'pipe',
                    timeout: 5000 
                });
                if (result.status === 0) {
                    console.log(`Found Python at: ${pythonPath}`);
                    return pythonPath;
                }
            } catch (error) {
                console.log(`Python not found at: ${pythonPath}`);
            }
        }
        
        throw new Error('Python executable not found. Please ensure Python is installed and in PATH.');
    }

    loadConfig() {
        try {
            const configPath = path.join(__dirname, 'config.json');
            const configData = fs.readFileSync(configPath, 'utf8');
            return JSON.parse(configData);
        } catch (error) {
            console.error('Error loading config.json:', error.message);
            return {};
        }
    }

    async startPylonViewer() {
        console.log('Starting Pylon Viewer integration...');
        
        try {
            // Start Pylon Viewer integration script
            this.pylonViewerProcess = spawn(this.pythonPath, ['pylon_viewer_integration.py'], {
                stdio: ['pipe', 'pipe', 'pipe'],
                cwd: __dirname
            });

            this.pylonViewerProcess.stdout.on('data', (data) => {
                const output = data.toString();
                console.log(`[Pylon Viewer] ${output.trim()}`);
                this.emit('processOutput', { script: 'pylon-viewer', output: output.trim(), type: 'stdout' });
            });

            this.pylonViewerProcess.stderr.on('data', (data) => {
                const output = data.toString();
                console.error(`[Pylon Viewer ERROR] ${output.trim()}`);
                this.emit('processOutput', { script: 'pylon-viewer', output: output.trim(), type: 'stderr' });
            });

            this.pylonViewerProcess.on('close', (code) => {
                console.log(`[Pylon Viewer] Process exited with code ${code}`);
                this.pylonViewerProcess = null;
            });

            // Wait for Pylon Viewer to start
            await new Promise((resolve) => {
                setTimeout(resolve, 3000); // Give Pylon Viewer time to start
            });

            console.log('Pylon Viewer integration started successfully');
            return true;
        } catch (error) {
            console.error('Failed to start Pylon Viewer:', error.message);
            return false;
        }
    }

    startProcess(scriptName, scriptPath, args = []) {
        return new Promise((resolve, reject) => {
            console.log(`Starting ${scriptName}...`);
            
            const process = spawn(this.pythonPath, [scriptPath, ...args], {
                stdio: ['pipe', 'pipe', 'pipe'],
                cwd: __dirname
            });

            this.processes.set(scriptName, process);
            this.processOutputs.set(scriptName, { stdout: '', stderr: '' });

            process.stdout.on('data', (data) => {
                const output = data.toString();
                this.processOutputs.get(scriptName).stdout += output;
                console.log(`[${scriptName}] ${output.trim()}`);
                this.emit('processOutput', { script: scriptName, output: output.trim(), type: 'stdout' });
            });

            process.stderr.on('data', (data) => {
                const output = data.toString();
                this.processOutputs.get(scriptName).stderr += output;
                console.error(`[${scriptName} ERROR] ${output.trim()}`);
                this.emit('processOutput', { script: scriptName, output: output.trim(), type: 'stderr' });
            });

            process.on('close', (code) => {
                console.log(`[${scriptName}] Process exited with code ${code}`);
                this.emit('processClosed', { script: scriptName, code: code });
                
                if (code !== 0) {
                    reject(new Error(`${scriptName} exited with code ${code}`));
                } else {
                    resolve();
                }
            });

            process.on('error', (error) => {
                console.error(`[${scriptName}] Process error:`, error);
                this.emit('processError', { script: scriptName, error: error });
                reject(error);
            });

            // Wait a bit to see if process starts successfully
            setTimeout(() => {
                if (process.exitCode === null) {
                    console.log(`[${scriptName}] Process started successfully`);
                    resolve();
                }
            }, 2000);
        });
    }

    async startAllProcesses() {
        if (this.isRunning) {
            console.log('Orchestrator is already running');
            return;
        }

        this.isRunning = true;
        console.log('Starting Core Detection Orchestrator with Pylon Viewer...');
        console.log(`Using Python path: ${this.pythonPath}`);

        // Start Pylon Viewer first
        const pylonViewerStarted = await this.startPylonViewer();
        if (!pylonViewerStarted) {
            console.log('Pylon Viewer failed to start, continuing without it...');
        }

        // Wait a moment for Pylon Viewer to initialize
        await new Promise(resolve => setTimeout(resolve, 2000));

        const scripts = [
            { name: 'main', path: 'main.py', args: [] }, // Use real camera
            { name: 'auto-core-detection', path: 'auto-core-detection.py', args: [] },
            { name: 'live-feed', path: 'live_feed.py', args: [] } // Use real camera
        ];

        const startPromises = scripts.map(script => {
            return this.startProcess(script.name, script.path, script.args)
                .catch(error => {
                    console.error(`Failed to start ${script.name}:`, error.message);
                    return Promise.reject(error);
                });
        });

        try {
            await Promise.all(startPromises);
            console.log('All processes started successfully!');
            console.log('🎉 Core Detection System is now running with:');
            console.log('   - Pylon Viewer integration');
            console.log('   - Circle overlay controls');
            console.log('   - Auto core detection');
            console.log('   - Live feed processing');
            console.log('   - Web monitoring interface at http://localhost:3000');
            this.emit('allProcessesStarted');
        } catch (error) {
            console.error('Failed to start all processes:', error.message);
            this.stopAllProcesses();
            throw error;
        }
    }

    stopProcess(scriptName) {
        const process = this.processes.get(scriptName);
        if (process) {
            console.log(`Stopping ${scriptName}...`);
            process.kill('SIGTERM');
            
            // Force kill after 5 seconds if process doesn't terminate gracefully
            setTimeout(() => {
                if (process.exitCode === null) {
                    console.log(`Force killing ${scriptName}...`);
                    process.kill('SIGKILL');
                }
            }, 5000);
        }
    }

    stopAllProcesses() {
        console.log('Stopping all processes...');
        
        // Stop main processes
        for (const [scriptName, process] of this.processes) {
            this.stopProcess(scriptName);
        }
        
        // Stop Pylon Viewer
        if (this.pylonViewerProcess) {
            console.log('Stopping Pylon Viewer...');
            this.pylonViewerProcess.kill('SIGTERM');
            setTimeout(() => {
                if (this.pylonViewerProcess && this.pylonViewerProcess.exitCode === null) {
                    this.pylonViewerProcess.kill('SIGKILL');
                }
            }, 5000);
        }
        
        this.isRunning = false;
        this.emit('allProcessesStopped');
    }

    getProcessStatus() {
        const status = {};
        for (const [scriptName, process] of this.processes) {
            status[scriptName] = {
                pid: process.pid,
                exitCode: process.exitCode,
                killed: process.killed,
                running: process.exitCode === null && !process.killed
            };
        }
        
        // Add Pylon Viewer status
        if (this.pylonViewerProcess) {
            status['pylon-viewer'] = {
                pid: this.pylonViewerProcess.pid,
                exitCode: this.pylonViewerProcess.exitCode,
                killed: this.pylonViewerProcess.killed,
                running: this.pylonViewerProcess.exitCode === null && !this.pylonViewerProcess.killed
            };
        }
        
        return status;
    }

    getProcessOutput(scriptName) {
        return this.processOutputs.get(scriptName) || { stdout: '', stderr: '' };
    }

    getAllProcessOutputs() {
        const outputs = {};
        for (const [scriptName] of this.processes) {
            outputs[scriptName] = this.getProcessOutput(scriptName);
        }
        return outputs;
    }

    sendSignalToProcess(scriptName, signal) {
        const process = this.processes.get(scriptName);
        if (process && process.exitCode === null) {
            process.kill(signal);
            console.log(`Sent signal ${signal} to ${scriptName}`);
        }
    }

    restartProcess(scriptName) {
        console.log(`Restarting ${scriptName}...`);
        this.stopProcess(scriptName);
        
        setTimeout(async () => {
            try {
                const scriptPath = `${scriptName.replace('-', '_')}.py`;
                // Don't force any specific camera mode - let auto-detection work
                const args = [];
                await this.startProcess(scriptName, scriptPath, args);
                console.log(`${scriptName} restarted successfully`);
            } catch (error) {
                console.error(`Failed to restart ${scriptName}:`, error.message);
            }
        }, 2000);
    }

    getSystemInfo() {
        return {
            pythonPath: this.pythonPath,
            isRunning: this.isRunning,
            pylonViewerRunning: this.pylonViewerProcess && this.pylonViewerProcess.exitCode === null,
            integratedMode: this.integratedMode,
            processes: this.getProcessStatus(),
            config: this.config
        };
    }
}

// Create and export the orchestrator instance
const orchestrator = new CoreDetectionOrchestrator();

// Handle graceful shutdown
process.on('SIGINT', () => {
    console.log('\nReceived SIGINT, shutting down gracefully...');
    orchestrator.stopAllProcesses();
    process.exit(0);
});

process.on('SIGTERM', () => {
    console.log('\nReceived SIGTERM, shutting down gracefully...');
    orchestrator.stopAllProcesses();
    process.exit(0);
});

// Event listeners for monitoring
orchestrator.on('processOutput', ({ script, output, type }) => {
    // You can add custom logging or monitoring here
    if (type === 'stderr') {
        // Log errors to a file or monitoring system
        fs.appendFileSync('orchestrator_errors.log', `[${new Date().toISOString()}] ${script}: ${output}\n`);
    }
});

orchestrator.on('processClosed', ({ script, code }) => {
    console.log(`Process ${script} closed with code ${code}`);
});

orchestrator.on('allProcessesStarted', () => {
    console.log('🎉 All core detection processes are now running in parallel!');
    console.log('Process status:', orchestrator.getProcessStatus());
    console.log('📊 Open http://localhost:3000 to monitor the system');
    console.log('🎮 Use WASD keys to control the circle overlay');
    console.log('🔍 Pylon Viewer should be open for camera interaction');
});

orchestrator.on('allProcessesStopped', () => {
    console.log('All processes have been stopped');
});

// Export for use in other modules
module.exports = orchestrator;

// If this file is run directly, start the orchestrator
if (require.main === module) {
    orchestrator.startAllProcesses()
        .then(() => {
            console.log('Orchestrator started successfully');
            
            // Keep the process running
            process.stdin.resume();
        })
        .catch(error => {
            console.error('Failed to start orchestrator:', error.message);
            process.exit(1);
        });
} 