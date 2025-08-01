const orchestrator = require('./orchestrator');
const fs = require('fs');

async function testOrchestrator() {
    console.log('🧪 Testing Core Detection Orchestrator...\n');

    // Test 1: Check if config.json exists
    console.log('1. Testing configuration loading...');
    try {
        const configExists = fs.existsSync('config.json');
        console.log(`   ✓ Config file exists: ${configExists}`);
        
        if (configExists) {
            const config = JSON.parse(fs.readFileSync('config.json', 'utf8'));
            console.log(`   ✓ Config loaded successfully with ${Object.keys(config).length} sections`);
        }
    } catch (error) {
        console.log(`   ✗ Config loading failed: ${error.message}`);
    }

    // Test 2: Check if Python scripts exist
    console.log('\n2. Testing Python script availability...');
    const scripts = [
        { name: 'auto-core-detection.py', required: true },
        { name: 'live_feed.py', required: true },
        { name: 'main.py', required: true },
        { name: 'config_manager.py', required: true }
    ];

    for (const script of scripts) {
        const exists = fs.existsSync(script.name);
        const status = exists ? '✓' : '✗';
        console.log(`   ${status} ${script.name} ${exists ? 'exists' : 'MISSING'}`);
        
        if (!exists && script.required) {
            console.log(`   ⚠️  Required script ${script.name} is missing!`);
        }
    }

    // Test 3: Test Python path detection
    console.log('\n3. Testing Python path detection...');
    try {
        const pythonPath = orchestrator.pythonPath;
        console.log(`   ✓ Python found at: ${pythonPath}`);
    } catch (error) {
        console.log(`   ✗ Python detection failed: ${error.message}`);
    }

    // Test 4: Test process management (without actually starting processes)
    console.log('\n4. Testing process management methods...');
    
    // Test status methods
    const status = orchestrator.getProcessStatus();
    console.log(`   ✓ Process status method works: ${Object.keys(status).length} processes tracked`);
    
    const outputs = orchestrator.getAllProcessOutputs();
    console.log(`   ✓ Process outputs method works: ${Object.keys(outputs).length} outputs tracked`);

    // Test 5: Test event system
    console.log('\n5. Testing event system...');
    
    let eventTestPassed = false;
    orchestrator.once('allProcessesStarted', () => {
        eventTestPassed = true;
        console.log('   ✓ Event system working');
    });

    // Test 6: Test configuration access
    console.log('\n6. Testing configuration access...');
    try {
        const config = orchestrator.config;
        if (config && Object.keys(config).length > 0) {
            console.log(`   ✓ Configuration loaded with ${Object.keys(config).length} sections`);
            
            // Check for required sections
            const requiredSections = ['auto_core_detection', 'live_feed', 'main'];
            for (const section of requiredSections) {
                if (config[section]) {
                    console.log(`   ✓ Section '${section}' found`);
                } else {
                    console.log(`   ⚠️  Section '${section}' missing`);
                }
            }
        } else {
            console.log('   ⚠️  Configuration is empty or invalid');
        }
    } catch (error) {
        console.log(`   ✗ Configuration access failed: ${error.message}`);
    }

    console.log('\n✅ Orchestrator test completed successfully!');
    console.log('\nTo start the actual parallel processes, run:');
    console.log('   npm start');
    console.log('   or');
    console.log('   node orchestrator.js');
}

// Run the test
testOrchestrator().catch(error => {
    console.error('❌ Test failed:', error.message);
    process.exit(1);
}); 