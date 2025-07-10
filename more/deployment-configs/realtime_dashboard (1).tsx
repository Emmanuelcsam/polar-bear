import React, { useState, useEffect, useRef, useCallback } from 'react';
import { LineChart, Line, AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { AlertCircle, Activity, Camera, Wifi, WifiOff, TrendingUp, TrendingDown, CheckCircle, XCircle, AlertTriangle, Zap } from 'lucide-react';

// WebSocket connection
const WS_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8765';

// Color scheme
const COLORS = {
  normal: '#4ade80',
  low: '#fbbf24',
  medium: '#fb923c',
  high: '#f87171',
  critical: '#dc2626',
  primary: '#3b82f6',
  secondary: '#6366f1',
  background: '#f3f4f6',
  card: '#ffffff',
  text: '#1f2937'
};

// Main Dashboard Component
export default function FiberInspectionDashboard() {
  const [connected, setConnected] = useState(false);
  const [streams, setStreams] = useState({});
  const [realtimeData, setRealtimeData] = useState([]);
  const [statistics, setStatistics] = useState({
    totalFrames: 0,
    totalAnomalies: 0,
    avgQuality: 0,
    activeStreams: 0
  });
  const [alerts, setAlerts] = useState([]);
  const [selectedStream, setSelectedStream] = useState(null);
  
  const ws = useRef(null);
  const chartDataRef = useRef([]);
  const MAX_DATA_POINTS = 100;

  // WebSocket connection management
  useEffect(() => {
    connectWebSocket();
    
    return () => {
      if (ws.current) {
        ws.current.close();
      }
    };
  }, []);

  const connectWebSocket = useCallback(() => {
    ws.current = new WebSocket(WS_URL);
    
    ws.current.onopen = () => {
      console.log('WebSocket connected');
      setConnected(true);
    };
    
    ws.current.onmessage = (event) => {
      const message = JSON.parse(event.data);
      handleWebSocketMessage(message);
    };
    
    ws.current.onclose = () => {
      console.log('WebSocket disconnected');
      setConnected(false);
      // Reconnect after 3 seconds
      setTimeout(connectWebSocket, 3000);
    };
    
    ws.current.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  }, []);

  const handleWebSocketMessage = (message) => {
    if (message.type === 'inspection_result') {
      const result = message.data;
      
      // Update stream data
      setStreams(prev => ({
        ...prev,
        [result.stream_id]: {
          ...prev[result.stream_id],
          lastUpdate: result.timestamp,
          lastResult: result,
          status: result.is_anomalous ? 'anomaly' : 'normal'
        }
      }));
      
      // Update real-time chart data
      const newDataPoint = {
        time: new Date(result.timestamp).toLocaleTimeString(),
        anomalyScore: result.anomaly_score,
        qualityScore: result.quality_score,
        streamId: result.stream_id
      };
      
      chartDataRef.current = [...chartDataRef.current, newDataPoint].slice(-MAX_DATA_POINTS);
      setRealtimeData([...chartDataRef.current]);
      
      // Update statistics
      setStatistics(prev => ({
        totalFrames: prev.totalFrames + 1,
        totalAnomalies: prev.totalAnomalies + (result.is_anomalous ? 1 : 0),
        avgQuality: ((prev.avgQuality * prev.totalFrames) + result.quality_score) / (prev.totalFrames + 1),
        activeStreams: Object.keys(streams).length
      }));
      
      // Add alert if critical
      if (result.severity === 'CRITICAL' || result.severity === 'HIGH') {
        setAlerts(prev => [{
          id: Date.now(),
          timestamp: new Date(),
          severity: result.severity,
          streamId: result.stream_id,
          message: `${result.severity} anomaly detected in ${result.stream_id}`,
          defects: result.defects
        }, ...prev].slice(0, 10)); // Keep last 10 alerts
      }
    }
  };

  // Stream card component
  const StreamCard = ({ streamId, data }) => {
    const status = data?.status || 'offline';
    const lastResult = data?.lastResult;
    
    return (
      <div 
        className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
          status === 'anomaly' ? 'border-red-500 bg-red-50' : 
          status === 'normal' ? 'border-green-500 bg-green-50' : 
          'border-gray-300 bg-gray-50'
        }`}
        onClick={() => setSelectedStream(streamId)}
      >
        <div className="flex items-center justify-between mb-2">
          <h3 className="font-semibold text-lg">{streamId}</h3>
          <Camera className={`w-5 h-5 ${status === 'offline' ? 'text-gray-400' : 'text-blue-500'}`} />
        </div>
        
        {lastResult && (
          <>
            <div className="space-y-1 text-sm">
              <div className="flex justify-between">
                <span>Quality Score:</span>
                <span className="font-medium">{lastResult.quality_score.toFixed(1)}%</span>
              </div>
              <div className="flex justify-between">
                <span>Anomaly Score:</span>
                <span className={`font-medium ${lastResult.anomaly_score > 0.5 ? 'text-red-600' : 'text-green-600'}`}>
                  {lastResult.anomaly_score.toFixed(3)}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Defects:</span>
                <span className="font-medium">{lastResult.defects.length}</span>
              </div>
            </div>
            
            <div className="mt-2">
              <div className={`text-xs px-2 py-1 rounded inline-block ${
                SEVERITY_STYLES[lastResult.severity]
              }`}>
                {lastResult.severity}
              </div>
            </div>
          </>
        )}
      </div>
    );
  };

  // Alert component
  const AlertItem = ({ alert }) => {
    const icons = {
      CRITICAL: <XCircle className="w-5 h-5 text-red-600" />,
      HIGH: <AlertTriangle className="w-5 h-5 text-orange-600" />,
      MEDIUM: <AlertCircle className="w-5 h-5 text-yellow-600" />
    };
    
    return (
      <div className={`p-3 rounded-lg border-l-4 ${
        alert.severity === 'CRITICAL' ? 'border-red-600 bg-red-50' :
        alert.severity === 'HIGH' ? 'border-orange-600 bg-orange-50' :
        'border-yellow-600 bg-yellow-50'
      }`}>
        <div className="flex items-start space-x-3">
          {icons[alert.severity]}
          <div className="flex-1">
            <p className="font-medium text-sm">{alert.message}</p>
            <p className="text-xs text-gray-600 mt-1">
              {alert.timestamp.toLocaleTimeString()} - {alert.defects.length} defects detected
            </p>
          </div>
        </div>
      </div>
    );
  };

  // Statistics card
  const StatCard = ({ title, value, icon, trend, color }) => {
    return (
      <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm text-gray-600">{title}</p>
            <p className="text-2xl font-bold mt-1" style={{ color }}>
              {typeof value === 'number' ? value.toFixed(1) : value}
            </p>
          </div>
          <div className={`p-3 rounded-full ${color ? 'bg-opacity-10' : 'bg-gray-100'}`} 
               style={{ backgroundColor: color ? `${color}20` : undefined }}>
            {icon}
          </div>
        </div>
        {trend && (
          <div className="mt-3 flex items-center text-sm">
            {trend > 0 ? (
              <TrendingUp className="w-4 h-4 text-green-500 mr-1" />
            ) : (
              <TrendingDown className="w-4 h-4 text-red-500 mr-1" />
            )}
            <span className={trend > 0 ? 'text-green-600' : 'text-red-600'}>
              {Math.abs(trend)}% from last hour
            </span>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">
              Fiber Optic Inspection Dashboard
            </h1>
            <p className="text-gray-600 mt-1">Real-time anomaly detection and monitoring</p>
          </div>
          <div className="flex items-center space-x-4">
            <div className={`flex items-center px-3 py-2 rounded-lg ${
              connected ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
            }`}>
              {connected ? <Wifi className="w-4 h-4 mr-2" /> : <WifiOff className="w-4 h-4 mr-2" />}
              {connected ? 'Connected' : 'Disconnected'}
            </div>
          </div>
        </div>
      </div>

      {/* Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <StatCard
          title="Total Frames Processed"
          value={statistics.totalFrames}
          icon={<Activity className="w-6 h-6 text-blue-600" />}
          color={COLORS.primary}
        />
        <StatCard
          title="Anomalies Detected"
          value={statistics.totalAnomalies}
          icon={<AlertCircle className="w-6 h-6 text-red-600" />}
          color={COLORS.high}
          trend={-5}
        />
        <StatCard
          title="Average Quality Score"
          value={`${statistics.avgQuality.toFixed(1)}%`}
          icon={<CheckCircle className="w-6 h-6 text-green-600" />}
          color={COLORS.normal}
          trend={2.3}
        />
        <StatCard
          title="Active Streams"
          value={Object.keys(streams).length}
          icon={<Camera className="w-6 h-6 text-purple-600" />}
          color={COLORS.secondary}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Stream Grid */}
        <div className="lg:col-span-2">
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <h2 className="text-xl font-semibold mb-4">Active Inspection Streams</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {Object.entries(streams).map(([streamId, data]) => (
                <StreamCard key={streamId} streamId={streamId} data={data} />
              ))}
              {Object.keys(streams).length === 0 && (
                <div className="col-span-2 text-center py-8 text-gray-500">
                  No active streams detected
                </div>
              )}
            </div>
          </div>

          {/* Real-time Chart */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mt-6">
            <h2 className="text-xl font-semibold mb-4">Real-time Analysis</h2>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={realtimeData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis domain={[0, 1]} />
                <Tooltip />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="anomalyScore" 
                  stroke={COLORS.high} 
                  name="Anomaly Score"
                  strokeWidth={2}
                />
                <Line 
                  type="monotone" 
                  dataKey="qualityScore" 
                  stroke={COLORS.normal} 
                  name="Quality Score"
                  strokeWidth={2}
                  yAxisId="quality"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Alerts Panel */}
        <div className="lg:col-span-1">
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold">Recent Alerts</h2>
              <Zap className="w-5 h-5 text-yellow-500" />
            </div>
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {alerts.map(alert => (
                <AlertItem key={alert.id} alert={alert} />
              ))}
              {alerts.length === 0 && (
                <p className="text-center py-8 text-gray-500">
                  No alerts at this time
                </p>
              )}
            </div>
          </div>

          {/* Severity Distribution */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mt-6">
            <h2 className="text-xl font-semibold mb-4">Severity Distribution</h2>
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie
                  data={getSeverityDistribution()}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={renderCustomizedLabel}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {getSeverityDistribution().map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[entry.name.toLowerCase()]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Selected Stream Detail */}
      {selectedStream && streams[selectedStream]?.lastResult && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-6 z-50"
             onClick={() => setSelectedStream(null)}>
          <div className="bg-white rounded-lg p-6 max-w-2xl w-full max-h-96 overflow-y-auto"
               onClick={(e) => e.stopPropagation()}>
            <h3 className="text-xl font-semibold mb-4">
              Stream Details: {selectedStream}
            </h3>
            <div className="space-y-4">
              <div>
                <h4 className="font-medium mb-2">Last Analysis Result</h4>
                <pre className="bg-gray-100 p-3 rounded text-sm overflow-x-auto">
                  {JSON.stringify(streams[selectedStream].lastResult, null, 2)}
                </pre>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// Helper functions
const SEVERITY_STYLES = {
  NORMAL: 'bg-green-100 text-green-800',
  LOW: 'bg-yellow-100 text-yellow-800',
  MEDIUM: 'bg-orange-100 text-orange-800',
  HIGH: 'bg-red-100 text-red-800',
  CRITICAL: 'bg-red-200 text-red-900'
};

function getSeverityDistribution() {
  // Mock data - in production, calculate from actual results
  return [
    { name: 'normal', value: 75 },
    { name: 'low', value: 15 },
    { name: 'medium', value: 7 },
    { name: 'high', value: 2 },
    { name: 'critical', value: 1 }
  ];
}

function renderCustomizedLabel(props) {
  const { cx, cy, midAngle, innerRadius, outerRadius, percent } = props;
  const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
  const x = cx + radius * Math.cos(-midAngle * Math.PI / 180);
  const y = cy + radius * Math.sin(-midAngle * Math.PI / 180);

  return (
    <text x={x} y={y} fill="white" textAnchor={x > cx ? 'start' : 'end'} dominantBaseline="central">
      {`${(percent * 100).toFixed(0)}%`}
    </text>
  );
}