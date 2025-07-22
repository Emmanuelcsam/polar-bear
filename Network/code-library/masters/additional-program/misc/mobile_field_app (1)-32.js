// FiberInspectionApp.js
/**
 * React Native Mobile App for Fiber Optic Field Inspection
 * Features:
 * - Real-time camera capture and analysis
 * - Offline mode with sync
 * - AR overlay for defect visualization
 * - Voice commands
 * - GPS location tracking
 * - Report generation
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  Alert,
  ScrollView,
  Dimensions,
  Platform,
  ActivityIndicator,
  Modal,
  FlatList,
  Image,
  Vibration,
  StatusBar,
} from 'react-native';
import {
  Camera,
  useCameraDevices,
  useFrameProcessor,
} from 'react-native-vision-camera';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createStackNavigator } from '@react-navigation/stack';
import AsyncStorage from '@react-native-async-storage/async-storage';
import NetInfo from '@react-native-community/netinfo';
import Geolocation from '@react-native-community/geolocation';
import Voice from '@react-native-voice';
import RNFS from 'react-native-fs';
import Share from 'react-native-share';
import { ViroARSceneNavigator } from '@viro-community/react-viro';
import Icon from 'react-native-vector-icons/MaterialIcons';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-react-native';
import { fetch } from '@tensorflow/tfjs-react-native';
import axios from 'axios';
import moment from 'moment';
import uuid from 'react-native-uuid';
import PushNotification from 'react-native-push-notification';
import { LineChart, PieChart } from 'react-native-chart-kit';
import RNPickerSelect from 'react-native-picker-select';
import { SwipeListView } from 'react-native-swipe-list-view';
import LottieView from 'lottie-react-native';

const API_BASE_URL = 'https://api.fiber-inspection.com/v2';
const { width: screenWidth, height: screenHeight } = Dimensions.get('window');

// Navigation setup
const Tab = createBottomTabNavigator();
const Stack = createStackNavigator();

// Configure push notifications
PushNotification.configure({
  onNotification: function (notification) {
    console.log('NOTIFICATION:', notification);
  },
  permissions: {
    alert: true,
    badge: true,
    sound: true,
  },
  popInitialNotification: true,
  requestPermissions: true,
});

// Main App Component
export default function App() {
  const [isModelReady, setIsModelReady] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  useEffect(() => {
    // Initialize TensorFlow.js
    tf.ready().then(() => {
      setIsModelReady(true);
      loadModel();
    });

    // Check authentication
    checkAuth();
  }, []);

  const loadModel = async () => {
    try {
      // Load TFLite model for on-device inference
      const modelUrl = `${API_BASE_URL}/models/fiber_inspection_mobile.tflite`;
      // Model loading logic here
    } catch (error) {
      console.error('Failed to load model:', error);
    }
  };

  const checkAuth = async () => {
    const token = await AsyncStorage.getItem('auth_token');
    setIsAuthenticated(!!token);
  };

  if (!isModelReady || !isAuthenticated) {
    return <LoadingScreen />;
  }

  return (
    <NavigationContainer>
      <StatusBar barStyle="light-content" backgroundColor="#1a237e" />
      <Tab.Navigator
        screenOptions={({ route }) => ({
          tabBarIcon: ({ focused, color, size }) => {
            let iconName;
            if (route.name === 'Capture') iconName = 'camera';
            else if (route.name === 'History') iconName = 'history';
            else if (route.name === 'Analytics') iconName = 'analytics';
            else if (route.name === 'Settings') iconName = 'settings';
            
            return <Icon name={iconName} size={size} color={color} />;
          },
          tabBarActiveTintColor: '#3f51b5',
          tabBarInactiveTintColor: 'gray',
        })}
      >
        <Tab.Screen name="Capture" component={CaptureScreen} />
        <Tab.Screen name="History" component={HistoryScreen} />
        <Tab.Screen name="Analytics" component={AnalyticsScreen} />
        <Tab.Screen name="Settings" component={SettingsScreen} />
      </Tab.Navigator>
    </NavigationContainer>
  );
}

// Capture Screen - Main inspection interface
function CaptureScreen({ navigation }) {
  const camera = useRef(null);
  const devices = useCameraDevices();
  const device = devices.back;
  
  const [isRecording, setIsRecording] = useState(false);
  const [capturedImage, setCapturedImage] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [showAR, setShowAR] = useState(false);
  const [location, setLocation] = useState(null);
  const [isOffline, setIsOffline] = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(false);

  useEffect(() => {
    // Get current location
    Geolocation.getCurrentPosition(
      (position) => {
        setLocation({
          latitude: position.coords.latitude,
          longitude: position.coords.longitude,
          accuracy: position.coords.accuracy,
        });
      },
      (error) => console.log('Location error:', error),
      { enableHighAccuracy: true, timeout: 20000, maximumAge: 1000 }
    );

    // Monitor network status
    const unsubscribe = NetInfo.addEventListener((state) => {
      setIsOffline(!state.isConnected);
    });

    // Setup voice commands
    Voice.onSpeechResults = onSpeechResults;

    return () => {
      unsubscribe();
      Voice.destroy().then(Voice.removeAllListeners);
    };
  }, []);

  const onSpeechResults = (e) => {
    const command = e.value[0].toLowerCase();
    
    if (command.includes('capture') || command.includes('analyze')) {
      captureImage();
    } else if (command.includes('save')) {
      saveInspection();
    } else if (command.includes('report')) {
      generateReport();
    }
  };

  const captureImage = async () => {
    if (camera.current) {
      try {
        const photo = await camera.current.takePhoto({
          qualityPrioritization: 'quality',
          flash: 'auto',
          enableAutoRedEyeReduction: true,
        });
        
        setCapturedImage(photo);
        analyzeImage(photo);
      } catch (error) {
        Alert.alert('Error', 'Failed to capture image');
      }
    }
  };

  const analyzeImage = async (photo) => {
    setIsAnalyzing(true);
    
    try {
      if (isOffline) {
        // Offline analysis using local model
        const result = await performLocalAnalysis(photo);
        setAnalysisResult(result);
        
        // Queue for sync when online
        await queueForSync({
          id: uuid.v4(),
          image: photo,
          result: result,
          timestamp: new Date().toISOString(),
          location: location,
        });
      } else {
        // Online analysis
        const formData = new FormData();
        formData.append('image', {
          uri: photo.path,
          type: 'image/jpeg',
          name: 'fiber_inspection.jpg',
        });
        formData.append('location', JSON.stringify(location));
        
        const response = await axios.post(
          `${API_BASE_URL}/analyze`,
          formData,
          {
            headers: {
              'Content-Type': 'multipart/form-data',
              'Authorization': `Bearer ${await AsyncStorage.getItem('auth_token')}`,
            },
          }
        );
        
        setAnalysisResult(response.data);
      }
      
      // Vibrate on anomaly detection
      if (analysisResult?.is_anomalous) {
        Vibration.vibrate([0, 500, 200, 500]);
      }
      
    } catch (error) {
      Alert.alert('Analysis Error', error.message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const performLocalAnalysis = async (photo) => {
    // Load image and preprocess
    const imageTensor = await preprocessImage(photo.path);
    
    // Run inference
    const model = await tf.loadLayersModel('bundleresource://model.json');
    const prediction = model.predict(imageTensor);
    const result = await prediction.data();
    
    // Process results
    return {
      is_anomalous: result[0] > 0.5,
      anomaly_score: result[0],
      severity: getSeverity(result[0]),
      quality_score: (1 - result[0]) * 100,
      defects: detectDefects(imageTensor),
      processing_mode: 'offline',
    };
  };

  const saveInspection = async () => {
    if (!analysisResult) return;
    
    try {
      const inspection = {
        id: uuid.v4(),
        timestamp: new Date().toISOString(),
        location: location,
        image_path: capturedImage.path,
        analysis: analysisResult,
        technician_id: await AsyncStorage.getItem('technician_id'),
        notes: '',
      };
      
      // Save locally
      const inspections = JSON.parse(await AsyncStorage.getItem('inspections') || '[]');
      inspections.push(inspection);
      await AsyncStorage.setItem('inspections', JSON.stringify(inspections));
      
      Alert.alert('Success', 'Inspection saved successfully');
      
      // Reset state
      setCapturedImage(null);
      setAnalysisResult(null);
      
    } catch (error) {
      Alert.alert('Error', 'Failed to save inspection');
    }
  };

  const generateReport = async () => {
    // Generate PDF report
    const report = await createInspectionReport(analysisResult, capturedImage, location);
    
    // Share report
    Share.open({
      url: `file://${report.path}`,
      type: 'application/pdf',
      title: 'Fiber Inspection Report',
    });
  };

  const toggleVoiceCommands = () => {
    if (voiceEnabled) {
      Voice.stop();
    } else {
      Voice.start('en-US');
    }
    setVoiceEnabled(!voiceEnabled);
  };

  if (device == null) return <Text>No camera available</Text>;

  return (
    <View style={styles.container}>
      {!capturedImage ? (
        <Camera
          ref={camera}
          style={StyleSheet.absoluteFill}
          device={device}
          isActive={true}
          photo={true}
          video={false}
        >
          {/* Camera overlay UI */}
          <View style={styles.cameraOverlay}>
            {/* Status bar */}
            <View style={styles.statusBar}>
              <View style={styles.statusItem}>
                <Icon name="location-on" size={20} color="white" />
                <Text style={styles.statusText}>
                  {location ? `±${location.accuracy.toFixed(0)}m` : 'Getting location...'}
                </Text>
              </View>
              <View style={styles.statusItem}>
                <Icon 
                  name={isOffline ? 'signal-wifi-off' : 'signal-wifi-4-bar'} 
                  size={20} 
                  color={isOffline ? '#ff5252' : '#4caf50'} 
                />
                <Text style={styles.statusText}>
                  {isOffline ? 'Offline' : 'Online'}
                </Text>
              </View>
            </View>
            
            {/* Capture controls */}
            <View style={styles.captureControls}>
              <TouchableOpacity
                style={styles.secondaryButton}
                onPress={() => setShowAR(!showAR)}
              >
                <Icon name="view-in-ar" size={30} color="white" />
              </TouchableOpacity>
              
              <TouchableOpacity
                style={styles.captureButton}
                onPress={captureImage}
              >
                <View style={styles.captureButtonInner} />
              </TouchableOpacity>
              
              <TouchableOpacity
                style={[styles.secondaryButton, voiceEnabled && styles.activeButton]}
                onPress={toggleVoiceCommands}
              >
                <Icon name="mic" size={30} color="white" />
              </TouchableOpacity>
            </View>
          </View>
          
          {/* AR Overlay */}
          {showAR && (
            <ViroARSceneNavigator
              initialScene={{ scene: ARDefectOverlay }}
              viroAppProps={{ defects: analysisResult?.defects }}
              style={StyleSheet.absoluteFill}
            />
          )}
        </Camera>
      ) : (
        <AnalysisView
          image={capturedImage}
          result={analysisResult}
          isAnalyzing={isAnalyzing}
          onSave={saveInspection}
          onRetake={() => {
            setCapturedImage(null);
            setAnalysisResult(null);
          }}
          onGenerateReport={generateReport}
        />
      )}
    </View>
  );
}

// Analysis View Component
function AnalysisView({ image, result, isAnalyzing, onSave, onRetake, onGenerateReport }) {
  if (isAnalyzing) {
    return (
      <View style={styles.analysisContainer}>
        <Image source={{ uri: `file://${image.path}` }} style={styles.previewImage} />
        <View style={styles.analyzingOverlay}>
          <ActivityIndicator size="large" color="#3f51b5" />
          <Text style={styles.analyzingText}>Analyzing fiber optic cable...</Text>
          <LottieView
            source={require('./animations/scanning.json')}
            autoPlay
            loop
            style={styles.scanAnimation}
          />
        </View>
      </View>
    );
  }

  return (
    <ScrollView style={styles.analysisContainer}>
      <Image source={{ uri: `file://${image.path}` }} style={styles.previewImage} />
      
      {result && (
        <View style={styles.resultContainer}>
          {/* Overall Status */}
          <View style={[styles.statusCard, result.is_anomalous ? styles.anomalousCard : styles.normalCard]}>
            <Icon 
              name={result.is_anomalous ? 'warning' : 'check-circle'} 
              size={48} 
              color="white" 
            />
            <Text style={styles.statusTitle}>
              {result.is_anomalous ? 'Anomaly Detected' : 'Normal'}
            </Text>
            <Text style={styles.statusSubtitle}>
              Quality Score: {result.quality_score.toFixed(1)}%
            </Text>
          </View>
          
          {/* Severity Indicator */}
          {result.is_anomalous && (
            <View style={styles.severityContainer}>
              <Text style={styles.sectionTitle}>Severity</Text>
              <View style={[styles.severityBadge, styles[`severity${result.severity}`]]}>
                <Text style={styles.severityText}>{result.severity}</Text>
              </View>
            </View>
          )}
          
          {/* Defects List */}
          {result.defects && result.defects.length > 0 && (
            <View style={styles.defectsContainer}>
              <Text style={styles.sectionTitle}>Detected Defects</Text>
              {result.defects.map((defect, index) => (
                <View key={index} style={styles.defectItem}>
                  <Icon name="error-outline" size={24} color="#ff5252" />
                  <View style={styles.defectDetails}>
                    <Text style={styles.defectType}>{defect.type}</Text>
                    <Text style={styles.defectConfidence}>
                      Confidence: {(defect.confidence * 100).toFixed(0)}%
                    </Text>
                  </View>
                </View>
              ))}
            </View>
          )}
          
          {/* Action Buttons */}
          <View style={styles.actionButtons}>
            <TouchableOpacity style={styles.primaryButton} onPress={onSave}>
              <Icon name="save" size={24} color="white" />
              <Text style={styles.buttonText}>Save Inspection</Text>
            </TouchableOpacity>
            
            <TouchableOpacity style={styles.secondaryActionButton} onPress={onGenerateReport}>
              <Icon name="description" size={24} color="#3f51b5" />
              <Text style={styles.secondaryButtonText}>Generate Report</Text>
            </TouchableOpacity>
            
            <TouchableOpacity style={styles.tertiaryButton} onPress={onRetake}>
              <Icon name="refresh" size={24} color="#666" />
              <Text style={styles.tertiaryButtonText}>Retake</Text>
            </TouchableOpacity>
          </View>
        </View>
      )}
    </ScrollView>
  );
}

// History Screen
function HistoryScreen({ navigation }) {
  const [inspections, setInspections] = useState([]);
  const [refreshing, setRefreshing] = useState(false);
  const [filter, setFilter] = useState('all');

  useEffect(() => {
    loadInspections();
  }, [filter]);

  const loadInspections = async () => {
    try {
      const stored = await AsyncStorage.getItem('inspections');
      let allInspections = JSON.parse(stored || '[]');
      
      // Apply filter
      if (filter === 'anomalous') {
        allInspections = allInspections.filter(i => i.analysis?.is_anomalous);
      } else if (filter === 'normal') {
        allInspections = allInspections.filter(i => !i.analysis?.is_anomalous);
      }
      
      // Sort by date
      allInspections.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
      
      setInspections(allInspections);
    } catch (error) {
      console.error('Failed to load inspections:', error);
    }
  };

  const syncInspections = async () => {
    setRefreshing(true);
    
    try {
      // Check for queued offline inspections
      const queued = await AsyncStorage.getItem('sync_queue');
      const queuedInspections = JSON.parse(queued || '[]');
      
      if (queuedInspections.length > 0) {
        // Sync with server
        for (const inspection of queuedInspections) {
          await uploadInspection(inspection);
        }
        
        // Clear queue
        await AsyncStorage.setItem('sync_queue', '[]');
        
        Alert.alert('Success', `Synced ${queuedInspections.length} inspections`);
      }
      
      // Reload
      await loadInspections();
      
    } catch (error) {
      Alert.alert('Sync Error', error.message);
    } finally {
      setRefreshing(false);
    }
  };

  const deleteInspection = async (id) => {
    Alert.alert(
      'Delete Inspection',
      'Are you sure you want to delete this inspection?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            const stored = await AsyncStorage.getItem('inspections');
            let allInspections = JSON.parse(stored || '[]');
            allInspections = allInspections.filter(i => i.id !== id);
            await AsyncStorage.setItem('inspections', JSON.stringify(allInspections));
            loadInspections();
          },
        },
      ]
    );
  };

  const renderInspectionItem = ({ item }) => (
    <TouchableOpacity
      style={styles.inspectionItem}
      onPress={() => navigation.navigate('InspectionDetail', { inspection: item })}
    >
      <Image 
        source={{ uri: `file://${item.image_path}` }} 
        style={styles.thumbnailImage} 
      />
      <View style={styles.inspectionInfo}>
        <Text style={styles.inspectionDate}>
          {moment(item.timestamp).format('MMM DD, YYYY HH:mm')}
        </Text>
        <View style={styles.inspectionStatus}>
          <Icon 
            name={item.analysis?.is_anomalous ? 'warning' : 'check-circle'} 
            size={20} 
            color={item.analysis?.is_anomalous ? '#ff5252' : '#4caf50'} 
          />
          <Text style={styles.inspectionStatusText}>
            {item.analysis?.is_anomalous ? 'Anomaly' : 'Normal'}
          </Text>
        </View>
        <Text style={styles.qualityScore}>
          Quality: {item.analysis?.quality_score?.toFixed(0)}%
        </Text>
      </View>
      <Icon name="chevron-right" size={24} color="#ccc" />
    </TouchableOpacity>
  );

  return (
    <View style={styles.container}>
      {/* Filter Bar */}
      <View style={styles.filterBar}>
        <TouchableOpacity
          style={[styles.filterButton, filter === 'all' && styles.activeFilter]}
          onPress={() => setFilter('all')}
        >
          <Text style={styles.filterText}>All</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.filterButton, filter === 'anomalous' && styles.activeFilter]}
          onPress={() => setFilter('anomalous')}
        >
          <Text style={styles.filterText}>Anomalies</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.filterButton, filter === 'normal' && styles.activeFilter]}
          onPress={() => setFilter('normal')}
        >
          <Text style={styles.filterText}>Normal</Text>
        </TouchableOpacity>
      </View>
      
      {/* Sync Button */}
      <TouchableOpacity style={styles.syncButton} onPress={syncInspections}>
        <Icon name="sync" size={24} color="white" />
        <Text style={styles.syncButtonText}>Sync Offline Inspections</Text>
      </TouchableOpacity>
      
      {/* Inspections List */}
      <SwipeListView
        data={inspections}
        renderItem={renderInspectionItem}
        renderHiddenItem={(data, rowMap) => (
          <View style={styles.rowBack}>
            <TouchableOpacity
              style={[styles.backRightBtn, styles.backRightBtnRight]}
              onPress={() => deleteInspection(data.item.id)}
            >
              <Icon name="delete" size={24} color="white" />
            </TouchableOpacity>
          </View>
        )}
        rightOpenValue={-75}
        keyExtractor={(item) => item.id}
        refreshing={refreshing}
        onRefresh={syncInspections}
        ListEmptyComponent={
          <View style={styles.emptyContainer}>
            <Icon name="history" size={64} color="#ccc" />
            <Text style={styles.emptyText}>No inspections yet</Text>
          </View>
        }
      />
    </View>
  );
}

// Analytics Screen
function AnalyticsScreen() {
  const [analytics, setAnalytics] = useState(null);
  const [timeRange, setTimeRange] = useState('week');

  useEffect(() => {
    loadAnalytics();
  }, [timeRange]);

  const loadAnalytics = async () => {
    try {
      const stored = await AsyncStorage.getItem('inspections');
      const inspections = JSON.parse(stored || '[]');
      
      // Calculate analytics
      const analytics = calculateAnalytics(inspections, timeRange);
      setAnalytics(analytics);
      
    } catch (error) {
      console.error('Failed to load analytics:', error);
    }
  };

  const calculateAnalytics = (inspections, range) => {
    // Filter by time range
    const now = moment();
    const filtered = inspections.filter(i => {
      const inspectionDate = moment(i.timestamp);
      if (range === 'day') return now.diff(inspectionDate, 'days') < 1;
      if (range === 'week') return now.diff(inspectionDate, 'weeks') < 1;
      if (range === 'month') return now.diff(inspectionDate, 'months') < 1;
      return true;
    });
    
    // Calculate metrics
    const total = filtered.length;
    const anomalous = filtered.filter(i => i.analysis?.is_anomalous).length;
    const avgQuality = filtered.reduce((sum, i) => sum + (i.analysis?.quality_score || 0), 0) / (total || 1);
    
    // Defect distribution
    const defectCounts = {};
    filtered.forEach(i => {
      (i.analysis?.defects || []).forEach(d => {
        defectCounts[d.type] = (defectCounts[d.type] || 0) + 1;
      });
    });
    
    // Daily trend
    const dailyData = {};
    filtered.forEach(i => {
      const day = moment(i.timestamp).format('MMM DD');
      if (!dailyData[day]) {
        dailyData[day] = { total: 0, anomalous: 0 };
      }
      dailyData[day].total++;
      if (i.analysis?.is_anomalous) dailyData[day].anomalous++;
    });
    
    return {
      total,
      anomalous,
      normal: total - anomalous,
      anomalyRate: (anomalous / (total || 1)) * 100,
      avgQuality,
      defectDistribution: Object.entries(defectCounts).map(([type, count]) => ({
        name: type,
        value: count,
        color: getDefectColor(type),
      })),
      dailyTrend: Object.entries(dailyData).map(([date, data]) => ({
        date,
        total: data.total,
        anomalous: data.anomalous,
      })),
    };
  };

  if (!analytics) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#3f51b5" />
      </View>
    );
  }

  return (
    <ScrollView style={styles.container}>
      {/* Time Range Selector */}
      <View style={styles.timeRangeSelector}>
        <RNPickerSelect
          value={timeRange}
          onValueChange={(value) => setTimeRange(value)}
          items={[
            { label: 'Today', value: 'day' },
            { label: 'This Week', value: 'week' },
            { label: 'This Month', value: 'month' },
            { label: 'All Time', value: 'all' },
          ]}
          style={pickerSelectStyles}
        />
      </View>
      
      {/* Summary Cards */}
      <View style={styles.summaryCards}>
        <View style={styles.summaryCard}>
          <Text style={styles.summaryValue}>{analytics.total}</Text>
          <Text style={styles.summaryLabel}>Total Inspections</Text>
        </View>
        <View style={[styles.summaryCard, styles.anomalyCard]}>
          <Text style={styles.summaryValue}>{analytics.anomalous}</Text>
          <Text style={styles.summaryLabel}>Anomalies</Text>
        </View>
        <View style={styles.summaryCard}>
          <Text style={styles.summaryValue}>{analytics.anomalyRate.toFixed(1)}%</Text>
          <Text style={styles.summaryLabel}>Anomaly Rate</Text>
        </View>
        <View style={styles.summaryCard}>
          <Text style={styles.summaryValue}>{analytics.avgQuality.toFixed(0)}%</Text>
          <Text style={styles.summaryLabel}>Avg Quality</Text>
        </View>
      </View>
      
      {/* Daily Trend Chart */}
      <View style={styles.chartContainer}>
        <Text style={styles.chartTitle}>Daily Trend</Text>
        <LineChart
          data={{
            labels: analytics.dailyTrend.map(d => d.date),
            datasets: [
              {
                data: analytics.dailyTrend.map(d => d.total),
                color: (opacity = 1) => `rgba(63, 81, 181, ${opacity})`,
                strokeWidth: 2,
              },
              {
                data: analytics.dailyTrend.map(d => d.anomalous),
                color: (opacity = 1) => `rgba(255, 82, 82, ${opacity})`,
                strokeWidth: 2,
              },
            ],
            legend: ['Total', 'Anomalies'],
          }}
          width={screenWidth - 32}
          height={220}
          chartConfig={{
            backgroundColor: '#ffffff',
            backgroundGradientFrom: '#ffffff',
            backgroundGradientTo: '#ffffff',
            decimalPlaces: 0,
            color: (opacity = 1) => `rgba(0, 0, 0, ${opacity})`,
            labelColor: (opacity = 1) => `rgba(0, 0, 0, ${opacity})`,
            style: {
              borderRadius: 16,
            },
          }}
          bezier
          style={styles.chart}
        />
      </View>
      
      {/* Defect Distribution */}
      {analytics.defectDistribution.length > 0 && (
        <View style={styles.chartContainer}>
          <Text style={styles.chartTitle}>Defect Distribution</Text>
          <PieChart
            data={analytics.defectDistribution}
            width={screenWidth - 32}
            height={220}
            chartConfig={{
              color: (opacity = 1) => `rgba(0, 0, 0, ${opacity})`,
            }}
            accessor="value"
            backgroundColor="transparent"
            paddingLeft="15"
            absolute
          />
        </View>
      )}
    </ScrollView>
  );
}

// Settings Screen
function SettingsScreen() {
  const [settings, setSettings] = useState({
    autoSync: true,
    voiceCommands: false,
    arOverlay: true,
    highQualityCapture: true,
    notifications: true,
  });

  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    try {
      const stored = await AsyncStorage.getItem('settings');
      if (stored) {
        setSettings(JSON.parse(stored));
      }
    } catch (error) {
      console.error('Failed to load settings:', error);
    }
  };

  const updateSetting = async (key, value) => {
    const newSettings = { ...settings, [key]: value };
    setSettings(newSettings);
    await AsyncStorage.setItem('settings', JSON.stringify(newSettings));
  };

  const logout = async () => {
    Alert.alert(
      'Logout',
      'Are you sure you want to logout?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Logout',
          style: 'destructive',
          onPress: async () => {
            await AsyncStorage.multiRemove(['auth_token', 'technician_id']);
            // Navigate to login screen
          },
        },
      ]
    );
  };

  return (
    <ScrollView style={styles.container}>
      <View style={styles.settingsSection}>
        <Text style={styles.sectionTitle}>Capture Settings</Text>
        
        <View style={styles.settingItem}>
          <Text style={styles.settingLabel}>High Quality Capture</Text>
          <Switch
            value={settings.highQualityCapture}
            onValueChange={(value) => updateSetting('highQualityCapture', value)}
          />
        </View>
        
        <View style={styles.settingItem}>
          <Text style={styles.settingLabel}>AR Overlay</Text>
          <Switch
            value={settings.arOverlay}
            onValueChange={(value) => updateSetting('arOverlay', value)}
          />
        </View>
        
        <View style={styles.settingItem}>
          <Text style={styles.settingLabel}>Voice Commands</Text>
          <Switch
            value={settings.voiceCommands}
            onValueChange={(value) => updateSetting('voiceCommands', value)}
          />
        </View>
      </View>
      
      <View style={styles.settingsSection}>
        <Text style={styles.sectionTitle}>Sync Settings</Text>
        
        <View style={styles.settingItem}>
          <Text style={styles.settingLabel}>Auto Sync When Online</Text>
          <Switch
            value={settings.autoSync}
            onValueChange={(value) => updateSetting('autoSync', value)}
          />
        </View>
        
        <View style={styles.settingItem}>
          <Text style={styles.settingLabel}>Push Notifications</Text>
          <Switch
            value={settings.notifications}
            onValueChange={(value) => updateSetting('notifications', value)}
          />
        </View>
      </View>
      
      <TouchableOpacity style={styles.logoutButton} onPress={logout}>
        <Text style={styles.logoutText}>Logout</Text>
      </TouchableOpacity>
    </ScrollView>
  );
}

// Helper functions
const preprocessImage = async (imagePath) => {
  // Image preprocessing for TensorFlow.js
  const imageAssetPath = Image.resolveAssetSource({ uri: imagePath });
  const response = await fetch(imageAssetPath.uri, {}, { isBinary: true });
  const imageData = await response.arrayBuffer();
  const imageTensor = decodeJpeg(new Uint8Array(imageData));
  
  // Resize and normalize
  const resized = tf.image.resizeBilinear(imageTensor, [224, 224]);
  const normalized = resized.div(255.0);
  
  return normalized.expandDims(0);
};

const getSeverity = (score) => {
  if (score < 0.3) return 'LOW';
  if (score < 0.6) return 'MEDIUM';
  if (score < 0.8) return 'HIGH';
  return 'CRITICAL';
};

const getDefectColor = (type) => {
  const colors = {
    scratch: '#ff5252',
    contamination: '#ff9800',
    dig: '#ffc107',
    other: '#9e9e9e',
  };
  return colors[type] || colors.other;
};

const queueForSync = async (inspection) => {
  const queue = JSON.parse(await AsyncStorage.getItem('sync_queue') || '[]');
  queue.push(inspection);
  await AsyncStorage.setItem('sync_queue', JSON.stringify(queue));
};

const uploadInspection = async (inspection) => {
  const formData = new FormData();
  formData.append('inspection_data', JSON.stringify(inspection));
  formData.append('image', {
    uri: inspection.image.path,
    type: 'image/jpeg',
    name: `inspection_${inspection.id}.jpg`,
  });
  
  await axios.post(`${API_BASE_URL}/inspections/upload`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
      'Authorization': `Bearer ${await AsyncStorage.getItem('auth_token')}`,
    },
  });
};

const createInspectionReport = async (analysis, image, location) => {
  // Generate PDF report
  // Implementation would use react-native-pdf-lib or similar
  const reportPath = `${RNFS.DocumentDirectoryPath}/report_${Date.now()}.pdf`;
  
  // For demo, create a simple text report
  const reportContent = `
Fiber Optic Inspection Report
=============================
Date: ${moment().format('MMMM DD, YYYY HH:mm')}
Location: ${location ? `${location.latitude}, ${location.longitude}` : 'Unknown'}

Analysis Results:
- Status: ${analysis.is_anomalous ? 'ANOMALY DETECTED' : 'NORMAL'}
- Quality Score: ${analysis.quality_score.toFixed(1)}%
- Severity: ${analysis.severity}

Defects Found: ${analysis.defects.length}
${analysis.defects.map(d => `- ${d.type} (Confidence: ${(d.confidence * 100).toFixed(0)}%)`).join('\n')}

Generated by Fiber Inspection Mobile App
  `;
  
  await RNFS.writeFile(reportPath, reportContent, 'utf8');
  
  return { path: reportPath };
};

// Styles
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  cameraOverlay: {
    flex: 1,
    justifyContent: 'space-between',
  },
  statusBar: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    padding: 16,
    backgroundColor: 'rgba(0,0,0,0.3)',
  },
  statusItem: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  statusText: {
    color: 'white',
    marginLeft: 4,
    fontSize: 14,
  },
  captureControls: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    paddingBottom: 32,
    paddingHorizontal: 32,
  },
  captureButton: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: 'white',
    justifyContent: 'center',
    alignItems: 'center',
  },
  captureButtonInner: {
    width: 70,
    height: 70,
    borderRadius: 35,
    backgroundColor: '#ff5252',
  },
  secondaryButton: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: 'rgba(255,255,255,0.3)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  activeButton: {
    backgroundColor: 'rgba(63,81,181,0.8)',
  },
  analysisContainer: {
    flex: 1,
    backgroundColor: 'white',
  },
  previewImage: {
    width: screenWidth,
    height: screenWidth * 0.75,
    resizeMode: 'cover',
  },
  analyzingOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0,0,0,0.7)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  analyzingText: {
    color: 'white',
    fontSize: 18,
    marginTop: 16,
  },
  scanAnimation: {
    width: 200,
    height: 200,
  },
  resultContainer: {
    padding: 16,
  },
  statusCard: {
    padding: 24,
    borderRadius: 12,
    alignItems: 'center',
    marginBottom: 16,
  },
  normalCard: {
    backgroundColor: '#4caf50',
  },
  anomalousCard: {
    backgroundColor: '#ff5252',
  },
  statusTitle: {
    color: 'white',
    fontSize: 24,
    fontWeight: 'bold',
    marginTop: 8,
  },
  statusSubtitle: {
    color: 'white',
    fontSize: 16,
    marginTop: 4,
  },
  severityContainer: {
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  severityBadge: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    alignSelf: 'flex-start',
  },
  severityLOW: {
    backgroundColor: '#ffc107',
  },
  severityMEDIUM: {
    backgroundColor: '#ff9800',
  },
  severityHIGH: {
    backgroundColor: '#ff5722',
  },
  severityCRITICAL: {
    backgroundColor: '#f44336',
  },
  severityText: {
    color: 'white',
    fontWeight: 'bold',
  },
  defectsContainer: {
    marginBottom: 16,
  },
  defectItem: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 12,
    backgroundColor: '#f5f5f5',
    borderRadius: 8,
    marginBottom: 8,
  },
  defectDetails: {
    marginLeft: 12,
    flex: 1,
  },
  defectType: {
    fontSize: 16,
    fontWeight: '500',
    textTransform: 'capitalize',
  },
  defectConfidence: {
    fontSize: 14,
    color: '#666',
  },
  actionButtons: {
    marginTop: 24,
  },
  primaryButton: {
    flexDirection: 'row',
    backgroundColor: '#3f51b5',
    paddingVertical: 16,
    paddingHorizontal: 24,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 12,
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
    marginLeft: 8,
  },
  secondaryActionButton: {
    flexDirection: 'row',
    backgroundColor: 'white',
    borderWidth: 2,
    borderColor: '#3f51b5',
    paddingVertical: 16,
    paddingHorizontal: 24,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 12,
  },
  secondaryButtonText: {
    color: '#3f51b5',
    fontSize: 16,
    fontWeight: 'bold',
    marginLeft: 8,
  },
  tertiaryButton: {
    flexDirection: 'row',
    paddingVertical: 16,
    paddingHorizontal: 24,
    alignItems: 'center',
    justifyContent: 'center',
  },
  tertiaryButtonText: {
    color: '#666',
    fontSize: 16,
    marginLeft: 8,
  },
  filterBar: {
    flexDirection: 'row',
    padding: 16,
    backgroundColor: 'white',
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  filterButton: {
    flex: 1,
    paddingVertical: 8,
    alignItems: 'center',
  },
  activeFilter: {
    borderBottomWidth: 2,
    borderBottomColor: '#3f51b5',
  },
  filterText: {
    fontSize: 16,
    color: '#666',
  },
  syncButton: {
    flexDirection: 'row',
    backgroundColor: '#3f51b5',
    margin: 16,
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
  },
  syncButtonText: {
    color: 'white',
    fontSize: 16,
    marginLeft: 8,
  },
  inspectionItem: {
    flexDirection: 'row',
    padding: 16,
    backgroundColor: 'white',
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
    alignItems: 'center',
  },
  thumbnailImage: {
    width: 60,
    height: 60,
    borderRadius: 8,
  },
  inspectionInfo: {
    flex: 1,
    marginLeft: 16,
  },
  inspectionDate: {
    fontSize: 16,
    fontWeight: '500',
  },
  inspectionStatus: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 4,
  },
  inspectionStatusText: {
    marginLeft: 4,
    fontSize: 14,
  },
  qualityScore: {
    fontSize: 14,
    color: '#666',
    marginTop: 2,
  },
  rowBack: {
    alignItems: 'center',
    backgroundColor: '#ff5252',
    flex: 1,
    flexDirection: 'row',
    justifyContent: 'flex-end',
  },
  backRightBtn: {
    alignItems: 'center',
    bottom: 0,
    justifyContent: 'center',
    position: 'absolute',
    top: 0,
    width: 75,
  },
  backRightBtnRight: {
    backgroundColor: '#ff5252',
    right: 0,
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: 64,
  },
  emptyText: {
    fontSize: 18,
    color: '#999',
    marginTop: 16,
  },
  timeRangeSelector: {
    padding: 16,
    backgroundColor: 'white',
  },
  summaryCards: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    padding: 8,
  },
  summaryCard: {
    width: (screenWidth - 48) / 2,
    backgroundColor: 'white',
    padding: 16,
    margin: 8,
    borderRadius: 12,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  anomalyCard: {
    backgroundColor: '#ffebee',
  },
  summaryValue: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#333',
  },
  summaryLabel: {
    fontSize: 14,
    color: '#666',
    marginTop: 4,
  },
  chartContainer: {
    backgroundColor: 'white',
    margin: 16,
    padding: 16,
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  chartTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 16,
  },
  chart: {
    marginVertical: 8,
    borderRadius: 16,
  },
  settingsSection: {
    backgroundColor: 'white',
    marginBottom: 16,
    paddingVertical: 8,
  },
  settingItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  settingLabel: {
    fontSize: 16,
  },
  logoutButton: {
    backgroundColor: '#ff5252',
    margin: 16,
    padding: 16,
    borderRadius: 8,
    alignItems: 'center',
  },
  logoutText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
});

const pickerSelectStyles = StyleSheet.create({
  inputIOS: {
    fontSize: 16,
    paddingVertical: 12,
    paddingHorizontal: 10,
    borderWidth: 1,
    borderColor: '#e0e0e0',
    borderRadius: 4,
    color: 'black',
    paddingRight: 30,
  },
  inputAndroid: {
    fontSize: 16,
    paddingHorizontal: 10,
    paddingVertical: 8,
    borderWidth: 0.5,
    borderColor: '#e0e0e0',
    borderRadius: 8,
    color: 'black',
    paddingRight: 30,
  },
});

// Loading Screen Component
function LoadingScreen() {
  return (
    <View style={styles.loadingContainer}>
      <LottieView
        source={require('./animations/loading.json')}
        autoPlay
        loop
        style={{ width: 200, height: 200 }}
      />
      <Text style={{ fontSize: 18, marginTop: 16 }}>
        Initializing Fiber Inspection System...
      </Text>
    </View>
  );
}

// AR Defect Overlay Scene
const ARDefectOverlay = ({ defects }) => {
  // AR implementation would go here
  // This would overlay defect indicators on the camera view
  return null;
};