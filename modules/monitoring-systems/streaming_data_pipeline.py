# streaming_pipeline.py
"""
Real-time Data Streaming Pipeline for Fiber Inspection
Features:
- Apache Kafka integration
- Stream processing with Faust
- Real-time analytics
- Data lake integration
- Stream enrichment
- Windowed aggregations
"""

import faust
from typing import Optional, List, Dict, Any
import asyncio
from datetime import datetime, timedelta
import json
import numpy as np
from dataclasses import dataclass, field
import aiokafka
from confluent_kafka import Producer, Consumer
from confluent_kafka.avro import AvroProducer, AvroConsumer
from confluent_kafka.schema_registry import SchemaRegistryClient
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.io import ReadFromKafka, WriteToBigQuery
import logging
from prometheus_client import Counter, Histogram, Gauge
import redis
import aioredis
from google.cloud import bigquery, storage
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import AsyncIterator
import uvloop
from sklearn.preprocessing import StandardScaler
from river import anomaly, metrics, preprocessing

# Use uvloop for better performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

logger = logging.getLogger(__name__)

# Metrics
MESSAGES_PROCESSED = Counter('pipeline_messages_processed_total', 'Total messages processed')
PROCESSING_LAG = Histogram('pipeline_processing_lag_seconds', 'Processing lag')
ANOMALIES_DETECTED = Counter('pipeline_anomalies_detected_total', 'Anomalies detected in stream')
ENRICHMENT_ERRORS = Counter('pipeline_enrichment_errors_total', 'Enrichment errors')

# Faust app configuration
app = faust.App(
    'fiber-inspection-pipeline',
    broker='kafka://kafka-broker:9092',
    value_serializer='json',
    table_cleanup_interval=30.0,
    stream_wait_empty=False,
    producer_max_request_size=10485760,  # 10MB
)

# Redis streams
redis_client = None

# BigQuery client
bq_client = bigquery.Client()
storage_client = storage.Client()

# Topics
inspection_topic = app.topic('fiber-inspection-raw', partitions=12)
enriched_topic = app.topic('fiber-inspection-enriched', partitions=12)
alerts_topic = app.topic('fiber-inspection-alerts', partitions=6)
metrics_topic = app.topic('fiber-inspection-metrics', partitions=6)


# Data Models
class InspectionEvent(faust.Record, serializer='json'):
    """Raw inspection event"""
    event_id: str
    stream_id: str
    frame_id: str
    timestamp: datetime
    device_id: str
    location: str
    image_url: Optional[str] = None
    features: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class EnrichedInspectionEvent(faust.Record, serializer='json'):
    """Enriched inspection event with predictions"""
    event_id: str
    stream_id: str
    frame_id: str
    timestamp: datetime
    device_id: str
    location: str
    image_url: Optional[str] = None
    features: Optional[List[float]] = None
    prediction: Dict[str, Any] = field(default_factory=dict)
    anomaly_score: float = 0.0
    quality_score: float = 100.0
    severity: str = 'NORMAL'
    defects: List[Dict] = field(default_factory=list)
    enrichment_timestamp: Optional[datetime] = None
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class AggregatedMetrics(faust.Record, serializer='json'):
    """Aggregated metrics for a time window"""
    window_start: datetime
    window_end: datetime
    stream_id: str
    total_frames: int = 0
    anomalous_frames: int = 0
    anomaly_rate: float = 0.0
    avg_quality_score: float = 0.0
    avg_processing_time: float = 0.0
    defect_counts: Dict[str, int] = field(default_factory=dict)
    severity_distribution: Dict[str, int] = field(default_factory=dict)


# Tables for stateful processing
inspection_stats = app.Table(
    'inspection-stats',
    default=dict,
    partitions=12,
    on_window_close=lambda key, value: logger.info(f"Window closed for {key}: {value}")
)

device_state = app.Table(
    'device-state',
    default=dict,
    partitions=12
)

# Stream processors
@app.agent(inspection_topic)
async def process_raw_inspections(stream: AsyncIterator[InspectionEvent]) -> AsyncIterator[EnrichedInspectionEvent]:
    """Process raw inspection events"""
    # Initialize online anomaly detector
    anomaly_detector = anomaly.HalfSpaceTrees(n_trees=10, height=8)
    scaler = preprocessing.StandardScaler()
    
    async for event in stream:
        try:
            start_time = datetime.now()
            MESSAGES_PROCESSED.inc()
            
            # Update device state
            device_state[event.device_id] = {
                'last_seen': event.timestamp,
                'stream_id': event.stream_id,
                'location': event.location,
                'status': 'active'
            }
            
            # Enrich event
            enriched = await enrich_inspection_event(event)
            
            # Online anomaly detection if features available
            if enriched.features:
                # Scale features
                features_dict = {f'f_{i}': v for i, v in enumerate(enriched.features)}
                scaled_features = scaler.learn_one(features_dict).transform_one(features_dict)
                
                # Detect anomalies
                anomaly_score = anomaly_detector.score_one(scaled_features)
                anomaly_detector.learn_one(scaled_features)
                
                enriched.anomaly_score = float(anomaly_score)
                enriched.severity = classify_severity(anomaly_score)
                
                if anomaly_score > 0.7:
                    ANOMALIES_DETECTED.inc()
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            enriched.processing_time_ms = processing_time
            
            # Update processing lag metric
            lag = (datetime.now() - event.timestamp).total_seconds()
            PROCESSING_LAG.observe(lag)
            
            # Forward to enriched topic
            await enriched_topic.send(value=enriched)
            
            yield enriched
            
        except Exception as e:
            logger.error(f"Error processing event {event.event_id}: {e}")
            ENRICHMENT_ERRORS.inc()


@app.agent(enriched_topic)
async def aggregate_metrics(stream: AsyncIterator[EnrichedInspectionEvent]) -> None:
    """Aggregate metrics in time windows"""
    # 5-minute tumbling windows
    async for event in stream.tumbling(300):  # 5 minutes
        # Update statistics table
        key = f"{event.stream_id}:{event.timestamp.strftime('%Y%m%d%H')}"
        
        stats = inspection_stats[key]
        stats['total_frames'] = stats.get('total_frames', 0) + 1
        
        if event.anomaly_score > 0.5:
            stats['anomalous_frames'] = stats.get('anomalous_frames', 0) + 1
            
        # Update quality scores
        quality_sum = stats.get('quality_sum', 0) + event.quality_score
        stats['quality_sum'] = quality_sum
        
        # Update defect counts
        defect_counts = stats.get('defect_counts', {})
        for defect in event.defects:
            defect_type = defect.get('type', 'unknown')
            defect_counts[defect_type] = defect_counts.get(defect_type, 0) + 1
        stats['defect_counts'] = defect_counts
        
        # Update severity distribution
        severity_dist = stats.get('severity_distribution', {})
        severity_dist[event.severity] = severity_dist.get(event.severity, 0) + 1
        stats['severity_distribution'] = severity_dist
        
        inspection_stats[key] = stats


@app.agent(enriched_topic)
async def detect_alert_conditions(stream: AsyncIterator[EnrichedInspectionEvent]) -> None:
    """Detect alert conditions in the stream"""
    # Sliding window for anomaly rate calculation
    window_size = 60  # 1 minute
    anomaly_window = []
    
    async for event in stream:
        # Add to window
        anomaly_window.append({
            'timestamp': event.timestamp,
            'is_anomalous': event.anomaly_score > 0.5
        })
        
        # Remove old entries
        cutoff_time = datetime.now() - timedelta(seconds=window_size)
        anomaly_window = [e for e in anomaly_window if e['timestamp'] > cutoff_time]
        
        # Calculate anomaly rate
        if len(anomaly_window) > 10:
            anomaly_rate = sum(1 for e in anomaly_window if e['is_anomalous']) / len(anomaly_window)
            
            # Check alert conditions
            alerts = []
            
            # High anomaly rate alert
            if anomaly_rate > 0.3:
                alerts.append({
                    'type': 'HIGH_ANOMALY_RATE',
                    'severity': 'HIGH',
                    'stream_id': event.stream_id,
                    'anomaly_rate': anomaly_rate,
                    'message': f'High anomaly rate detected: {anomaly_rate:.1%}'
                })
            
            # Critical defect alert
            if event.severity == 'CRITICAL':
                alerts.append({
                    'type': 'CRITICAL_DEFECT',
                    'severity': 'CRITICAL',
                    'stream_id': event.stream_id,
                    'frame_id': event.frame_id,
                    'defects': event.defects,
                    'message': f'Critical defect detected in frame {event.frame_id}'
                })
            
            # Send alerts
            for alert in alerts:
                await alerts_topic.send(value=alert)


@app.timer(interval=60.0)  # Every minute
async def publish_metrics():
    """Publish aggregated metrics periodically"""
    # Calculate and publish metrics for each stream
    for key, stats in inspection_stats.items():
        stream_id, hour = key.split(':')
        
        total_frames = stats.get('total_frames', 0)
        if total_frames > 0:
            metrics = AggregatedMetrics(
                window_start=datetime.strptime(hour, '%Y%m%d%H'),
                window_end=datetime.now(),
                stream_id=stream_id,
                total_frames=total_frames,
                anomalous_frames=stats.get('anomalous_frames', 0),
                anomaly_rate=stats.get('anomalous_frames', 0) / total_frames,
                avg_quality_score=stats.get('quality_sum', 0) / total_frames,
                defect_counts=stats.get('defect_counts', {}),
                severity_distribution=stats.get('severity_distribution', {})
            )
            
            await metrics_topic.send(value=metrics)


@app.timer(interval=300.0)  # Every 5 minutes
async def archive_to_storage():
    """Archive processed data to cloud storage"""
    try:
        # Get recent enriched events
        cutoff_time = datetime.now() - timedelta(minutes=10)
        
        # In production, this would query from a proper store
        # For now, we'll demonstrate the pattern
        
        # Create Parquet file
        bucket_name = 'fiber-inspection-archive'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        blob_name = f'enriched_events/{timestamp}.parquet'
        
        # Write to Cloud Storage
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        logger.info(f"Archived data to gs://{bucket_name}/{blob_name}")
        
    except Exception as e:
        logger.error(f"Archive error: {e}")


# Helper functions
async def enrich_inspection_event(event: InspectionEvent) -> EnrichedInspectionEvent:
    """Enrich inspection event with additional data"""
    enriched = EnrichedInspectionEvent(
        event_id=event.event_id,
        stream_id=event.stream_id,
        frame_id=event.frame_id,
        timestamp=event.timestamp,
        device_id=event.device_id,
        location=event.location,
        image_url=event.image_url,
        features=event.features,
        metadata=event.metadata,
        enrichment_timestamp=datetime.now()
    )
    
    # Add predictions if ML service is available
    if event.features:
        try:
            # In production, call ML service
            # For demo, simulate prediction
            prediction = await get_ml_prediction(event.features)
            enriched.prediction = prediction
            enriched.anomaly_score = prediction.get('anomaly_score', 0.0)
            enriched.quality_score = prediction.get('quality_score', 100.0)
            enriched.defects = prediction.get('defects', [])
        except Exception as e:
            logger.error(f"Failed to get ML prediction: {e}")
    
    # Add location metadata
    enriched.metadata['processing_node'] = 'stream-processor-1'
    enriched.metadata['pipeline_version'] = '2.0'
    
    return enriched


async def get_ml_prediction(features: List[float]) -> Dict[str, Any]:
    """Get ML prediction for features"""
    # In production, this would call the ML service
    # For demo, return mock prediction
    is_anomalous = np.random.random() < 0.1
    
    prediction = {
        'anomaly_score': np.random.random() if is_anomalous else np.random.random() * 0.3,
        'quality_score': np.random.uniform(40, 70) if is_anomalous else np.random.uniform(80, 100),
        'defects': []
    }
    
    if is_anomalous:
        num_defects = np.random.randint(1, 4)
        for i in range(num_defects):
            defect = {
                'type': np.random.choice(['scratch', 'contamination', 'dig']),
                'confidence': np.random.uniform(0.5, 1.0),
                'location': [
                    np.random.randint(0, 640),
                    np.random.randint(0, 480)
                ]
            }
            prediction['defects'].append(defect)
    
    return prediction


def classify_severity(anomaly_score: float) -> str:
    """Classify severity based on anomaly score"""
    if anomaly_score < 0.3:
        return 'NORMAL'
    elif anomaly_score < 0.5:
        return 'LOW'
    elif anomaly_score < 0.7:
        return 'MEDIUM'
    elif anomaly_score < 0.9:
        return 'HIGH'
    else:
        return 'CRITICAL'


# Apache Beam pipeline for batch processing
class ProcessInspectionData(beam.DoFn):
    """Process inspection data in batch"""
    
    def process(self, element):
        # Parse and process element
        try:
            data = json.loads(element)
            
            # Extract features
            features = data.get('features', [])
            
            # Calculate statistics
            if features:
                stats = {
                    'mean': np.mean(features),
                    'std': np.std(features),
                    'min': np.min(features),
                    'max': np.max(features)
                }
                data['feature_stats'] = stats
            
            yield data
            
        except Exception as e:
            logging.error(f"Processing error: {e}")


def run_batch_pipeline():
    """Run batch processing pipeline"""
    pipeline_options = PipelineOptions([
        '--runner=DataflowRunner',
        '--project=your-project-id',
        '--region=us-central1',
        '--temp_location=gs://your-bucket/temp',
        '--job_name=fiber-inspection-batch'
    ])
    
    with beam.Pipeline(options=pipeline_options) as pipeline:
        # Read from Kafka
        raw_data = (
            pipeline
            | 'ReadFromKafka' >> ReadFromKafka(
                consumer_config={
                    'bootstrap.servers': 'kafka-broker:9092',
                    'group.id': 'beam-batch-processor'
                },
                topics=['fiber-inspection-enriched']
            )
        )
        
        # Process data
        processed = (
            raw_data
            | 'ProcessData' >> beam.ParDo(ProcessInspectionData())
            | 'WindowIntoHours' >> beam.WindowInto(
                beam.window.FixedWindows(3600)  # 1 hour windows
            )
        )
        
        # Calculate aggregations
        hourly_stats = (
            processed
            | 'ExtractStreamId' >> beam.Map(lambda x: (x['stream_id'], x))
            | 'GroupByStream' >> beam.GroupByKey()
            | 'CalculateStats' >> beam.Map(calculate_hourly_statistics)
        )
        
        # Write to BigQuery
        hourly_stats | 'WriteToBigQuery' >> WriteToBigQuery(
            table='your-project:fiber_inspection.hourly_statistics',
            schema='stream_id:STRING,hour:TIMESTAMP,total_frames:INTEGER,'
                   'anomaly_rate:FLOAT,avg_quality:FLOAT',
            write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND
        )


def calculate_hourly_statistics(element):
    """Calculate hourly statistics for a stream"""
    stream_id, events = element
    events_list = list(events)
    
    total_frames = len(events_list)
    anomalous_frames = sum(1 for e in events_list if e.get('anomaly_score', 0) > 0.5)
    avg_quality = np.mean([e.get('quality_score', 100) for e in events_list])
    
    return {
        'stream_id': stream_id,
        'hour': datetime.now().replace(minute=0, second=0, microsecond=0),
        'total_frames': total_frames,
        'anomaly_rate': anomalous_frames / total_frames if total_frames > 0 else 0,
        'avg_quality': avg_quality
    }


# Redis Streams integration
async def publish_to_redis_stream(event: EnrichedInspectionEvent):
    """Publish event to Redis Stream for real-time consumers"""
    if not redis_client:
        redis_client = await aioredis.create_redis_pool('redis://redis:6379')
    
    # Prepare event data
    event_data = {
        'event_id': event.event_id,
        'stream_id': event.stream_id,
        'timestamp': event.timestamp.isoformat(),
        'anomaly_score': str(event.anomaly_score),
        'severity': event.severity,
        'defects': json.dumps(event.defects)
    }
    
    # Add to stream
    await redis_client.xadd(
        f'inspection:{event.stream_id}',
        event_data,
        max_len=10000  # Keep last 10k events
    )


# Stream analytics queries
@app.page('/analytics/anomaly_trends/{stream_id}/')
async def anomaly_trends(web, request, stream_id: str):
    """Get anomaly trends for a stream"""
    # Query from inspection_stats table
    stats = []
    
    for key, value in inspection_stats.items():
        if key.startswith(f"{stream_id}:"):
            stats.append({
                'hour': key.split(':')[1],
                'total_frames': value.get('total_frames', 0),
                'anomalous_frames': value.get('anomalous_frames', 0),
                'anomaly_rate': value.get('anomalous_frames', 0) / value.get('total_frames', 1)
            })
    
    return web.json({
        'stream_id': stream_id,
        'trends': sorted(stats, key=lambda x: x['hour'])
    })


@app.page('/analytics/device_status/')
async def device_status_endpoint(web, request):
    """Get current device status"""
    devices = []
    
    for device_id, state in device_state.items():
        devices.append({
            'device_id': device_id,
            'last_seen': state.get('last_seen', '').isoformat() if state.get('last_seen') else None,
            'stream_id': state.get('stream_id'),
            'location': state.get('location'),
            'status': state.get('status', 'unknown')
        })
    
    return web.json({'devices': devices})


# Main entry point
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'batch':
        # Run batch pipeline
        run_batch_pipeline()
    else:
        # Run streaming pipeline
        app.main()
