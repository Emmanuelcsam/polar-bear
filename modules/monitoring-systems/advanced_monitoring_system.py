# advanced_monitoring.py
"""
Advanced Monitoring and Continuous Learning System
Features:
- Real-time performance monitoring
- Anomaly pattern analysis
- Model drift detection
- Automated retraining pipeline
- Intelligent alerting
- Performance optimization
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from prometheus_client import Counter, Histogram, Gauge, Summary
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import slack_sdk
from twilio.rest import Client as TwilioClient
import firebase_admin
from firebase_admin import firestore, ml
from google.cloud import aiplatform
from google.cloud import bigquery
import mlflow
import optuna
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Metrics
MODEL_DRIFT_SCORE = Gauge('fiber_model_drift_score', 'Model drift detection score')
RETRAINING_TRIGGERED = Counter('fiber_retraining_triggered_total', 'Number of times retraining was triggered')
ALERT_SENT = Counter('fiber_alerts_sent_total', 'Total alerts sent', ['severity', 'channel'])
FALSE_POSITIVE_RATE = Gauge('fiber_false_positive_rate', 'Current false positive rate')
FALSE_NEGATIVE_RATE = Gauge('fiber_false_negative_rate', 'Current false negative rate')
MODEL_ACCURACY = Gauge('fiber_model_accuracy', 'Current model accuracy')


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring"""
    timestamp: datetime
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    false_negative_rate: float
    processing_time_avg: float
    processing_time_p95: float
    queue_size_avg: float
    anomaly_detection_rate: float
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'false_positive_rate': self.false_positive_rate,
            'false_negative_rate': self.false_negative_rate,
            'processing_time_avg': self.processing_time_avg,
            'processing_time_p95': self.processing_time_p95,
            'queue_size_avg': self.queue_size_avg,
            'anomaly_detection_rate': self.anomaly_detection_rate
        }


@dataclass
class ModelDriftReport:
    """Model drift analysis report"""
    drift_score: float
    feature_drift: Dict[str, float]
    prediction_drift: float
    data_quality_issues: List[str]
    recommendations: List[str]
    requires_retraining: bool


class AdvancedMonitoringSystem:
    """Comprehensive monitoring and continuous learning system"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.db = firestore.client()
        self.bq_client = bigquery.Client()
        self.ml_client = aiplatform.gapic.ModelServiceClient()
        
        # Initialize alert channels
        self.alert_manager = AlertManager(config)
        
        # Initialize MLflow for experiment tracking
        mlflow.set_tracking_uri(config.get('mlflow_uri'))
        mlflow.set_experiment('fiber_inspection_monitoring')
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.drift_history = deque(maxlen=100)
        
        # Reference distributions for drift detection
        self.reference_distributions = {}
        self.load_reference_distributions()
        
    async def start(self):
        """Start monitoring system"""
        logger.info("Starting Advanced Monitoring System")
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self.performance_monitor()),
            asyncio.create_task(self.drift_detector()),
            asyncio.create_task(self.quality_analyzer()),
            asyncio.create_task(self.retraining_pipeline()),
            asyncio.create_task(self.alert_processor()),
            asyncio.create_task(self.optimization_engine())
        ]
        
        await asyncio.gather(*tasks)
    
    async def performance_monitor(self):
        """Monitor system performance metrics"""
        while True:
            try:
                # Collect metrics from various sources
                metrics = await self.collect_performance_metrics()
                
                # Store metrics
                self.performance_history.append(metrics)
                await self.store_metrics(metrics)
                
                # Update Prometheus metrics
                MODEL_ACCURACY.set(metrics.accuracy)
                FALSE_POSITIVE_RATE.set(metrics.false_positive_rate)
                FALSE_NEGATIVE_RATE.set(metrics.false_negative_rate)
                
                # Check for performance degradation
                if await self.check_performance_degradation(metrics):
                    await self.alert_manager.send_alert(
                        severity='WARNING',
                        title='Performance Degradation Detected',
                        message=f'Model accuracy dropped to {metrics.accuracy:.2%}'
                    )
                
                # Generate performance report
                if datetime.now().hour == 0:  # Daily report
                    await self.generate_performance_report()
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(300)
    
    async def collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect performance metrics from various sources"""
        # Query recent predictions and ground truth
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)
        
        # Get predictions from BigQuery
        query = f"""
        SELECT 
            p.prediction,
            p.confidence,
            p.ground_truth,
            p.processing_time,
            p.queue_size,
            p.timestamp
        FROM `{self.config['project_id']}.fiber_inspection.predictions` p
        WHERE p.timestamp BETWEEN @start_time AND @end_time
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("start_time", "TIMESTAMP", start_time),
                bigquery.ScalarQueryParameter("end_time", "TIMESTAMP", end_time)
            ]
        )
        
        results = self.bq_client.query(query, job_config=job_config).to_dataframe()
        
        if len(results) == 0:
            # Return default metrics if no data
            return PerformanceMetrics(
                timestamp=datetime.now(),
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                false_positive_rate=0.0,
                false_negative_rate=0.0,
                processing_time_avg=0.0,
                processing_time_p95=0.0,
                queue_size_avg=0.0,
                anomaly_detection_rate=0.0
            )
        
        # Calculate metrics
        y_true = results['ground_truth'].values
        y_pred = results['prediction'].values
        
        # Remove samples without ground truth
        mask = ~pd.isna(y_true)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) > 0:
            # Calculate classification metrics
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = f1_score(y_true, y_pred)
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        else:
            accuracy = precision = recall = f1 = fpr = fnr = 0
        
        # Calculate operational metrics
        processing_time_avg = results['processing_time'].mean()
        processing_time_p95 = results['processing_time'].quantile(0.95)
        queue_size_avg = results['queue_size'].mean()
        anomaly_rate = results['prediction'].mean()
        
        return PerformanceMetrics(
            timestamp=datetime.now(),
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            false_positive_rate=fpr,
            false_negative_rate=fnr,
            processing_time_avg=processing_time_avg,
            processing_time_p95=processing_time_p95,
            queue_size_avg=queue_size_avg,
            anomaly_detection_rate=anomaly_rate
        )
    
    async def drift_detector(self):
        """Detect model and data drift"""
        while True:
            try:
                # Analyze drift
                drift_report = await self.analyze_drift()
                
                # Store drift metrics
                self.drift_history.append(drift_report)
                MODEL_DRIFT_SCORE.set(drift_report.drift_score)
                
                # Log to MLflow
                with mlflow.start_run():
                    mlflow.log_metric('drift_score', drift_report.drift_score)
                    mlflow.log_metric('prediction_drift', drift_report.prediction_drift)
                    
                    for feature, drift in drift_report.feature_drift.items():
                        mlflow.log_metric(f'drift_{feature}', drift)
                
                # Check if retraining is needed
                if drift_report.requires_retraining:
                    logger.warning(f"High drift detected: {drift_report.drift_score}")
                    await self.trigger_retraining(drift_report)
                
                # Alert on significant drift
                if drift_report.drift_score > 0.3:
                    await self.alert_manager.send_alert(
                        severity='HIGH',
                        title='Significant Model Drift Detected',
                        message=f'Drift score: {drift_report.drift_score:.3f}\n' +
                               f'Recommendations: {", ".join(drift_report.recommendations)}'
                    )
                
                await asyncio.sleep(3600)  # Check hourly
                
            except Exception as e:
                logger.error(f"Drift detection error: {e}")
                await asyncio.sleep(3600)
    
    async def analyze_drift(self) -> ModelDriftReport:
        """Analyze model and data drift"""
        # Get recent feature distributions
        recent_data = await self.get_recent_feature_data()
        
        feature_drift = {}
        data_quality_issues = []
        
        # Compare with reference distributions
        for feature, ref_dist in self.reference_distributions.items():
            if feature in recent_data:
                # Kolmogorov-Smirnov test for distribution drift
                ks_stat, p_value = stats.ks_2samp(ref_dist, recent_data[feature])
                feature_drift[feature] = ks_stat
                
                # Check for significant drift
                if p_value < 0.05:
                    data_quality_issues.append(f"Significant drift in {feature}")
        
        # Analyze prediction drift
        prediction_drift = await self.analyze_prediction_drift()
        
        # Calculate overall drift score
        drift_score = np.mean(list(feature_drift.values()) + [prediction_drift])
        
        # Generate recommendations
        recommendations = []
        if drift_score > 0.5:
            recommendations.append("Immediate model retraining recommended")
        elif drift_score > 0.3:
            recommendations.append("Schedule model retraining soon")
        
        if len(data_quality_issues) > 0:
            recommendations.append("Investigate data quality issues")
        
        # Determine if retraining is required
        requires_retraining = drift_score > 0.5 or len(data_quality_issues) > 3
        
        return ModelDriftReport(
            drift_score=drift_score,
            feature_drift=feature_drift,
            prediction_drift=prediction_drift,
            data_quality_issues=data_quality_issues,
            recommendations=recommendations,
            requires_retraining=requires_retraining
        )
    
    async def quality_analyzer(self):
        """Analyze data and prediction quality"""
        while True:
            try:
                # Analyze data quality
                quality_report = await self.analyze_data_quality()
                
                # Check for anomalies in the data pipeline
                if quality_report['missing_data_rate'] > 0.1:
                    await self.alert_manager.send_alert(
                        severity='MEDIUM',
                        title='High Missing Data Rate',
                        message=f'Missing data rate: {quality_report["missing_data_rate"]:.1%}'
                    )
                
                # Analyze prediction patterns
                pattern_analysis = await self.analyze_prediction_patterns()
                
                # Detect suspicious patterns
                if pattern_analysis['anomaly_clustering'] > 0.8:
                    await self.alert_manager.send_alert(
                        severity='HIGH',
                        title='Suspicious Anomaly Clustering',
                        message='Detected unusual clustering of anomalies - possible systematic issue'
                    )
                
                await asyncio.sleep(1800)  # Check every 30 minutes
                
            except Exception as e:
                logger.error(f"Quality analysis error: {e}")
                await asyncio.sleep(1800)
    
    async def retraining_pipeline(self):
        """Automated model retraining pipeline"""
        while True:
            try:
                # Check if retraining is scheduled
                if await self.should_retrain():
                    logger.info("Starting automated model retraining")
                    
                    # Collect training data
                    training_data = await self.collect_training_data()
                    
                    # Hyperparameter optimization
                    best_params = await self.optimize_hyperparameters(training_data)
                    
                    # Train new model
                    new_model = await self.train_model(training_data, best_params)
                    
                    # Validate new model
                    validation_results = await self.validate_model(new_model)
                    
                    # Deploy if performance improved
                    if validation_results['f1_score'] > await self.get_current_model_performance():
                        await self.deploy_model(new_model, validation_results)
                        
                        await self.alert_manager.send_alert(
                            severity='INFO',
                            title='Model Successfully Retrained and Deployed',
                            message=f'New F1 score: {validation_results["f1_score"]:.3f}'
                        )
                    else:
                        logger.info("New model did not improve performance, keeping current model")
                    
                    RETRAINING_TRIGGERED.inc()
                
                await asyncio.sleep(86400)  # Check daily
                
            except Exception as e:
                logger.error(f"Retraining pipeline error: {e}")
                await asyncio.sleep(86400)
    
    async def optimize_hyperparameters(self, training_data: pd.DataFrame) -> Dict:
        """Optimize model hyperparameters using Optuna"""
        def objective(trial):
            # Suggest hyperparameters
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5)
            }
            
            # Train and evaluate model with cross-validation
            # (Implementation depends on your specific model)
            score = self._cross_validate_model(training_data, params)
            
            return score
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            study_name='fiber_model_optimization'
        )
        
        # Optimize
        study.optimize(objective, n_trials=50, n_jobs=-1)
        
        # Log best parameters
        with mlflow.start_run():
            mlflow.log_params(study.best_params)
            mlflow.log_metric('best_cv_score', study.best_value)
        
        return study.best_params
    
    async def alert_processor(self):
        """Process and manage alerts"""
        while True:
            try:
                # Check alert conditions
                alerts = await self.check_alert_conditions()
                
                # Process each alert
                for alert in alerts:
                    # Check if alert should be suppressed
                    if not await self.should_suppress_alert(alert):
                        await self.alert_manager.send_alert(**alert)
                        ALERT_SENT.labels(
                            severity=alert['severity'],
                            channel=alert.get('channel', 'default')
                        ).inc()
                
                # Clean up old alerts
                await self.cleanup_old_alerts()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Alert processing error: {e}")
                await asyncio.sleep(60)
    
    async def optimization_engine(self):
        """Continuous optimization of system parameters"""
        while True:
            try:
                # Analyze system performance
                performance_analysis = await self.analyze_system_performance()
                
                # Optimize processing parameters
                if performance_analysis['avg_latency'] > 500:  # ms
                    # Adjust batch size
                    new_batch_size = max(5, performance_analysis['current_batch_size'] - 2)
                    await self.update_system_parameter('batch_size', new_batch_size)
                    
                    logger.info(f"Reduced batch size to {new_batch_size} due to high latency")
                
                # Optimize resource allocation
                if performance_analysis['cpu_utilization'] > 0.8:
                    # Request more resources
                    await self.scale_up_resources()
                elif performance_analysis['cpu_utilization'] < 0.3:
                    # Scale down to save costs
                    await self.scale_down_resources()
                
                # Optimize model serving
                if performance_analysis['model_cache_hit_rate'] < 0.7:
                    await self.optimize_model_cache()
                
                await asyncio.sleep(600)  # Optimize every 10 minutes
                
            except Exception as e:
                logger.error(f"Optimization error: {e}")
                await asyncio.sleep(600)
    
    async def generate_performance_report(self):
        """Generate comprehensive performance report"""
        # Collect metrics for the past 24 hours
        metrics_df = pd.DataFrame([m.to_dict() for m in self.performance_history])
        
        # Create visualizations
        fig = go.Figure()
        
        # Accuracy over time
        fig.add_trace(go.Scatter(
            x=metrics_df['timestamp'],
            y=metrics_df['accuracy'],
            mode='lines',
            name='Accuracy'
        ))
        
        # F1 score over time
        fig.add_trace(go.Scatter(
            x=metrics_df['timestamp'],
            y=metrics_df['f1_score'],
            mode='lines',
            name='F1 Score'
        ))
        
        fig.update_layout(
            title='Model Performance Over Time',
            xaxis_title='Time',
            yaxis_title='Score',
            hovermode='x unified'
        )
        
        # Save report
        report_path = f'/reports/performance_{datetime.now().strftime("%Y%m%d")}.html'
        fig.write_html(report_path)
        
        # Calculate summary statistics
        summary = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'avg_accuracy': metrics_df['accuracy'].mean(),
            'avg_f1_score': metrics_df['f1_score'].mean(),
            'total_predictions': len(metrics_df) * 1000,  # Approximate
            'avg_processing_time': metrics_df['processing_time_avg'].mean(),
            'max_queue_size': metrics_df['queue_size_avg'].max()
        }
        
        # Store report metadata
        self.db.collection('performance_reports').add(summary)
        
        # Send daily summary
        await self.alert_manager.send_daily_summary(summary, report_path)
    
    def load_reference_distributions(self):
        """Load reference feature distributions"""
        try:
            # Load from Cloud Storage or database
            ref_data = pd.read_parquet('gs://your-bucket/reference_distributions.parquet')
            
            for column in ref_data.columns:
                self.reference_distributions[column] = ref_data[column].values
                
            logger.info(f"Loaded reference distributions for {len(self.reference_distributions)} features")
            
        except Exception as e:
            logger.error(f"Failed to load reference distributions: {e}")


class AlertManager:
    """Manage multi-channel alerting"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize alert channels
        self.email_client = self._init_email()
        self.slack_client = self._init_slack()
        self.twilio_client = self._init_twilio()
        
        # Alert history for deduplication
        self.alert_history = deque(maxlen=1000)
        
    def _init_email(self):
        """Initialize email client"""
        return {
            'smtp_server': self.config.get('smtp_server'),
            'smtp_port': self.config.get('smtp_port', 587),
            'username': self.config.get('email_username'),
            'password': self.config.get('email_password')
        }
    
    def _init_slack(self):
        """Initialize Slack client"""
        if self.config.get('slack_token'):
            return slack_sdk.WebClient(token=self.config['slack_token'])
        return None
    
    def _init_twilio(self):
        """Initialize Twilio client for SMS"""
        if self.config.get('twilio_account_sid'):
            return TwilioClient(
                self.config['twilio_account_sid'],
                self.config['twilio_auth_token']
            )
        return None
    
    async def send_alert(self, severity: str, title: str, message: str, 
                        channel: Optional[str] = None, **kwargs):
        """Send alert through appropriate channels"""
        # Create alert record
        alert = {
            'severity': severity,
            'title': title,
            'message': message,
            'timestamp': datetime.now(),
            'additional_data': kwargs
        }
        
        # Check for duplicate alerts
        alert_hash = hash(f"{severity}:{title}:{message[:50]}")
        if alert_hash in [a['hash'] for a in self.alert_history]:
            logger.debug(f"Suppressing duplicate alert: {title}")
            return
        
        alert['hash'] = alert_hash
        self.alert_history.append(alert)
        
        # Determine channels based on severity
        if not channel:
            if severity == 'CRITICAL':
                channels = ['email', 'slack', 'sms']
            elif severity == 'HIGH':
                channels = ['email', 'slack']
            elif severity == 'MEDIUM':
                channels = ['slack']
            else:
                channels = ['slack']
        else:
            channels = [channel]
        
        # Send through each channel
        for ch in channels:
            try:
                if ch == 'email':
                    await self._send_email_alert(alert)
                elif ch == 'slack':
                    await self._send_slack_alert(alert)
                elif ch == 'sms':
                    await self._send_sms_alert(alert)
            except Exception as e:
                logger.error(f"Failed to send alert via {ch}: {e}")
    
    async def _send_email_alert(self, alert: Dict):
        """Send email alert"""
        if not self.email_client['username']:
            return
            
        msg = MIMEMultipart()
        msg['From'] = self.email_client['username']
        msg['To'] = self.config.get('alert_email_to', '')
        msg['Subject'] = f"[{alert['severity']}] {alert['title']}"
        
        body = f"""
        Severity: {alert['severity']}
        Time: {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
        
        {alert['message']}
        
        Additional Information:
        {json.dumps(alert.get('additional_data', {}), indent=2)}
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        with smtplib.SMTP(self.email_client['smtp_server'], self.email_client['smtp_port']) as server:
            server.starttls()
            server.login(self.email_client['username'], self.email_client['password'])
            server.send_message(msg)
    
    async def _send_slack_alert(self, alert: Dict):
        """Send Slack alert"""
        if not self.slack_client:
            return
            
        color_map = {
            'CRITICAL': '#FF0000',
            'HIGH': '#FFA500',
            'MEDIUM': '#FFFF00',
            'LOW': '#00FF00',
            'INFO': '#0000FF'
        }
        
        self.slack_client.chat_postMessage(
            channel=self.config.get('slack_channel', '#alerts'),
            attachments=[{
                'color': color_map.get(alert['severity'], '#808080'),
                'title': alert['title'],
                'text': alert['message'],
                'fields': [
                    {
                        'title': 'Severity',
                        'value': alert['severity'],
                        'short': True
                    },
                    {
                        'title': 'Time',
                        'value': alert['timestamp'].strftime('%H:%M:%S'),
                        'short': True
                    }
                ],
                'footer': 'Fiber Inspection System',
                'ts': int(alert['timestamp'].timestamp())
            }]
        )
    
    async def _send_sms_alert(self, alert: Dict):
        """Send SMS alert for critical issues"""
        if not self.twilio_client or alert['severity'] != 'CRITICAL':
            return
            
        message = f"CRITICAL ALERT: {alert['title']}\n{alert['message'][:100]}"
        
        self.twilio_client.messages.create(
            body=message,
            from_=self.config['twilio_from_number'],
            to=self.config['alert_phone_number']
        )
    
    async def send_daily_summary(self, summary: Dict, report_path: str):
        """Send daily performance summary"""
        subject = f"Daily Fiber Inspection Report - {summary['date']}"
        
        body = f"""
        Daily Performance Summary
        ========================
        
        Date: {summary['date']}
        
        Key Metrics:
        - Average Accuracy: {summary['avg_accuracy']:.2%}
        - Average F1 Score: {summary['avg_f1_score']:.3f}
        - Total Predictions: {summary['total_predictions']:,}
        - Avg Processing Time: {summary['avg_processing_time']:.1f}ms
        - Max Queue Size: {summary['max_queue_size']:.0f}
        
        Detailed report attached.
        """
        
        # Send email with attachment
        msg = MIMEMultipart()
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        # Attach HTML report
        with open(report_path, 'rb') as f:
            attachment = MIMEText(f.read().decode(), 'html')
            attachment.add_header('Content-Disposition', 'attachment', 
                                filename=os.path.basename(report_path))
            msg.attach(attachment)
        
        # Send to distribution list
        # Implementation depends on your email setup


async def main():
    """Main entry point for monitoring system"""
    config = {
        'project_id': 'your-project-id',
        'mlflow_uri': 'http://mlflow-server:5000',
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'email_username': 'alerts@your-domain.com',
        'email_password': 'your-password',
        'alert_email_to': 'team@your-domain.com',
        'slack_token': 'xoxb-your-slack-token',
        'slack_channel': '#fiber-alerts',
        'twilio_account_sid': 'your-account-sid',
        'twilio_auth_token': 'your-auth-token',
        'twilio_from_number': '+1234567890',
        'alert_phone_number': '+0987654321'
    }
    
    monitoring = AdvancedMonitoringSystem(config)
    await monitoring.start()


if __name__ == "__main__":
    asyncio.run(main())
