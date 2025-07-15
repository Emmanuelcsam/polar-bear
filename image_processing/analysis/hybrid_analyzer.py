import json
import numpy as np
import os

def analyze_hybrid():
    """Combine neural network and vision processing results"""
    
    hybrid_results = {
        'neural_vision_correlation': {},
        'predictive_features': {},
        'advanced_patterns': {},
        'synthesis': {}
    }
    
    # Load neural results
    neural_data = {}
    if os.path.exists('neural_results.json'):
        with open('neural_results.json', 'r') as f:
            neural_data = json.load(f)
        print("[HYBRID] Loaded neural network results")
    
    # Load vision results
    vision_data = {}
    if os.path.exists('vision_results.json'):
        with open('vision_results.json', 'r') as f:
            vision_data = json.load(f)
        print("[HYBRID] Loaded vision processing results")
    
    # Load original patterns
    pattern_data = {}
    if os.path.exists('patterns.json'):
        with open('patterns.json', 'r') as f:
            pattern_data = json.load(f)
    
    # 1. Correlate neural predictions with vision features
    if neural_data and vision_data:
        # Check if edge regions correspond to predicted values
        if 'predictions' in neural_data and 'edges' in vision_data:
            edge_ratio = vision_data['edges']['canny_edge_ratio']
            predictions = neural_data['predictions']
            
            # High-frequency predictions might correlate with edges
            pred_variance = np.var(predictions) if predictions else 0
            
            hybrid_results['neural_vision_correlation'] = {
                'edge_prediction_correlation': float(pred_variance * edge_ratio),
                'complexity_score': float(edge_ratio * len(predictions)),
                'neural_edge_agreement': pred_variance > 1000 and edge_ratio > 0.1
            }
        
        # Texture complexity vs neural patterns
        if 'texture' in vision_data and 'predictions' in neural_data:
            texture_entropy = vision_data['texture']['lbp_entropy']
            pred_entropy = calculate_entropy(neural_data['predictions'])
            
            hybrid_results['neural_vision_correlation']['entropy_ratio'] = (
                float(pred_entropy / texture_entropy) if texture_entropy > 0 else 0
            )
    
    # 2. Predictive features based on vision
    if vision_data and neural_data:
        features = []
        
        # Corner regions might have specific patterns
        if 'corners' in vision_data:
            corner_density = vision_data['corners']['corner_density']
            features.append({
                'feature': 'corner_density',
                'value': corner_density,
                'prediction_impact': 'high' if corner_density > 0.01 else 'low'
            })
        
        # Contour complexity
        if 'contours' in vision_data:
            contour_count = vision_data['contours']['total_count']
            features.append({
                'feature': 'contour_complexity',
                'value': contour_count,
                'prediction_impact': 'high' if contour_count > 50 else 'medium'
            })
        
        hybrid_results['predictive_features'] = features
    
    # 3. Advanced pattern detection
    if all([neural_data, vision_data, pattern_data]):
        # Combine all pattern sources
        patterns = []
        
        # Neural sequential patterns
        if 'predictions' in neural_data:
            preds = neural_data['predictions']
            for i in range(len(preds) - 5):
                window = preds[i:i+5]
                if is_interesting_pattern(window):
                    patterns.append({
                        'type': 'neural_sequence',
                        'values': window,
                        'position': i
                    })
        
        # Vision geometric patterns
        if 'features' in vision_data:
            hu_moments = vision_data['features']['hu_moments']
            if hu_moments:
                patterns.append({
                    'type': 'shape_descriptor',
                    'hu_moments': hu_moments,
                    'shape_class': classify_shape(hu_moments)
                })
        
        # Statistical patterns from original analysis
        if 'statistics' in pattern_data:
            stats = pattern_data['statistics']
            if stats:
                patterns.append({
                    'type': 'statistical',
                    'mean': stats[0].get('mean', 0),
                    'std': stats[0].get('std', 0)
                })
        
        hybrid_results['advanced_patterns'] = patterns[:20]  # Limit to 20
    
    # 4. Synthesis - combine everything
    synthesis = synthesize_insights(neural_data, vision_data, pattern_data)
    hybrid_results['synthesis'] = synthesis
    
    # Save results
    with open('hybrid_analysis.json', 'w') as f:
        json.dump(hybrid_results, f)
    
    print("[HYBRID] Analysis complete")
    print(f"[HYBRID] Found {len(hybrid_results['advanced_patterns'])} advanced patterns")
    
    # Generate recommendations
    generate_recommendations(hybrid_results)

def calculate_entropy(values):
    """Calculate Shannon entropy"""
    if not values:
        return 0
    
    # Create histogram
    hist, _ = np.histogram(values, bins=50)
    hist = hist[hist > 0]
    
    if len(hist) == 0:
        return 0
    
    # Calculate probabilities
    probs = hist / np.sum(hist)
    
    # Calculate entropy
    entropy = -np.sum(probs * np.log2(probs))
    
    return float(entropy)

def is_interesting_pattern(window):
    """Check if a sequence is an interesting pattern"""
    if len(window) < 2:
        return False
    
    # Check for ascending/descending
    if all(window[i] <= window[i+1] for i in range(len(window)-1)):
        return True
    if all(window[i] >= window[i+1] for i in range(len(window)-1)):
        return True
    
    # Check for alternating
    if all(window[i] < window[i+1] if i % 2 == 0 else window[i] > window[i+1] 
           for i in range(len(window)-1)):
        return True
    
    # Check for periodic
    if len(set(window[::2])) == 1 and len(set(window[1::2])) == 1:
        return True
    
    return False

def classify_shape(hu_moments):
    """Classify shape based on Hu moments"""
    if not hu_moments or len(hu_moments) < 2:
        return "unknown"
    
    # Simple classification based on moment ratios
    m1, m2 = abs(hu_moments[0]), abs(hu_moments[1])
    
    if m1 < 0.01 and m2 < 0.01:
        return "uniform"
    elif m1 / (m2 + 1e-10) > 10:
        return "elongated"
    elif abs(m1 - m2) < 0.1:
        return "circular"
    else:
        return "irregular"

def synthesize_insights(neural, vision, patterns):
    """Synthesize insights from all sources"""
    synthesis = {
        'complexity_level': 'unknown',
        'dominant_characteristics': [],
        'recommended_processing': [],
        'quality_score': 0
    }
    
    scores = []
    
    # Assess complexity
    if vision and 'edges' in vision:
        edge_score = vision['edges']['canny_edge_ratio'] * 100
        scores.append(edge_score)
        
        if edge_score > 20:
            synthesis['dominant_characteristics'].append('high_edge_density')
        elif edge_score < 5:
            synthesis['dominant_characteristics'].append('smooth_regions')
    
    if neural and 'predictions' in neural:
        pred_variance = np.var(neural['predictions']) if neural['predictions'] else 0
        neural_score = min(100, pred_variance / 100)
        scores.append(neural_score)
        
        if pred_variance > 5000:
            synthesis['dominant_characteristics'].append('high_variability')
    
    if patterns and 'statistics' in patterns:
        stats = patterns['statistics']
        if stats and len(stats) > 0:
            pattern_score = min(100, stats[0].get('std', 0) / 2.55)
            scores.append(pattern_score)
    
    # Overall complexity
    avg_score = np.mean(scores) if scores else 50
    
    if avg_score > 70:
        synthesis['complexity_level'] = 'high'
        synthesis['recommended_processing'] = [
            'advanced_filtering',
            'deep_learning',
            'multi_scale_analysis'
        ]
    elif avg_score > 30:
        synthesis['complexity_level'] = 'medium'
        synthesis['recommended_processing'] = [
            'standard_filtering',
            'pattern_matching',
            'statistical_analysis'
        ]
    else:
        synthesis['complexity_level'] = 'low'
        synthesis['recommended_processing'] = [
            'basic_processing',
            'histogram_analysis',
            'simple_transforms'
        ]
    
    synthesis['quality_score'] = float(avg_score)
    
    return synthesis

def generate_recommendations(results):
    """Generate processing recommendations"""
    
    recommendations = {
        'immediate_actions': [],
        'future_analysis': [],
        'parameter_tuning': {}
    }
    
    # Based on correlations
    if 'neural_vision_correlation' in results:
        corr = results['neural_vision_correlation']
        
        if corr.get('neural_edge_agreement', False):
            recommendations['immediate_actions'].append(
                "Focus on edge regions for pattern extraction"
            )
    
    # Based on synthesis
    if 'synthesis' in results:
        synth = results['synthesis']
        
        if synth['complexity_level'] == 'high':
            recommendations['immediate_actions'].extend([
                "Apply noise reduction before further analysis",
                "Use larger neural network architectures"
            ])
            recommendations['parameter_tuning'] = {
                'neural_epochs': 100,
                'edge_threshold': [30, 100],
                'pattern_window': 20
            }
        elif synth['complexity_level'] == 'low':
            recommendations['immediate_actions'].append(
                "Simple threshold-based processing sufficient"
            )
            recommendations['parameter_tuning'] = {
                'neural_epochs': 30,
                'edge_threshold': [50, 150],
                'pattern_window': 5
            }
    
    # Future analysis suggestions
    if 'advanced_patterns' in results:
        pattern_types = [p['type'] for p in results['advanced_patterns']]
        
        if 'shape_descriptor' in pattern_types:
            recommendations['future_analysis'].append(
                "Shape matching with template database"
            )
        
        if 'neural_sequence' in pattern_types:
            recommendations['future_analysis'].append(
                "Time series forecasting with LSTM"
            )
    
    # Save recommendations
    with open('hybrid_recommendations.json', 'w') as f:
        json.dump(recommendations, f)
    
    print("[HYBRID] Generated recommendations")
    for action in recommendations['immediate_actions'][:3]:
        print(f"[HYBRID] Recommended: {action}")

if __name__ == "__main__":
    analyze_hybrid()