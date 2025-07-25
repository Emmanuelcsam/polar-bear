import pickle
import numpy as np
import json
from datetime import datetime
import os
from connector_interface import setup_connector, send_hivemind_status

print("Learning Optimizer starting...")

# Setup hivemind connector
connector = setup_connector('learning-optimizer.py')
connector.register_parameter('boost_threshold', 0.7, 'Confidence threshold for boosting')
connector.register_parameter('reduce_threshold', 0.4, 'Confidence threshold for reducing')
connector.register_parameter('popularity_threshold', 0.1, 'Popularity threshold for boosting')

# Load current state
with open('pixel_db.pkl', 'rb') as f:
    pixel_db = pickle.load(f)

weights = {cat: 1.0 for cat in pixel_db}
if os.path.exists('weights.pkl'):
    with open('weights.pkl', 'rb') as f:
        weights = pickle.load(f)

history = []
if os.path.exists('learning_history.pkl'):
    with open('learning_history.pkl', 'rb') as f:
        history = pickle.load(f)

print(f"Current weights: {len(weights)} categories")
print(f"Learning history: {len(history)} entries")

send_hivemind_status({
    'status': 'loaded',
    'categories': len(weights),
    'history_entries': len(history)
}, connector)

while True:
    print("\n" + "="*50)
    print("1. Optimize from results file")
    print("2. Show learning progress")
    print("3. Reset weights")
    print("4. Prune weak categories")
    print("5. Exit")
    
    choice = input("\nChoice: ")
    
    if choice == '1':
        results_file = input("Results JSON file: ")
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        print("Learning from categorizations...")
        
        # Calculate category performance
        cat_stats = {}
        for img, data in results.items():
            cat = data['category']
            conf = data['confidence']
            
            if cat not in cat_stats:
                cat_stats[cat] = {'total': 0, 'confidence': 0, 'count': 0}
            
            cat_stats[cat]['total'] += conf
            cat_stats[cat]['count'] += 1
            cat_stats[cat]['confidence'] = cat_stats[cat]['total'] / cat_stats[cat]['count']
        
        # Get thresholds from hivemind
        boost_threshold = connector.get_parameter('boost_threshold', 0.7)
        reduce_threshold = connector.get_parameter('reduce_threshold', 0.4)
        popularity_threshold = connector.get_parameter('popularity_threshold', 0.1)

        # Update weights based on performance
        updates_made = 0
        for cat, stats in cat_stats.items():
            avg_conf = stats['confidence']
            count = stats['count']
            
            # Categories with higher average confidence get boosted
            if avg_conf > boost_threshold:
                weights[cat] *= 1.05
                print(f"  ↑ Boosting {cat} (avg conf: {avg_conf:.2f})")
                updates_made += 1
            elif avg_conf < reduce_threshold:
                weights[cat] *= 0.95
                print(f"  ↓ Reducing {cat} (avg conf: {avg_conf:.2f})")
                updates_made += 1
            
            # Popular categories get slight boost
            if count > len(results) * popularity_threshold:
                weights[cat] *= 1.02
                print(f"  + Popular category {cat} ({count} images)")
                updates_made += 1
        
        # Normalize weights
        max_weight = max(weights.values())
        weights = {k: v/max_weight for k, v in weights.items()}
        
        # Save history
        history.append({
            'timestamp': datetime.now().isoformat(),
            'results_file': results_file,
            'cat_stats': cat_stats,
            'weights': weights.copy()
        })
        
        with open('weights.pkl', 'wb') as f:
            pickle.dump(weights, f)
        with open('learning_history.pkl', 'wb') as f:
            pickle.dump(history, f)
        
        print("✓ Weights updated and saved")
        
        send_hivemind_status({
            'status': 'optimization_complete',
            'updates_made': updates_made,
            'results_file': results_file,
            'categories_processed': len(cat_stats)
        }, connector)
    
    elif choice == '2':
        if not history:
            print("No learning history yet")
            continue
            
        print("\nLearning Progress:")
        for i, entry in enumerate(history[-10:]):  # Last 10 entries
            print(f"\n{i+1}. {entry['timestamp']}")
            print(f"   File: {entry['results_file']}")
            print("   Top categories:")
            sorted_cats = sorted(entry['cat_stats'].items(), 
                               key=lambda x: x[1]['count'], reverse=True)[:5]
            for cat, stats in sorted_cats:
                print(f"     {cat}: {stats['count']} images, "
                      f"avg confidence: {stats['confidence']:.2f}")
    
    elif choice == '3':
        confirm = input("Reset all weights to 1.0? (y/n): ")
        if confirm == 'y':
            weights = {cat: 1.0 for cat in pixel_db}
            with open('weights.pkl', 'wb') as f:
                pickle.dump(weights, f)
            print("✓ Weights reset")
    
    elif choice == '4':
        threshold = float(input("Minimum weight threshold (e.g., 0.5): "))
        pruned = [cat for cat, w in weights.items() if w < threshold]
        
        if pruned:
            print(f"\nWill prune {len(pruned)} categories:")
            for cat in pruned[:10]:
                print(f"  - {cat} (weight: {weights[cat]:.3f})")
            if len(pruned) > 10:
                print(f"  ... and {len(pruned)-10} more")
            
            confirm = input("\nProceed? (y/n): ")
            if confirm == 'y':
                for cat in pruned:
                    del weights[cat]
                    if cat in pixel_db:
                        del pixel_db[cat]
                
                with open('weights.pkl', 'wb') as f:
                    pickle.dump(weights, f)
                with open('pixel_db.pkl', 'wb') as f:
                    pickle.dump(pixel_db, f)
                
                print(f"✓ Pruned {len(pruned)} categories")
        else:
            print("No categories below threshold")
    
    elif choice == '5':
        break

print("\nOptimizer shutting down...")
send_hivemind_status({'status': 'shutting_down'}, connector)