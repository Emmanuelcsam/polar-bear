import numpy as np
from scipy import stats
import sys
import json

def find_correlations(data=None):
    """Find correlations in data - can be called independently or via connector"""
    if data is None:
        # Independent mode - read from stdin
        data = []
        while len(data) < 100:
            try:
                line = input().split()
                data.append((int(line[0]), float(line[1].rstrip('%'))))
            except EOFError:
                break
            except:
                pass
    
    if len(data) < 2:
        return {"error": "Insufficient data points"}
    
    x, y = zip(*data)
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    linear_result = f"y={slope:.4f}x+{intercept:.4f} r²={r_value**2:.4f}"
    
    # Polynomial fitting
    poly = np.polyfit(x, y, 3)
    cubic_result = f"cubic: {' '.join(f'{c:.4e}' for c in poly)}"
    
    # Peak detection
    peaks = [x[i] for i in range(1, len(y)-1) if y[i] > y[i-1] and y[i] > y[i+1]]
    peaks_result = f"peaks: {peaks}"
    
    results = {
        "linear": linear_result,
        "cubic": cubic_result,
        "peaks": peaks_result,
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_value**2,
        "peak_count": len(peaks)
    }
    
    # Share results if running under connector
    if '_set_shared' in globals():
        _set_shared('correlation_results', results)
        _send_message('all', f"Correlation analysis complete: {len(data)} points analyzed")
    
    return results

def main():
    """Main function for independent execution"""
    results = find_correlations()
    if "error" not in results:
        print(results["linear"])
        print(results["cubic"])
        print(results["peaks"])
    else:
        print(f"Error: {results['error']}")

# Support both independent and connector-based execution
if __name__ == "__main__":
    main()
elif '_connector_control' in globals():
    # Running under connector control
    print("Correlation finder loaded under connector control")