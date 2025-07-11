import numpy as np; from scipy import stats
data = []
while len(data) < 100:
    try: line = input().split(); data.append((int(line[0]), float(line[1].rstrip('%'))))
    except: pass
x, y = zip(*data)
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
print(f"y={slope:.4f}x+{intercept:.4f} r²={r_value**2:.4f}")
poly = np.polyfit(x, y, 3)
print(f"cubic: {' '.join(f'{c:.4e}' for c in poly)}")
print(f"peaks: {[x[i] for i in range(1,len(y)-1) if y[i]>y[i-1] and y[i]>y[i+1]]}")
