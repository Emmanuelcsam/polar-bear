from PIL import Image; import numpy as np
img1 = np.array(Image.open("image1.jpg").convert('L'))
img2 = np.array(Image.open("image2.jpg").convert('L'))
diff = np.abs(img1.astype(int) - img2.astype(int))
threshold = np.percentile(diff, 95)
anomalies = np.where(diff > threshold)
hist1, hist2 = np.histogram(img1, 256, [0,256])[0], np.histogram(img2, 256, [0,256])[0]
kl_div = np.sum(hist1 * np.log((hist1+1e-10)/(hist2+1e-10)))
print(f"Anomalies: {len(anomalies[0])} pixels, KL-div: {kl_div:.4f}")
for i in range(min(10, len(anomalies[0]))): print(f"({anomalies[1][i]},{anomalies[0][i]}): {diff[anomalies[0][i],anomalies[1][i]]}")
