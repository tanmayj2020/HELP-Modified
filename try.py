latency = 10
min_val = 3
max_val = 9 
portion = 0.3


a = (latency - min_val) / (max_val - min_val) * portion + (1 - portion) / 2
print(a)