Formula for each layer:
n_out = (n_in + 2*p - k)/s + 1
RF = RF_prev + (k-1)*j_prev

where:
n_out = output size
n_in = input size
p = padding
k = kernel size
s = stride
j = jump/stride of previous layer

Initial RF = 1x1

1. convblock1: 3x3 conv, padding=0
   Input: 28x28
   RF_in = 1, j_in = 1
   RF = RF_in + (k-1)*j_in = 1 + (3-1)*1 = 3
   j_out = 1
   Output: 26x26

2. convblock2: 3x3 conv, padding=0
   RF_in = 3, j_in = 1
   RF = 3 + (3-1)*1 = 5
   j_out = 1
   Output: 24x24

3. convblock3: 1x1 conv, padding=0
   RF_in = 5, j_in = 1
   RF = 5 (no change as kernel is 1x1)
   j_out = 1
   Output: 24x24

4. pool1: 2x2 MaxPool, stride=2
   RF_in = 5, j_in = 1
   RF = 5 + (2-1)*1 = 6
   j_out = 2
   Output: 12x12
 5. convblock4: 3x3 conv, padding=0
   RF_in = 6, j_in = 2
   RF = 6 + (3-1)*2 = 10
   j_out = 2
   Output: 10x10

6. gap: 6x6 AvgPool
   RF_in = 10, j_in = 2
   RF = 10 + (6-1)*2 = 20
   Output: 1x1

7. convblock8: 1x1 conv
   RF = 20 (no change as kernel is 1x1)
   Output: 1x1

Final Receptive Field: 20x20
