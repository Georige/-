# 修复了0-1之间的小数无法搜索平方根的bug
x = 0.3
epsilon = 0.0001
low = 0.0
high = x
guess = (high + low)/2
n = 0

if x < 1:
    low = x
    high = 1
    guess = (high + low)/2

while abs(guess**2 - x) >= epsilon:
    n += 1
    if guess**2 < x:
        low = guess
    else:
        high = guess
    guess = (high + low)/2
    print(guess)
    
print('num of loops = ', n)
print(guess, 'is close to the square root of', x)