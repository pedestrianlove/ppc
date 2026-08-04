# cp3a: Fastest double

## 1. baseline
```
test                    time  result
benchmarks/1.txt      0.014s  pass
benchmarks/2a.txt     0.234s  pass
benchmarks/2b.txt     0.236s  pass
benchmarks/2c.txt     0.236s  pass
benchmarks/3.txt      6.545s  pass
benchmarks/4.txt     28.909s  pass
```

## 2. Proper multicore
```
test                    time  result
benchmarks/1.txt      0.001s  pass
benchmarks/2a.txt     0.014s  pass
benchmarks/2b.txt     0.015s  pass
benchmarks/2c.txt     0.015s  pass
benchmarks/3.txt      0.379s  pass
benchmarks/4.txt      1.806s  pass
```

## 3. Fixing correctness
```
test                    time  result
benchmarks/1.txt      0.114s  pass
benchmarks/2a.txt     1.787s  pass
benchmarks/2b.txt     1.780s  pass
benchmarks/2c.txt     1.775s  pass
benchmarks/3.txt   [failed]
```
