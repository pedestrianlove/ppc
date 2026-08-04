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

## 4. add easy multicore for cor computation
```
test                    time  result
benchmarks/1.txt      0.023s  pass
benchmarks/2a.txt     0.346s  pass
benchmarks/2b.txt     0.355s  pass
benchmarks/2c.txt     0.337s  pass
benchmarks/3.txt      7.776s  pass
benchmarks/4.txt     32.703s  pass
```
