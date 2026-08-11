# cp3a: Fastest double

- Ryzen 5700x

## 1. baseline
```
test                    time  result
benchmarks/1.txt      0.112s  pass
benchmarks/2a.txt     1.758s  pass
benchmarks/2b.txt     1.755s  pass
benchmarks/2c.txt     1.764s  pass
benchmarks/3.txt   [failed]
```

## 2. Normalize the problem
```
test                                time  result
benchmarks/1.txt                  0.071s  pass
benchmarks/2a.txt                 1.089s  pass
benchmarks/2b.txt                 1.072s  pass
benchmarks/2c.txt                 1.063s  pass
benchmarks/3.txt               [failed]
```

## 3. Add easy multicore
```
test                    time  result
benchmarks/1.txt      0.021s  pass
benchmarks/2a.txt     0.258s  pass
benchmarks/2b.txt     0.246s  pass
benchmarks/2c.txt     0.245s  pass
benchmarks/3.txt     19.150s  pass
benchmarks/4.txt   [failed]
```

## 4. New processor
- Ryzen 5950x
```
test                    time  result
benchmarks/1.txt      0.011s  pass
benchmarks/2a.txt     0.153s  pass
benchmarks/2b.txt     0.151s  pass
benchmarks/2c.txt     0.154s  pass
benchmarks/3.txt     15.903s  pass
benchmarks/4.txt   [failed]
```

## 5. 
