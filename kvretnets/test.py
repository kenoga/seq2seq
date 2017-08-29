import time

time1 = time.time()
for i in range(1000000):
    pass
time2 = time.time()
print "%0.2f" % (time2 - time1)