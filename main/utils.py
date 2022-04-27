import time

def loadbar(iteration, total, prefix="", suffix="", decimal=0, 
            length=100, fill="=", extras=""):
    per_val = iteration*100/float(total)
    
    percent = ("{0:." + str(decimal) + "f}").format(per_val)
    
    cur_percent = ( ' ' * (len(str(total))-len(str(round(per_val)))) + percent)
    
    filledLen = int(length * iteration//total)
    if per_val == 100:
        bar = fill * filledLen + "." * (length - filledLen)
    else:
        bar = fill * filledLen + ">" + "." * (length - filledLen - 1)
    print(f"\r{prefix} [{bar}] {cur_percent}% {suffix}", end="\r")
    if iteration == total: 
        print(f"\r{prefix} [{bar}] {cur_percent}% {suffix} {extras}", end="\n")
        
    time.sleep(0.1)