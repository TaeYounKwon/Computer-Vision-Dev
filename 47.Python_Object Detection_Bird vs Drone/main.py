from multiprocessing import Process, Queue 
import time # 시간에 관한 작업 
import psutil # CPU 와 RAM 메모리에 관한 작업 

import detect 
import alarm 

start = time.perf_counter()

result = Queue()
th1 = Process(target=)
th2 = Process(target=work, args=(2, END//2, END, result))

th1.start()
th2.start()
th1.join()
th2.join()
# AFTER  code
memory_usage_dict = dict(psutil.virtual_memory()._asdict())
memory_usage_percent = memory_usage_dict['percent']
print(f"AFTER  CODE: memory_usage_percent: {memory_usage_percent}%")
# current process RAM usage
pid = os.getpid()
current_process = psutil.Process(pid)
current_process_memory_usage_as_KB = current_process.memory_info()[0] / 2.**20
print(f"AFTER  CODE: Current memory KB   : {current_process_memory_usage_as_KB: 9.3f} KB")
result.put('STOP')
total = 0
while True:
    tmp = result.get()
    if tmp == 'STOP':
        break
    else:
        total += tmp

finish = time.perf_counter()
print(f'Finished in {round(finish-start, 2)} second(s)')