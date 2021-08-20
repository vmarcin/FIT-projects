import os
import subprocess

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

print("Starting testing...")
if(not(os.path.exists("../plg-2-nka"))):
    print(f"{bcolors.FAIL}Fail: plg-2-nka not found!{bcolors.ENDC}")
    exit(1)
print("****************************************")
print("Valid tests...")
print("****************************************")
for i in range(1, 19):
    if i < 10:
        testname = "test0" + str(i)
    else:
        testname = "test" + str(i)

    # -i
    os.system(".././plg-2-nka -i " + testname + ".in > " + testname + "_i.temp")
    print(f"{bcolors.FAIL}")
    command = "diff " + testname + "_i.temp " + testname + "_i.out"
    popen = subprocess.Popen(command, shell=True)
    ret_code = popen.wait()
    print(f"{bcolors.ENDC}", end=" ")
    if(ret_code == 0):
        print("[TEST 0" + str(i) + "] ./plg-2-nka -i " + testname + ".in " + f"{bcolors.OKGREEN}OK{bcolors.ENDC}")
        os.system("rm " + testname + "_i.temp")
    else:
        print("[TEST 0" + str(i) + "] ./plg-2-nka -i " + testname + ".in " + f"{bcolors.FAIL}FAIL{bcolors.ENDC}")

    # -1
    os.system(".././plg-2-nka -1 " + testname + ".in > " + testname + "_1.temp")
    print(f"{bcolors.FAIL}")
    command = "diff " + testname + "_1.temp " + testname + "_1.out"
    popen = subprocess.Popen(command, shell=True)
    ret_code = popen.wait()
    print(f"{bcolors.ENDC}", end=" ")
    if(ret_code == 0):
        print("[TEST 0" + str(i) + "] ./plg-2-nka -1 " + testname + ".in " + f"{bcolors.OKGREEN}OK{bcolors.ENDC}")
        os.system("rm " + testname + "_1.temp")
    else:
        print("[TEST 0" + str(i) + "] ./plg-2-nka -1 " + testname + ".in " + f"{bcolors.FAIL}FAIL{bcolors.ENDC}")

    # -2
    os.system(".././plg-2-nka -2 " + testname + ".in > " + testname + "_2.temp")
    print(f"{bcolors.FAIL}")
    command = "diff " + testname + "_2.temp " + testname + "_2.out"
    popen = subprocess.Popen(command, shell=True)
    ret_code = popen.wait()
    print(f"{bcolors.ENDC}", end=" ")
    if(ret_code == 0):
        print("[TEST 0" + str(i) + "] ./plg-2-nka -2 " + testname + ".in " + f"{bcolors.OKGREEN}OK{bcolors.ENDC}")
        os.system("rm " + testname + "_2.temp")
    else:
        print("[TEST 0" + str(i) + "] ./plg-2-nka -2 " + testname + ".in " + f"{bcolors.FAIL}FAIL{bcolors.ENDC}") 

    print("_________________________________________")

print("****************************************")
print("Invalid tests...")
print("****************************************")
for i in range(1, 12):
    if i < 10:
        testname = "invalid_tests/test0" + str(i)
    else:
        testname = "invalid_tests/test" + str(i)

    # -i
    command = ".././plg-2-nka -i " + testname + ".in > " + testname + "_i.temp"
    print(f"{bcolors.BOLD}")
    popen = subprocess.Popen(command, shell=True)
    ret_code = popen.wait()
    print(f"{bcolors.ENDC}", end=" ")
    if(ret_code != 0):
        print("[TEST 0" + str(i) + "] ./plg-2-nka -i " + testname + ".in " + f"{bcolors.OKGREEN}OK{bcolors.ENDC}")
        os.system("rm " + testname + "_i.temp")
    else:
        print("[TEST 0" + str(i) + "] ./plg-2-nka -i " + testname + ".in " + f"{bcolors.FAIL}FAIL{bcolors.ENDC}")


    print("_________________________________________")