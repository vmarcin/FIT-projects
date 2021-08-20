#!/usr/bin/env python3
"""
Dynamic analyser of a cart controller.
"""
import enum
from functools import total_ordering

@total_ordering
class Status(enum.Enum):
    "monitor status"
    Fails = 3   # the property cannot be met
    Pending = 2 # future obligations haven't been met and the property may or may not be valid
    Risk = 1    # holds but property can be violated in future
    Holds = 0   # holds strongly
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

class Monitor:
    def __init__(self, status, slots=None, src=None, dst=None, content=None):
        self._slots  = slots
        self._status = status
        self._src = src
        self._dst = dst
        self._unloaded = False
        self._content = content
        self._loaded = False
        self._loaded_weight = 0
        self._cargo_count = 0
    
    @property
    def status(self):
        return self._status

    @property
    def src(self):
        return self._src

    @property
    def dst(self):
        return self._dst
    
    @property
    def unloaded(self):
        return self._unloaded
    
    @property
    def content(self):
        return self._content
    
    @property
    def loaded(self):
        return self._loaded
    
    @property
    def loaded_weight(self):
        return self._loaded_weight
    
    @property
    def cargo_count(self):
        return self._cargo_count
    
    def get_slot(self, slot):
        return self._slots[slot]
    
    def set_slot(self, slot, value):
        self._slots[slot] = value
    
    def set_status(self, status):
        self._status = status
    
    def destination_reached(self):
        self._unloaded = True

    def content_loaded(self):
        self._loaded = True

    def add_weight(self, weight):
        self._loaded_weight += weight
    
    def remove_weight(self, weight):
        self._loaded_weight -= weight

    def add_cargo(self):
        self._cargo_count += 1

    def remove_cargo(self):
        self._cargo_count -= 1

class Monitoring:
    monitor1 = Monitor(Status.Risk, slots = [0,0,0,0]) # slot = 1 means bussy 
    monitor2 = Monitor(Status.Risk, slots = [1,1,1,1]) # slot = 1 means free 
    # if empty the global state is Risk so Risk monitor inserted
    # parametric each request append a new monitor
    monitor3 = [Monitor(Status.Risk)] 
    # optional
    # parametric each request append a new monitor
    monitor4 = [Monitor(Status.Risk)]
    # if empty the global state is Risk som Risk monitor inserted
    # parametric each request append a new monitor
    monitor5 = [Monitor(Status.Risk)] 
    monitor6 = Monitor(Status.Risk)
    monitor7 = Monitor(Status.Risk)
    # optional
    monitor8 = None
    # optional
    monitor9 = None
    # coverege for station-slot combination
    coverage = dict({"A": [0,0,0,0], "B" : [0,0,0,0], "C" : [0,0,0,0], "D" : [0,0,0,0] })

    @staticmethod
    def load(time, slot, weight, pos, content):
        succ = True

        # check properties
        exist_request = False
        req_index = 0
        for monitor in Monitoring.monitor5:
            if (monitor.src == pos and monitor.content == content):
                exist_request = True
                break
            req_index += 1

        if (Monitoring.monitor1.get_slot(slot)):
            print('%d:error: loading into an occupied slot #%d' % (time,slot))
            Monitoring.monitor1.set_status(Status.Fails)
            succ &= False
        if (Monitoring.monitor6.cargo_count + 1 > 4):
            print('%d:error: cannot load more than 4 cargos' % (time))
            Monitoring.monitor6.set_status(Status.Fails)
            succ &= False
        if (Monitoring.monitor7.loaded_weight + weight > 150):
            print('%d:error: the cart has been overloaded %d > 150' % (time, Monitoring.monitor7.loaded_weight + weight))
            Monitoring.monitor7.set_status(Status.Fails)
            succ &= False
        if (not(exist_request)):
            print('%d:error: trying to load an unrequested cargo %s at pos %s' % (time, content, pos))
            Monitoring.monitor5.append(Monitor(Status.Fails))
            succ &= False

        # 'content' will be loaded (all properties hold) so update monitors
        if (succ):
            # MONITOR 1 update
            Monitoring.monitor1.set_slot(slot, 1)
            # MONITOR 2 update
            Monitoring.monitor2.set_slot(slot, 0)
            # MONITOR 6 update
            Monitoring.monitor6.add_cargo()
            # MONITOR 7 update
            Monitoring.monitor7.add_weight(weight)
            # MONITOR 3 update
            # sets that cart has been loaded with content 
            for monitor in Monitoring.monitor3:
                if (monitor.src == pos and monitor.content == content):
                    monitor.content_loaded()
            # MONITOR 5 update
            # remove already processed requests
            Monitoring.monitor5.pop(req_index)
            # MONITOR 4
            index = 0
            for monitor in Monitoring.monitor4:
                if (monitor.src == pos and monitor.content == content):
                    break
                index += 1
            # monitor for acutally loaded cargo fullfiled (T)
            Monitoring.monitor4.pop(index)

            # remember that 'pos' 'slot' combination covered
            # only succesfull loads count
            Monitoring.coverage[pos][slot] = 1

    @staticmethod
    def unload(time, slot, weight, pos, content):
        succ = True
        
        # check properties 
        if (Monitoring.monitor2.get_slot(slot)):
            print('%d:error: unloading from an empty slot #%d' % (time,slot))
            Monitoring.monitor2.set_status(Status.Fails)
            succ &= False
        
        # cargo will be unloaded (all properites hold) so update monitors
        if(succ):
            # MONITOR 1 update
            Monitoring.monitor1.set_slot(slot, 0)
            # MONITOR 2 update
            Monitoring.monitor2.set_slot(slot, 1)
            # MONITOR 3 update
            # check if some request should be unload here
            for monitor in Monitoring.monitor3:
                if (monitor.dst == pos and monitor.loaded and monitor.content == content):
                    monitor.destination_reached()
            # MONITOR 6 update
            Monitoring.monitor6.remove_cargo()
            # MONITOR 7 update
            Monitoring.monitor7.remove_weight(weight)

    
    @staticmethod
    def move(time, pos1, pos2):
        # check if all request were unload at pos1
        not_processed = []
        for monitor in Monitoring.monitor3:
            if (monitor.loaded):
                if (monitor.dst == pos1 and not(monitor.unloaded)):
                    print('%d:error: the cart did not unload the %s at station %s' % (time, monitor.content, pos1))
                    monitor.set_status(Status.Fails)
                elif (monitor.dst == pos1 and monitor.unloaded):
                    # not necessarily needed cause the monitor will be removed
                    monitor.set_status(Status.Holds)
            else:
                not_processed.append(monitor)
    
        Monitoring.monitor3 = not_processed
    
    @staticmethod
    def stop(time):
        # at the end check if all cargos have been unloaded
        for monitor in Monitoring.monitor3:
            if (not(monitor.unloaded) and monitor.loaded):
                print('%d:error: the cart did not unload the %s at station %s' % (time, monitor.content, monitor.dst))
                monitor.set_status(Status.Fails)
        for monitor in Monitoring.monitor4:
            if (monitor.content != None):
                print('%d:error: the %s requested at %s but never loaded' % (time, monitor.content, monitor.src))
                monitor.set_status(Status.Fails)

def report_coverage():
    "Coverage reporter"

    # check if fail occured
    statuses = []
    for m in Monitoring.monitor3:
        statuses.append(m.status)
    for m in Monitoring.monitor4:
        statuses.append(m.status)
    for m in Monitoring.monitor5:
        statuses.append(m.status)
    statuses.append(Monitoring.monitor1.status) 
    statuses.append(Monitoring.monitor2.status) 
    statuses.append(Monitoring.monitor6.status) 
    statuses.append(Monitoring.monitor7.status) 

    if (Status.Fails != max(statuses)):
        print("All properties hold.")

    covered = 0
    for station in ['A', 'B', 'C', 'D']:
        # adds number of covered slots in the 'station'
        covered += Monitoring.coverage[station].count(1)
    print('CartCoverage %d%%' % ((covered/16)*100))

def onrequest(time, src, dst, content, weight):
    Monitoring.monitor3.append(Monitor(Status.Risk, src=src, dst=dst, content=content))
    Monitoring.monitor4.append(Monitor(Status.Pending, src=src, dst=dst, content=content))
    Monitoring.monitor5.append(Monitor(Status.Risk, src=src, dst=dst, content=content))

def onmoving(time, pos1, pos2):
    time = int(time)
    Monitoring.move(time, pos1, pos2)

def onloading(time, pos, content, weight, slot):
    time = int(time)
    slot = int(slot)
    weight = int(weight)
    Monitoring.load(time, slot, weight, pos, content)

def onunloading(time, pos, content, weight, slot):
    time = int(time)
    slot = int(slot)
    weight = int(weight)
    Monitoring.unload(time, slot, weight, pos, content)

def onstop(time):
    time = int(time)
    Monitoring.stop(time)

def onevent(event):
    event_id = event[1]
    del(event[1])
    
    # call appropriate action
    if event_id == 'moving':
        onmoving(*event)
    elif event_id == 'loading':
        onloading(*event)
    elif event_id == 'unloading':
        onunloading(*event)
    elif event_id == 'requesting':
        onrequest(*event)
    elif event_id == 'stop':
        onstop(*event)

###########################################################
# Nize netreba menit.

def monitor(reader):
    "Main function"
    for line in reader:
        line = line.strip()
        onevent(line.split())
    report_coverage()

if __name__ == "__main__":
    import sys
    monitor(sys.stdin)