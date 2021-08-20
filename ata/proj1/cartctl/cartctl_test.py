#!/usr/bin/env python3
"""
Example of usage/test of Cart controller implementation.
"""

import sys
from cartctl import CartCtl, Status
from cart import Cart, CargoReq
from jarvisenv import Jarvis
import unittest

def log(msg):
    "simple logging"
    print('  %s' % msg)

class TestCartRequests(unittest.TestCase):
    
    def test_one(self):
        "Test ID 1"

        def add_load(c: CartCtl, cargo_req: CargoReq):
            log('%d: Requesting %s at %s' % (Jarvis.time(), cargo_req, cargo_req.src))
            c.request(cargo_req)

        def on_move(c: Cart):
            log('%d: Cart is moving %s->%s' % (Jarvis.time(), c.pos, c.data))

        def on_load(c: Cart, cargo_req: CargoReq, ctrl: CartCtl):
            log('%d: Cart at %s: loading: %s' % (Jarvis.time(), c.pos, cargo_req))
            log(c)
            cargo_req.context = "loaded"
            # vynimka false
            self.assertLess(Jarvis.time() - cargo_req.born, 120)
            # prioritny material false
            self.assertEqual(cargo_req.prio, False)
            # vozik v normalnom rezime
            self.assertEqual(ctrl.status, Status.Normal)

        def on_unload(c: Cart, cargo_req: CargoReq, ctrl: CartCtl):
            log('%d: Cart at %s: unloading: %s' % (Jarvis.time(), c.pos, cargo_req))
            log(c)
            # nalozi material true
            self.assertEqual('loaded', cargo_req.context)
            cargo_req.context = 'unloaded'
            if cargo_req.content == 'helmet':
                self.assertEqual('D', c.pos)
            if cargo_req.content == 'heart':
                self.assertEqual('A', c.pos)
                # vozik v normalnom rezime
                self.assertEqual(ctrl.status, Status.Normal)

        # Setup Cart
        # 4 slots, 150 kg max payload capacity, 2=max debug
        cart_dev = Cart(4, 150, 0)
        cart_dev.onmove = on_move

        # Setup Cart Controller
        c = CartCtl(cart_dev, Jarvis)

        # Setup Cargo to move
        helmet = CargoReq('A', 'D', 20, 'helmet')
        helmet.onload = on_load
        helmet.onunload = on_unload

        heart = CargoReq('C', 'A', 40, 'heart')
        heart.onload = on_load
        heart.onunload = on_unload

        # Setup Plan
        Jarvis.reset_scheduler()
        #         when  event     called_with_params
        Jarvis.plan(10, add_load, (c,helmet))
        Jarvis.plan(20, add_load, (c,heart))
        
        # Exercise + Verify indirect output
        #   SUT is the Cart.
        #   Exercise means calling Cart.request in different time periods.
        #   Requests are called by add_load (via plan and its scheduler).
        #   Here, we run the plan.
        Jarvis.run()

        # Verify direct output
        log(cart_dev)
        # vylozi material true
        self.assertEqual('unloaded', helmet.context)
        self.assertEqual('unloaded', heart.context)
        self.assertTrue(cart_dev.empty())


    def test_two(self):
        "Test ID 2"

        def add_load(c: CartCtl, cargo_req: CargoReq):
            log('%d: Requesting %s at %s' % (Jarvis.time(), cargo_req, cargo_req.src))
            c.request(cargo_req)

        def on_move(c: Cart):
            log('%d: Cart is moving %s->%s' % (Jarvis.time(), c.pos, c.data))

        def on_load(c: Cart, cargo_req: CargoReq, ctrl: CartCtl):
            log('%d: Cart at %s: loading: %s' % (Jarvis.time(), c.pos, cargo_req))
            log(c)
            cargo_req.context = "loaded"
            # prioritny material True
            self.assertEqual(cargo_req.prio, True)
            # vozik v rezime iba vykladka True
            self.assertEqual(ctrl.status, Status.UnloadOnly)
            # vynimka false
            self.assertLess(Jarvis.time() - cargo_req.born, 120)

        def on_unload(c: Cart, cargo_req: CargoReq, ctrl: CartCtl):
            log('%d: Cart at %s: unloading: %s' % (Jarvis.time(), c.pos, cargo_req))
            log(c)
            cargo_req.context = 'unloaded'
            if cargo_req.content == 'helmet':
                self.assertEqual('D', c.pos)
            if cargo_req.content == 'pig':
                self.assertEqual('B', c.pos)

        # Setup Cart
        # 4 slots, 150 kg max payload capacity, 2=max debug
        cart_dev = Cart(4, 150, 0)
        cart_dev.onmove = on_move

        # Setup Cart Controller
        c = CartCtl(cart_dev, Jarvis)

        # Setup Cargo to move
        helmet = CargoReq('A', 'D', 20, 'helmet')
        helmet.onunload = on_unload

        pig = CargoReq('C', 'B', 130, 'pig')
        pig.onunload = on_unload

        # prioritny material
        heart = CargoReq('A', 'B', 20, 'heart')
        heart.onload = on_load

        # Setup Plan
        Jarvis.reset_scheduler()
        #         when  event     called_with_params
        Jarvis.plan(10, add_load, (c,helmet))
        Jarvis.plan(20, add_load, (c,heart))
        Jarvis.plan(40, add_load, (c,pig))

        # Exercise + Verify indirect output
        #   SUT is the Cart.
        #   Exercise means calling Cart.request in different time periods.
        #   Requests are called by add_load (via plan and its scheduler).
        #   Here, we run the plan.
        Jarvis.run()
        
        # Verify direct output
        log(cart_dev)
        
        # nalozi prioritny material
        self.assertEqual('loaded', heart.context)
        # vylozi material pri plnej kapacite true
        self.assertEqual('unloaded', helmet.context)
        # vylozi material pri volnej kapacite true
        self.assertEqual('unloaded', pig.context)
        self.assertTrue(cart_dev.empty())

    def test_three(self):
        "Test ID 3"

        def add_load(c: CartCtl, cargo_req: CargoReq):
            log('%d: Requesting %s at %s' % (Jarvis.time(), cargo_req, cargo_req.src))
            c.request(cargo_req)

        def on_move(c: Cart):
            log('%d: Cart is moving %s->%s' % (Jarvis.time(), c.pos, c.data))

        def on_load(c: Cart, cargo_req: CargoReq, ctrl: CartCtl):
            log('%d: Cart at %s: loading: %s' % (Jarvis.time(), c.pos, cargo_req))
            log(c)
            cargo_req.context = "loaded"
            # vynimka false
            self.assertLess(Jarvis.time() - cargo_req.born, 120)

            if cargo_req.content != 'heart':
                self.assertEqual(ctrl.status, Status.Normal)
            if cargo_req.content == 'heart':
                self.assertNotEqual(c.get_free_slot(), -1) 
                self.assertEqual(ctrl.status, Status.UnloadOnly)
                self.assertEqual(cargo_req.prio, True)

        def on_unload(c: Cart, cargo_req: CargoReq, ctrl: CartCtl):
            log('%d: Cart at %s: unloading: %s' % (Jarvis.time(), c.pos, cargo_req))
            log(c)
            cargo_req.context = 'unloaded'
            if cargo_req.content == 'helmet':
                self.assertEqual(ctrl.status, Status.Normal)
            if cargo_req.content == 'pig':
                self.assertEqual(ctrl.status, Status.Normal)
            if cargo_req.content == 'pen':
                self.assertEqual(ctrl.status, Status.Normal)
            if cargo_req.content == 'heart':
                self.assertEqual(ctrl.status, Status.Normal)

        # Setup Cart
        # 4 slots, 150 kg max payload capacity, 2=max debug
        cart_dev = Cart(4, 150, 0)
        cart_dev.onmove = on_move

        # Setup Cart Controller
        c = CartCtl(cart_dev, Jarvis)

        # Setup Cargo to move
        helmet = CargoReq('A', 'D', 20, 'helmet')
        helmet.onload = on_load
        helmet.onunload = on_unload

        pig = CargoReq('D', 'B', 30, 'pig')
        pig.onload = on_load
        pig.onunload = on_unload

        pen = CargoReq('C', 'B', 10, 'pen')
        pen.onload = on_load
        pen.onunload = on_unload

        # prioritny material
        heart = CargoReq('A', 'B', 20, 'heart')
        heart.onload = on_load
        heart.onunload = on_unload

        # Setup Plan
        Jarvis.reset_scheduler()
        #         when  event     called_with_params
        Jarvis.plan(10, add_load, (c,helmet))
        Jarvis.plan(13, add_load, (c,heart))
        Jarvis.plan(40, add_load, (c,pig))
        Jarvis.plan(25, add_load, (c,pen))

        # Exercise + Verify indirect output
        #   SUT is the Cart.
        #   Exercise means calling Cart.request in different time periods.
        #   Requests are called by add_load (via plan and its scheduler).
        #   Here, we run the plan.
        Jarvis.run()
        
        # Verify direct output
        log(cart_dev)

    def test_four(self):
        "Test ID 4"

        def add_load(c: CartCtl, cargo_req: CargoReq):
            log('%d: Requesting %s at %s' % (Jarvis.time(), cargo_req, cargo_req.src))
            c.request(cargo_req)

        def on_move(c: Cart):
            log('%d: Cart is moving %s->%s' % (Jarvis.time(), c.pos, c.data))

        def on_load(c: Cart, cargo_req: CargoReq, ctrl: CartCtl):
            log('%d: Cart at %s: loading: %s' % (Jarvis.time(), c.pos, cargo_req))
            log(c)
            self.assertEqual(ctrl.status, Status.Normal)
            self.assertEqual(c.load_capacity, cargo_req.weight)

        def on_unload(c: Cart, cargo_req: CargoReq, ctrl: CartCtl):
            log('%d: Cart at %s: unloading: %s' % (Jarvis.time(), c.pos, cargo_req))
            log(c)
            if cargo_req.content == 'helmet':
                self.assertEqual(ctrl.status, Status.Normal)
            if cargo_req.content == 'pig':
                self.assertEqual(ctrl.status, Status.Normal)

        # Setup Cart
        # 4 slots, 150 kg max payload capacity, 2=max debug
        cart_dev = Cart(3, 50, 0)
        cart_dev.onmove = on_move

        # Setup Cart Controller
        c = CartCtl(cart_dev, Jarvis)

        # Setup Cargo to move
        helmet = CargoReq('A', 'B', 50, 'helmet')
        helmet.onload = on_load
        helmet.onunload = on_unload

        pig = CargoReq('C', 'D', 20, 'pig')
        pig.onunload = on_unload

        # Setup Plan
        Jarvis.reset_scheduler()
        #         when  event     called_with_params
        Jarvis.plan(10, add_load, (c,helmet))
        Jarvis.plan(22, add_load, (c,pig))

        # Exercise + Verify indirect output
        #   SUT is the Cart.
        #   Exercise means calling Cart.request in different time periods.
        #   Requests are called by add_load (via plan and its scheduler).
        #   Here, we run the plan.
        Jarvis.run()
        
        # Verify direct output
        log(cart_dev)

    def test_five(self):
        "Test ID 5"

        def add_load(c: CartCtl, cargo_req: CargoReq):
            log('%d: Requesting %s at %s' % (Jarvis.time(), cargo_req, cargo_req.src))
            c.request(cargo_req)

        def on_move(c: Cart):
            log('%d: Cart is moving %s->%s' % (Jarvis.time(), c.pos, c.data))

        def on_load(c: Cart, cargo_req: CargoReq, ctrl: CartCtl):
            log('%d: Cart at %s: loading: %s' % (Jarvis.time(), c.pos, cargo_req))
            log(c)
            
            cargo_req.context = 'loaded'
            if cargo_req.content == 'pen':
                # po nalozeni pera musi byt rezim iba vykladka co sposobi ze 'pig'
                # sa nestihne spracovat do 120 sekund od poziadavky
                self.assertEqual(ctrl.status, Status.UnloadOnly)

        def on_unload(c: Cart, cargo_req: CargoReq, ctrl: CartCtl):
            log('%d: Cart at %s: unloading: %s' % (Jarvis.time(), c.pos, cargo_req))
            log(c)

        # Setup Cart
        # 4 slots, 150 kg max payload capacity, 2=max debug
        cart_dev = Cart(4, 150, 0)
        cart_dev.onmove = on_move

        # Setup Cart Controller
        c = CartCtl(cart_dev, Jarvis)

        # Setup Cargo to move
        helmet = CargoReq('A', 'D', 20, 'helmet')
        helmet.onload = on_load
      
        # prioritny material
        pig = CargoReq('A', 'C', 30, 'pig')
        pig.onload = on_load

        # prioritny material
        pen = CargoReq('D', 'C', 10, 'pen')
        pen.onload = on_load

        heart = CargoReq('B', 'C', 20, 'heart')
        heart.onload = on_load

        pencil = CargoReq('C', 'A', 20, 'pencil')
        pencil.onload = on_load

        # Setup Plan
        Jarvis.reset_scheduler()
        #         when  event     called_with_params
        Jarvis.plan(10, add_load, (c,helmet))
        Jarvis.plan(13, add_load, (c,pig))
        Jarvis.plan(14, add_load, (c,pen))
        Jarvis.plan(20, add_load, (c,heart))
        Jarvis.plan(21, add_load, (c,pencil))

        # Exercise + Verify indirect output
        #   SUT is the Cart.
        #   Exercise means calling Cart.request in different time periods.
        #   Requests are called by add_load (via plan and its scheduler).
        #   Here, we run the plan.
        Jarvis.run()
        
        # Verify direct output
        log(cart_dev)
        
        # nenalozenie z dovodu vyprsania casu vynimka X1
        self.assertEqual(pig.context, None)

    def test_six(self):
        "Test ID 6"

        def add_load(c: CartCtl, cargo_req: CargoReq):
            log('%d: Requesting %s at %s' % (Jarvis.time(), cargo_req, cargo_req.src))
            c.request(cargo_req)

        def on_move(c: Cart):
            log('%d: Cart is moving %s->%s' % (Jarvis.time(), c.pos, c.data))
            if (Jarvis.time() == 88):
                # vozik is full co sposobi ze prioritna poziadavka sa nenalozi
                # a potom sa nestihne spracovat do 120 sekund
                self.assertEqual(c.load_capacity, 50)

        def on_load(c: Cart, cargo_req: CargoReq, ctrl: CartCtl):
            log('%d: Cart at %s: loading: %s' % (Jarvis.time(), c.pos, cargo_req))
            log(c)
            cargo_req.context = 'loaded'

        def on_unload(c: Cart, cargo_req: CargoReq, ctrl: CartCtl):
            log('%d: Cart at %s: unloading: %s' % (Jarvis.time(), c.pos, cargo_req))
            log(c)

        # Setup Cart
        # 4 slots, 150 kg max payload capacity, 2=max debug
        cart_dev = Cart(2, 50, 0)
        cart_dev.onmove = on_move

        # Setup Cart Controller
        c = CartCtl(cart_dev, Jarvis)

        # Setup Cargo to move
        helmet = CargoReq('A', 'C', 20, 'helmet')
        helmet.onload = on_load
        helmet.onunload = on_unload
        
        # prioritny material
        pig = CargoReq('A', 'C', 30, 'pig')
        pig.onload = on_load
        pig.onunload = on_unload

        pen = CargoReq('C', 'B', 30, 'pen')
        pen.onload = on_load
        pen.onunload = on_unload

        heart = CargoReq('C', 'B', 20, 'heart')
        heart.onload = on_load
        heart.onunload = on_unload

        # Setup Plan
        Jarvis.reset_scheduler()
        #         when  event     called_with_params
        Jarvis.plan(10, add_load, (c,helmet))
        Jarvis.plan(13, add_load, (c,pig))
        Jarvis.plan(40, add_load, (c,pen))
        Jarvis.plan(20, add_load, (c,heart))

        # Exercise + Verify indirect output
        #   SUT is the Cart.
        #   Exercise means calling Cart.request in different time periods.
        #   Requests are called by add_load (via plan and its scheduler).
        #   Here, we run the plan.
        Jarvis.run()
        
        # Verify direct output
        log(cart_dev)
        
        # nenalozenie z dovodu vyprsania casu vynimka X1
        self.assertEqual(pig.context, None)

if __name__ == "__main__":
    unittest.main()
