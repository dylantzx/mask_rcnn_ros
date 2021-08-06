import time

class FPS:

    def __init__(self):
        self.t1 = 0
        self.t2 = 0
        self.fps_list = []
        self.fps = 0

    def start(self):
        self.t1 = time.time()

    def stop(self):
        self.t2 = time.time()
        self.update()

    def update(self):
        self.fps_list.append(self.t2 - self.t1)
        self.fps_list = self.fps_list[-20:]
        ms = sum(self.fps_list)/len(self.fps_list)*1000
        self.fps = 1000 / ms

    def elapsed(self):
        return (self.t2 - self.t1) * 1000

    def getFPS(self):
        return self.fps