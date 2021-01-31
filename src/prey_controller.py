from prey import *
from controller import Controller

class prey_controller_nn(Prey):
    def __init__(self, gene, robot):
        self.gene = gene
        super(prey_controller_nn, self).__init__(robot)

    def run(self):
        c = Controller(self.gene, 8, 2, 2)

        while not self.stopped():
            sensors = self._robot.read_irs()
            sensors = [sensors[i] * 5 if sensors[i] is not False else 1 for i in range(len(sensors))]
            left, right = c.act(sensors)
            self._robot.move(left, right, millis=200)
