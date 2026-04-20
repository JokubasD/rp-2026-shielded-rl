from simulator import Simulator

def main():
    sim = Simulator(3, 3)
    print(sim.ground_truth.traversability.matrix)


if __name__ == "__main__":
    main()