import configparser
import sim


def parseConfig(argPath):

    config = configparser.ConfigParser()
    config.read(argPath, encoding='utf-8')

    print("===== Configurations =====")

    for section in config.sections():
        print(f"[{section}]")

        for option in config.options(section):
            value = config.get(section, option)
            print(f"{option} = {value}")

        print()

    print("==========================")
    print()

    return config


if (__name__ == '__main__'):

    config = parseConfig('config.ini')

    hopSim = sim.HopSim(config)

    hopSim.run()

    hopSim.visualize()
    #hopSim.checkNodeType()
